import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import to_2tuple
from timm.models.vision_transformer import _load_weights
from einops import rearrange
import math
from utils.pos_embed import *
from collections import namedtuple

from mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

import random

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import torch.nn as nn
from timm.models.layers import DropPath



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask):
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn += mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self. attn2 = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_1 = norm_layer(dim)
        self.norm2_2 = norm_layer(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv, mask):
        q = q + self.attn2(self.norm2_1(q), self.norm2_2(kv), mask)
        q = q + self.mlp(self.norm2(q))
        return q



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.mlp = SwiGLU(dim, dim * 4 * 2 // 3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        hidden_states = hidden_states + self.drop_path(
            self.mixer(self.norm1(hidden_states), inference_params=inference_params))
        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))

        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, expand=1, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 stride=16,
                 depth=24,
                 embed_dim=192,
                 dec_embed_dim=192,
                 channels=3,
                 num_classes=1000,
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 if_bidirectional=False,
                 if_abs_pos_embed=False,
                 bimamba_type="none",
                 if_devide_out=False,
                 init_layer_scale=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.if_abs_pos_embed = if_abs_pos_embed
        if depth==12:
            self.skip = [6, 8, 10, 12]
        elif depth==24:
            self.skip = [12, 16, 20, 24]

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim), requires_grad=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    drop_path=0.,
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.dec_embed_dim = dec_embed_dim
        self.ar_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.dec_embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.enc2dec = nn.Linear(embed_dim * 4, self.dec_embed_dim * 4)
        self.dec_block = nn.ModuleList([
            DecoderBlock(self.dec_embed_dim, self.dec_embed_dim // 64, 4, qkv_bias=True, qk_scale=None,
                         norm_layer=nn.LayerNorm)
            for i in range(4)])
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.norm_3 = nn.LayerNorm(embed_dim)
        self.norm_4 = nn.LayerNorm(embed_dim)
        self.ar_norm = nn.LayerNorm(self.dec_embed_dim)
        self.ar_pred = nn.Linear(self.dec_embed_dim, patch_size ** 2 * 3)


        # original init
        self.patch_embed.apply(segm_init_weights)
        if if_abs_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            dec_pos_embed = get_2d_sincos_pos_embed(self.dec_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=False)

            self.dec_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
        trunc_normal_(self.ar_token, std=.02)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.dec_block.apply(self.atten_init_weights)
        # Arbitrary size
        assert patch_size==stride
        self.cluster_num = ((img_size//patch_size)//4)*((img_size//patch_size)//4)-1
        self.register_buffer("mask", self.mask_generate(self.cluster_num, 16))
    def mask_generate(self, segment, tokens_per_segment):
        mask = torch.tril(torch.ones((segment, segment), dtype=torch.float))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=0)
        mask = torch.repeat_interleave(mask, repeats=tokens_per_segment, dim=1)
        return mask
    def atten_init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        
        x = self.patch_embed(x)
        B, N, C = x.shape
        x = x + self.pos_embed
        x = self.pos_drop(x)
        H,W = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.reshape(B, H, W, C)
        
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=4, p2=4)
        # avoid information leak
        # x = torch.cat([x, x[:, :, -2:]], dim=2)
        # mamba impl
        hidden_states = x[:, :-1].reshape(B, -1, C)
        features = []
        count=0
        for layer in self.layers:

            hidden_states = layer(
                hidden_states, inference_params=inference_params
            )
            count += 1
            if count in self.skip:
                features.append(hidden_states)
        features = [self.norm_1(features[0]), self.norm_2(features[1]),
                    self.norm_3(features[2]),self.norm_4(features[3])]
        features = self.enc2dec(torch.cat(features, dim=-1))
        B, N, C = features.shape
        features = features.reshape(B, self.cluster_num, N//self.cluster_num, C)
        # avoid information leak
        # features = features[:, :, :-2]
        features = features.reshape(B, -1, C)
        B, N, C = features.shape
        assert N==16*self.cluster_num
        return features.reshape(B, N, C//4, 4)

    def forward_decoder(self, latent_ar, decoder_pos_embed):
        # embed tokens
        B, N, C, depth = latent_ar.shape
        
        ar_token = self.ar_token + decoder_pos_embed
        H, W = int(np.sqrt(ar_token.shape[1])), int(np.sqrt(ar_token.shape[1]))
        ar_token = ar_token.reshape(ar_token.shape[0], H, W, C)
        ar_token = rearrange(ar_token, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=4, p2=4)
        ar_token = ar_token[:, 1:].reshape(1, -1, C)
        ar_token = ar_token.repeat(B,1,1)
        # apply Transformer blocks
        count = 0
        for blk in self.dec_block:
            ar_token = blk(ar_token, latent_ar[:, :, :, count],self.mask)
            count += 1
        ar_token = self.ar_norm(ar_token)
        ar_token = self.ar_pred(ar_token)
        return ar_token

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def forward_loss(self, imgs, pred):
        target = self.patchify(imgs)
        # 计算均值和方差
        if True:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        B, N, C = target.shape
        H, W = int(np.sqrt(N)), int(np.sqrt(N))
        target = target.reshape(B, H, W, C)
        target = rearrange(target, "b (h p1) (w p2) c -> b (h w) (p1 p2) c", p1=4, p2=4)
        target = target[:, 1:].reshape(B, -1, C)
        loss = (pred - target) ** 2
        return loss

    def forward(self, x, inference_params=None):
        labels = x.clone()
        x = self.forward_features(x, inference_params)
        x = self.forward_decoder(x, self.dec_pos_embed)
        loss = self.forward_loss(labels, x)
        return loss.mean(-1).mean(0)

@register_model
def arm_base_pz16(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, img_size=192, embed_dim=768, depth=12, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="None", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def arm_large_pz16(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, img_size=192, embed_dim=1024, depth=24, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="None", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def arm_huge_pz16(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, stride=16, img_size=192, embed_dim=1536, depth=24, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="None", **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def arm_base_pz16_1280(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=64, stride=64, img_size=1280, embed_dim=768, depth=12, dec_embed_dim=512, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
        if_abs_pos_embed=True, bimamba_type="None", **kwargs)
    model.default_cfg = _cfg()
    return model
