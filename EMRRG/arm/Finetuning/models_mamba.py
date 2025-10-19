import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math
from mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None





class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
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
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
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
        self.mlp = SwiGLU(dim, dim*4*2//3, subln=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, segmentation: Optional[Tensor] = None, inference_params=None
    ):
        if segmentation is None:
            hidden_states = hidden_states + self.drop_path(self.mixer(self.norm1(hidden_states), inference_params=inference_params))
            hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
            return hidden_states
        else:
            residual_hidden_states = hidden_states
            residual_segmentation = segmentation
            features = self.mixer(self.norm1(hidden_states), segmenttation_features=self.norm1(segmentation), inference_params=inference_params)
            hidden_states = features[0] + residual_hidden_states
            segmentation = features[1] + residual_segmentation
            hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
            segmentation = segmentation + self.drop_path(self.mlp(self.norm2(segmentation)))
            return hidden_states, segmentation
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
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, expand=1, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
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


class ARM(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride=16,
                 depth=24, 
                 embed_dim=192, 
                 channels=3, 
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
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,
                 bimamba_type="none",
                 if_cls_token=False,
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,
                 use_middle_cls_token=False,
                 global_pool=False,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.global_pool=global_pool
        # pretrain parameters

        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
            
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
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
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = nn.LayerNorm(embed_dim)

        # self.pre_logits = nn.Identity()

        # original init
        self.patch_embed.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


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

    def forward_features(self, x, segmentation=None, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B, M, _ = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        token_position = M // 2

        if segmentation is not None:
            mask  = torch.zeros_like(x)
            for i in range(len(segmentation)):
                patches = segmentation[i]
                mask[i, patches, :] = 1
            x_masked = x * mask  # Apply the mask to input
            x_masked = torch.cat((x_masked[:, :token_position, :], cls_token, x_masked[:, token_position:, :]), dim=1)
            x_masked = x_masked + self.pos_embed
            x_masked = self.pos_drop(x_masked)
            segmentation = x_masked

        # add cls token in the middle
        x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # mamba impl
        hidden_states = x
        if segmentation is None:
            for layer in self.layers:
                hidden_states = layer(hidden_states, inference_params=inference_params)
        else:
            for layer in self.layers:
                hidden_states, segmentation = layer(hidden_states, segmentation=segmentation, inference_params=inference_params)

        hidden_states = self.norm_f(hidden_states)

        return hidden_states

    def forward(self, x, segmentation=None, return_features=False, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.patch_embed(x)
        fetures = self.forward_features(x, segmentation,inference_params, if_random_cls_token_position=if_random_cls_token_position, if_random_token_rank=if_random_token_rank)
        return fetures



@register_model
def arm_base_pz16(type,pretrained=False, **kwargs):
    model = ARM(
        patch_size=16, embed_dim=768, depth=12, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v3", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def arm_large_pz16(type,pretrained=False, **kwargs):
    model = ARM(
        patch_size=16, embed_dim=1024, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v3", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def arm_huge_pz16(type,pretrained=False, **kwargs):
    model = ARM(
        patch_size=16, embed_dim=1536, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v3", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



