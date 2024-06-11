
# Copyright (c) ByteDance, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# MAE: https://github.com/facebookresearch/mae 
# --------------------------------------------------------

from functools import partial
import torch
import numpy as np
import math
import random
import torch.nn as nn

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from pos_embed import get_2d_sincos_pos_embed
from patch_embed import SmallPatchEmbed


class MaskedAutoencoderViT(nn.Module):#基于vit实现的mae
    """ Masked Autoencoder with VisionTransformer backbone
    img_size：输入图像宽和高。patch_size：每一个patch的宽和高。in_chans：输入通道数。
    embed_dim：mse编码器的Hidden size。depth：mae中transform的块数Layers的层数。num_heads编码器的头数
    解码器的三个参数
    编码器的输出需要降维
    """

    def __init__(self, img_size=1280, patch_size=64, in_chans=1,
                 embed_dim=768, depth=12, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,mask_ratio=0.75,use_learnable_pos_emb=True,new_depth = 6):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed=SmallPatchEmbed(1,1024,1024)
        #num_patches：得到块的数量
        num_patches = self.patch_embed.num_patches
        #实例化两个参数cls_token和pos_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))#可训练的
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding不可训练的，位置编码requires_grad不可训练
        #定义编码器的transform blocks，使用Module的列表，不能使用普通列表，不然无法被Module识别
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)#多头注意力头数num_heads,, qk_scale=None
            for i in range(depth)])#最新版timm要注释掉qk_scale

        self.norm = norm_layer(embed_dim)#对encode output做归一化
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)#构建一个线性映射层，将编码器的输出映射到解码器的特征维度上
        
        self.apply(self._init_weights)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))#可训练的token，用于替换掉那些被mask的块

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        #定义解码器的transform blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)#qk_scale=None,
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)#对decoder output做归一化
        #patch_size**2 * in_chans，patch的面积乘上通道数，映射层

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.decoder_image = nn.Linear(196, 1, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss#是否要对像素做归一化再去算loss
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.initialize_weights()
    #初始化权重
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding对pos_embed进行初始化
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        #对decode的pos_embed进行初始化
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)均匀分布的初始化
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)高斯分布的初始化
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)#接受的参数是一个函数，函数会作用在当前这个Module和当前这个Module的子Module

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):#如果进入这个模块的是nn.Linear的实例，则会对权重做均匀分布的初始化
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:#如果有bias则做一个bias为0的初始化
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):#对层归一化逻辑进行判断，如果是则将权重和偏置做常数初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):#把图片划分成块
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)     patch_size**2 *3:单张图像的像素点个数,L:为图像尺寸，N为batch的大小
        """
        p = self.patch_embed.patch_size[0]  #p=patch_size的大小
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0    #若img图像尺寸相对于patch_size不整除则跳出

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))#batch_size*patch的数目*patch的大小
        return x

    def unpatchify(self, x):#把块的图片还原成图片
        """
        x: (N, L, patch_size**2 *3) #
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0],1, h * p, h * p))
        return imgs
    
    def random_masking(self, x, mask_ratio):#随机掩码
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))#算出保留块的数目
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]生成随机矩阵，均匀分布 batch_size*length
        
        # sort noise for each sample 对每个样本的噪声进行排序
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove排序从小到大取掩码和被掩码时用的
        ids_restore = torch.argsort(ids_shuffle, dim=1)#对索引进行排序，还原序列时要用到的

        # keep the first subset 保留第一个子集
        ids_keep = ids_shuffle[:, :len_keep]#
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))#dim:维度。未被掩码的序列

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)#batch_size*lenth大小的提供给decode使用的全一矩阵
        mask[:, :len_keep] = 0#对未掩码设置为0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)#在原图中被掩码的位置

        return x_masked, mask, ids_restore

    def random_masking_yiliao(self, x, mask_ratio_outer,mask_ratio_iner):#随机掩码
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        #创建两个mask矩阵，分别为外部区域和内部区域，两个相加为最终mask矩阵

        N, L, D = x.shape  # batch, length, dim
        L_new = int(math.sqrt(L))
        patch_iner = int(L_new*0.625*L_new*0.5)
        patch_outer = L - patch_iner
        patch_label = torch.zeros((L_new, L_new)) #区域标签
        #定义mask区域，可修改，目前是内部一个矩形，需要mask的块置一
        patch_label[int(L_new*0.25)+1:int(L_new*0.75)+1, int(L_new*0.125)+1:int(L_new*0.75)+1] = 1

        patch_label_new = patch_label.flatten()
        indices_out = torch.empty(0)  
        indices_in = torch.empty(0)
        patch_iner_new = 0
        patch_outer_new = 0
        for i, value in enumerate(patch_label_new):  
            if value == 0:  
                indices_out = torch.cat((indices_out, torch.tensor([i]))) 
                patch_outer_new += 1 
            else:  
                indices_in = torch.cat((indices_in, torch.tensor([i])))
                patch_iner_new += 1
        
        len_keep_outer = int(patch_outer_new * (1 - mask_ratio_outer))
        len_keep_iner = int(patch_iner_new * (1 - mask_ratio_iner))
                
        noise_outer = torch.rand(N, patch_outer_new,device=x.device)  # noise in [0, 1]生成随机矩阵，均匀分布 batch_size*length
        noise_iner = torch.rand(N, patch_iner_new,device=x.device)  # noise in [0, 1]生成随机矩阵，均匀分布 batch_size*length
                
        # sort noise for each sample 对每个样本的噪声进行排序
        ids_shuffle_outer = torch.argsort(noise_outer, dim=1)  # 对随机向量进行排序，输出索引
        ids_shuffle_outer_new = ids_shuffle_outer.clone()

        for i, ids in enumerate(ids_shuffle_outer):
            ids_shuffle_outer_new[i] = indices_out[ids]
        #ids_restore_outer = torch.argsort(ids_shuffle_outer, dim=1)#对索引进行排序，还原序列时要用到的

        ids_shuffle_iner = torch.argsort(noise_iner, dim=1)  # ascend: small is keep, large is remove排序从小到大取掩码和被掩码时用的
        ids_shuffle_iner_new = ids_shuffle_iner.clone()

        for i, ids in enumerate(ids_shuffle_iner):
            ids_shuffle_iner_new[i] = indices_in[ids]

        #ids_restore_iner = torch.argsort(ids_shuffle_iner, dim=1)#对索引进行排序，还原序列时要用到的  


        ids_keep_out = ids_shuffle_outer_new[:, :len_keep_outer]
        ids_keep_in = ids_shuffle_iner_new[:, :len_keep_iner]
        ids_keep = torch.cat((ids_keep_out,ids_keep_in),dim=1)
        ids_nokeep_out = ids_shuffle_outer_new[:, len_keep_outer:]
        ids_nokeep_in = ids_shuffle_iner_new[:, len_keep_iner:]
        ids_nokeep = torch.cat((ids_nokeep_out,ids_nokeep_in),dim=1)
        ids_shuffle = torch.cat((ids_keep,ids_nokeep),dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))#dim:维度。未被掩码的序列

        mask = torch.ones([N, L], device=x.device)#batch_size*lenth大小的提供给decode使用的全一矩阵
        mask[:, :len_keep_outer+len_keep_iner] = 0#对掩码设置为0

        mask = torch.gather(mask, dim=1, index=ids_restore)#在原图中被掩码的位置

        return x_masked, mask, ids_restore

    def forward_encoder(self, x,mask_type, mask_ratio_outer,mask_ratio_iner):#x：输入图像。mask_ratio：掩码比例
        # embed patches

        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        if mask_type == 1:
            x, mask, ids_restore = self.random_masking_yiliao(x, mask_ratio_outer,mask_ratio_iner)#随机掩码ids_restore：恢复的索引
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio_outer)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)#encode的输入

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        #for new_blk in self.new_blocks :
        #    x = new_blk(x)
        x = self.norm(x)#encode的输出

        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)#扩维
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token，拼接
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle，得到被还原后的顺序
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed加上位置编码
        x = x + self.decoder_pos_embed


        # apply Transformer blocks对decode的所有block进行递归的调用
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)#得到decode输出

        tezheng = x

        x = self.decoder_pred(x)

        # remove cls token移除cls token
        x = x[:, 1:, :]

        return x ,tezheng

    def forward_loss(self, imgs, pred, mask):#平方差loss
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)    #原始图像patch后
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)#计算均值
            var = target.var(dim=-1, keepdim=True)#计算方差
            target = (target - mean) / (var + 1.e-6)**.5#对target做均值方差的归一化，1.e-6：防止方差为0

        loss = (pred - target) ** 2 #计算均方损失函数
        loss = loss.mean(dim=-1)     # [N, L], mean loss per patch，均值
        #loss = (loss * mask).sum() / mask.sum()  # 将每个mask后的patch累加

        return loss

    def load_param_pth(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
        for k, v in param_dict.items():
            
            if 'head' in k or 'dist' in k:
                continue
            if 'mask_token'in k:
                continue
            if 'decoder'in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))

    def resize_pos_embed(posemb, posemb_new, hight, width):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        ntok_new = posemb_new.shape[1]

        posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1

        gs_old = int(math.sqrt(len(posemb_grid)))
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
        posemb = torch.cat([posemb_token, posemb_grid], dim=1)
        return posemb

    def del_tensor_ele_n(self,arr, index, n):
        """
        arr: 输入tensor
        index: 需要删除位置的索引
        n: 从index开始，需要删除的行数
        """
        arr1 = arr[0:index]
        arr2 = arr[index+n:]
        return torch.cat((arr1,arr2),dim=0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}
    
    def forward(self, imgs, mask_type,mask_ratio_outer,mask_ratio_iner):#结合返回

        latent, mask, ids_restore = self.forward_encoder(imgs,mask_type, mask_ratio_outer,mask_ratio_iner)
        pred,tezheng = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        loss= self.forward_loss(imgs, pred, mask)
        return loss,mask

#三种mae，mae_vit_base，mae_vit_large，mae_vit_huge
def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=64, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512,  decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=64, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=64, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks







