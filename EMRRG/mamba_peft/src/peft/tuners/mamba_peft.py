# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
from typing import Any, List

import torch
from torch import nn


import torch
from torch import nn
import math
from typing import Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import sys, os
from einops import rearrange, repeat
import math
from dataclasses import asdict, dataclass, field
from enum import Enum

from ..utils import PeftConfig, PeftType

from transformers.models.mamba.modeling_mamba import MambaMixer, MambaBlock, MambaModel, is_fast_path_available, MambaOutput, MambaCache

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    causal_conv1d_update, causal_conv1d_fn = None, None




@dataclass
class MambaPEFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """
    # Adaptformer
    adaptformer: bool = field(default=False,) 
    dim_adaptf: int = field(default=32,) 
    s_adaptf: float = field(default=1.,) 
    bit_adaptf: int = field(default=32,) 
    train_adaptformer: bool = field(default=True,) 
    # LoRA config out_proj (AdaptFormer)
    lora_out_proj: bool = field(default=False,) 
    dim: int = field(default=32,) 
    s: float = field(default=1.,) 
    bit: int = field(default=32,) 
    train_lora_out_proj: bool = field(default=True,)
    # LoRA config in_proj (X and Z),
    lora_in_proj: bool = field(default=False,) 
    dim_in_proj: int = field(default=32,) 
    s_in_proj: float = field(default=1.,) 
    bit_in_proj: int = field(default=32,) 
    train_lora_in_proj: bool = field(default=True,)
    # LoRA config X,
    lora_X: bool = field(default=False,) 
    dim_X: int = field(default=32,) 
    s_X: float = field(default=1.,) 
    bit_X: int = field(default=32,) 
    train_lora_X: bool = field(default=True,)
    # LoRA confg Z
    lora_Z: bool = field(default=False,) 
    dim_Z: int = field(default=32,) 
    s_Z: float = field(default=1.,) 
    bit_Z: int = field(default=32,) 
    train_lora_Z: bool = field(default=True,)
    # LoRA config x_proj (d, B, and C),
    lora_x_proj: bool = field(default=False,) 
    dim_x_proj: int = field(default=4,) 
    s_x_proj: float = field(default=1.,) 
    bit_x_proj: int = field(default=32,) 
    train_lora_x_proj: bool = field(default=True,)
    # LoRA config d,
    lora_d: bool = field(default=False,) 
    dim_d: int = field(default=4,) 
    s_d: float = field(default=1.,) 
    bit_d: int = field(default=32,) 
    train_lora_d: bool = field(default=True,)
    # LoRA config B,
    lora_B: bool = field(default=False,) 
    dim_B: int = field(default=4,) 
    s_B: float = field(default=1.,) 
    bit_B: int = field(default=32,) 
    train_lora_B: bool = field(default=True,)
    # LoRA config C,
    lora_C: bool = field(default=False,) 
    dim_C: int = field(default=4,) 
    s_C: float = field(default=1.,) 
    bit_C: int = field(default=32,) 
    train_lora_C: bool = field(default=True,)
    # LoRA config dt,
    lora_dt: bool = field(default=False,) 
    dim_dt: int = field(default=4,) 
    s_dt: float = field(default=1.,) 
    bit_dt: int = field(default=32,) 
    train_lora_dt: bool = field(default=True,)
    # LoRA config conv1d,
    lora_conv1d: bool = field(default=False,) 
    dim_conv1d: int = field(default=32,) 
    s_conv1d: float = field(default=1.,) 
    bit_conv1d: int = field(default=32,) 
    train_lora_conv1d: bool = field(default=True,)
    # LoRA config patch_embed Conv2d,
    lora_patch_embed: bool = field(default=False,) 
    dim_patch_embed: int = field(default=32,) 
    s_patch_embed: float = field(default=1.,) 
    bit_patch_embed: int = field(default=32,) 
    train_lora_patch_embed: bool = field(default=True,)
    # prefix tuning config
    prefix_tuning: bool = field(default=False,) 
    prefix_type: str = field(default="inner_single_prefix",) 
    prefix_projection: bool = field(default=True,) 
    num_virtual_tokens: int = field(default=1,) 
    encoder_hidden_size: bool = field(default=None,) 
    train_prefix: bool = field(default=True,)
    # prompt tuning config
    prompt_tuning: bool = field(default=False,) 
    prompt_type: str = field(default="prefix",) 
    prompt_projection: bool = field(default=True,) 
    prompt_num_tokens: int = field(default=2,) 
    train_prompt: bool = field(default=True,)
    # MambaならではのPEFTを考えてみた？Scanする数（state_d）を増やすという方向性
    additional_scan: bool = field(default=False,) 
    scan_addition_num: int = field(default=1,) 
    scan_addition_pos: str = field(default="suffix",) 
    scan_A_constant: bool = field(default=None,) 
    scan_A_copy_from_last: bool = field(default=False,) 
    zero_init_x_proj: bool = field(default=False,) 
    train_additional_scan: bool = field(default=True,)
    # Bias tuning like
    learnable_A: bool = field(default=False,) 
    learnable_A_v2: bool = field(default=False,) 
    train_A: bool = field(default=True,)
    learnable_D: bool = field(default=False,) 
    learnable_D_v2: bool = field(default=False,) 
    train_D: bool = field(default=True,)
    learnable_conv1d: bool = field(default=False,) 
    learnable_conv1d_v2: bool = field(default=False,) 
    train_conv1d: bool = field(default=True,)
    learnable_cls_token: bool = field(default=False,) 
    learnable_cls_token_v2: bool = field(default=False,) 
    train_cls_token: bool = field(default=True,)
    learnable_pos_embed: bool = field(default=False,) 
    learnable_pos_embed_v2: bool = field(default=False,) 
    train_pos_embed: bool = field(default=True,)
    learnable_bias: bool = field(default=False,)
    learnable_bias_v2: bool = field(default=False,) 
    train_bias: bool = field(default=True,) 
    
    
    def __post_init__(self):
        self.peft_type = PeftType.MAMBA_PEFT



class MambaPEFTModel(nn.Module):
    """

    Examples:
    """

    def __init__(self, config, model,
                 ):

        super().__init__()

        self.model = model
        self.peft_config = config
        self.adaptformer=config.adaptformer
        self.train_adaptformer=config.train_adaptformer
        self.lora_out_proj=config.lora_out_proj
        self.train_lora_out_proj =config. train_lora_out_proj
        self.lora_in_proj=config.lora_in_proj
        self.train_lora_in_proj =config. train_lora_in_proj
        self.lora_X=config.lora_X
        self.train_lora_X =config. train_lora_X
        self.lora_Z=config.lora_Z
        self.train_lora_Z =config. train_lora_Z
        self.lora_x_proj=config.lora_x_proj
        self.train_lora_x_proj =config. train_lora_x_proj
        self.lora_d=config.lora_d
        self.train_lora_d =config. train_lora_d
        self.lora_B=config.lora_B
        self.train_lora_B =config. train_lora_B
        self.lora_C=config.lora_C
        self.train_lora_C =config. train_lora_C
        self.lora_dt=config.lora_dt
        self.train_lora_dt =config. train_lora_dt
        self.lora_conv1d=config.lora_conv1d
        self.lora_patch_embed=config.lora_patch_embed
        self.train_lora_patch_embed =config. train_lora_patch_embed
        self.train_lora_conv1d =config. train_lora_conv1d
        self.prefix_tuning=config.prefix_tuning
        self.train_prefix =config. train_prefix
        self.prompt_tuning=config.prompt_tuning
        self.train_prompt =config. train_prompt
        self.additional_scan=config.additional_scan
        self.train_additional_scan =config. train_additional_scan
        self.learnable_A=config.learnable_A
        self.learnable_A_v2=config.learnable_A_v2
        self.train_A =config. train_A
        self.learnable_D=config.learnable_D
        self.learnable_D_v2=config.learnable_D_v2
        self.train_D =config. train_D
        self.learnable_conv1d=config.learnable_conv1d
        self.learnable_conv1d_v2=config.learnable_conv1d_v2
        self.train_conv1d =config. train_conv1d
        self.learnable_cls_token=config.learnable_cls_token
        self.learnable_cls_token_v2=config.learnable_cls_token_v2
        self.train_cls_token=config.train_cls_token
        self.learnable_pos_embed=config.learnable_pos_embed
        self.learnable_pos_embed_v2=config.learnable_pos_embed_v2
        self.train_pos_embed=config.train_pos_embed
        self.learnable_bias=config.learnable_bias
        self.learnable_bias_v2=config.learnable_bias_v2
        self.train_bias=config.train_bias
        set_peft(self,
                    # Adaptformer
                    config.adaptformer, config.dim_adaptf, config.s_adaptf, config.bit_adaptf, 
                    # LoRA config out_proj (AdaptFormer),
                    config.lora_out_proj, config.dim, config.s, config.bit,
                    # LoRA config in_proj
                    config.lora_in_proj, config.dim_in_proj, config.s_in_proj, config.bit_in_proj,
                    # LoRA config X,
                    config.lora_X, config.dim_X, config.s_X, config.bit_X,
                    # LoRA confg Z
                    config.lora_Z, config.dim_Z, config.s_Z, config.bit_Z,
                    # LoRA config x_proj (d, config.B, config.and C),
                    config.lora_x_proj, config.dim_x_proj, config.s_x_proj, config.bit_x_proj,
                    # LoRA config d,
                    config.lora_d, config.dim_d, config.s_d, config.bit_d,
                    # LoRA config B,
                    config.lora_B, config.dim_B, config.s_B, config.bit_B,
                    # LoRA config C,
                    config.lora_C, config.dim_C, config.s_C, config.bit_C,
                    # LoRA config dt,
                    config.lora_dt, config.dim_dt, config.s_dt, config.bit_dt,
                    # LoRA config conv1d,
                    config.lora_conv1d, config.dim_conv1d, config.s_conv1d, config.bit_conv1d,
                    # LoRA config patch_embed Conv2d,
                    config.lora_patch_embed, config.dim_patch_embed, config.s_patch_embed, config.bit_patch_embed,
                    # prefix tuning,
                    config.prefix_tuning, config.prefix_type, config.prefix_projection, config.num_virtual_tokens, config.encoder_hidden_size,
                    # prompt tuning config
                    config.prompt_tuning, config.prompt_type, config.prompt_projection, config.prompt_num_tokens,
                    # additional scan config
                    config.additional_scan, config.scan_addition_num, config.scan_addition_pos, config.scan_A_constant, config.scan_A_copy_from_last, config.zero_init_x_proj,
                    # Aのfinetuning
                    config.learnable_A, config.learnable_A_v2,
                    # Dのfinetuning
                    config.learnable_D, config.learnable_D_v2,
                    # conv1dのfinetuning
                    config.learnable_conv1d, config.learnable_conv1d_v2,
                    # cls_tokenのfinetuning
                    config.learnable_cls_token, config.learnable_cls_token_v2,
                    # pos_embedのfinetuning
                    config.learnable_pos_embed, config.learnable_pos_embed_v2,
                    # biasのfinetuning
                    config.learnable_bias, config.learnable_bias_v2)

        self._set_lora_trainable()
        # self._register_state_dict_hooks()

    def _set_lora_trainable(self):
        """Set only the lora parameters trainable."""
        for n, p in self.named_parameters():
            if 'head' in n and p.requires_grad:
                p.requires_grad = True
            elif 'adaptf' in n and self.train_adaptformer:
                p.requires_grad = True
            elif 'lora_out_proj' in n and self.train_lora_out_proj:
                p.requires_grad = True
            elif 'lora_in_proj' in n and self.train_lora_in_proj:
                p.requires_grad = True
            elif 'lora_X' in n and self.train_lora_X:
                p.requires_grad = True
            elif 'lora_Z' in n and self.train_lora_Z:
                p.requires_grad = True
            elif 'lora_x_proj' in n and self.train_lora_x_proj:
                p.requires_grad = True
            elif 'lora_d' in n and self.train_lora_d:
                p.requires_grad = True
            elif 'lora_B' in n and self.train_lora_B:
                p.requires_grad = True
            elif 'lora_C' in n and self.train_lora_C:
                p.requires_grad = True
            elif 'lora_dt' in n and self.train_lora_dt:
                p.requires_grad = True
            elif 'lora_conv1d' in n and self.train_lora_conv1d:
                p.requires_grad = True
            elif 'lora_patch_embed' in n and self.train_lora_patch_embed:
                p.requires_grad = True
            elif 'prefix_encoder' in n and self.train_prefix:
                p.requires_grad = True
            elif 'prompt_encoder' in n and self.train_prompt:
                p.requires_grad = True
            elif ("A_log" in n.split(".") or "A_b_log" in n.split(".")) and self.learnable_A and self.train_A and (not self.learnable_A_v2):
                p.requires_grad = True
            elif "learnable_A" in n and self.learnable_A and self.learnable_A_v2 and self.train_A:
                p.requires_grad = True
            elif ("D" in n.split(".") or "D_b" in n.split(".")) and self.learnable_D and self.train_D and (not self.learnable_D_v2):
                p.requires_grad = True
            elif "learnable_D" in n and self.learnable_D and self.learnable_D_v2 and self.train_D:
                p.requires_grad = True
            elif ("conv1d" in n.split(".") or "conv1d_b" in n.split(".")) and self.learnable_conv1d and self.train_conv1d and (not self.learnable_conv1d_v2):
                p.requires_grad = True
            elif "learnable_conv1d" in n and self.learnable_conv1d and self.learnable_conv1d_v2 and self.train_conv1d:
                p.requires_grad = True
            elif "cls_token" in n and self.learnable_cls_token and self.train_cls_token and (not self.learnable_cls_token_v2):
                p.requires_grad = True
            elif "learnable_cls_token" in n and self.learnable_cls_token and self.learnable_cls_token_v2 and self.train_cls_token:
                p.requires_grad = True
            elif "pos_embed" in n and self.learnable_pos_embed and self.train_pos_embed and (not self.learnable_pos_embed_v2):
                p.requires_grad = True
            elif "learnable_pos_embed" in n and self.learnable_pos_embed and self.learnable_pos_embed_v2 and self.train_pos_embed:
                p.requires_grad = True
            elif "bias" in n and "dt_proj" in n and self.learnable_bias and self.train_bias and (not self.learnable_bias_v2):
                p.requires_grad = True
            elif "learnable_bias" in n and self.learnable_bias and self.learnable_bias_v2 and self.train_bias:
                p.requires_grad = True
            elif '_scan_addi' in n and self.train_additional_scan:
                p.requires_grad = True
            else:
                p.requires_grad = False
        
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of params: {n_parameters}')


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return self.model.__getattribute__(name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        raise NotImplementedError

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


def mambamixer_cuda_kernels_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **peft_kwargs
):

    if peft_kwargs.get("outer_single_prefix") is not None: # biMamba考慮せずに先頭にprefix
        hidden_states = torch.cat([peft_kwargs.get("outer_single_prefix"), hidden_states], dim=1)
        vertual_token_num = peft_kwargs.get("outer_single_prefix").shape[1]

    # 1. Gated MLP's linear projection
    
    if self.lora_in_proj is None and self.lora_X is None and self.lora_Z is None:
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
    else:
        projected_states = self.in_proj(hidden_states)
        if self.lora_in_proj is not None:
             projected_states = projected_states + self.s_in_proj*self.lora_in_proj(hidden_states)
        if self.lora_X is not None:
             projected_states[..., :projected_states.shape[-1]//2] = projected_states[..., :projected_states.shape[-1]//2] + self.s_X*self.lora_X(hidden_states)
        if self.lora_Z is not None:
             projected_states[..., projected_states.shape[-1]//2:] = projected_states[..., projected_states.shape[-1]//2:] + self.s_Z*self.lora_Z(hidden_states)
        projected_states = projected_states.transpose(1, 2)

    if self.learnable_A_log is not None:
        A = -torch.exp((self.A_log+self.learnable_A_log).float())  # (d_inner, d_state)
    else:
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

    if self.scan_addition_num > 0:
        A_addi = -torch.exp(self.A_log_scan_addi.float())
        
        if self.scan_addition_pos=="suffix":
            A = torch.cat([A, A_addi], dim=1)
        else:
            A = torch.cat([A_addi, A], dim=1)
    
    x_proj_weight = self.x_proj.weight
    if self.lora_x_proj is not None:
        x_proj_weight = x_proj_weight + self.s_x_proj * self.lora_x_proj.adapter_up.weight @ self.lora_x_proj.adapter_down.weight
    if self.lora_d is not None:
        x_proj_weight = x_proj_weight + torch.cat([self.s_d * self.lora_d.adapter_up.weight @ self.lora_d.adapter_down.weight, torch.zeros((self.ssm_state_size*2, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
    if self.lora_B is not None:
        x_proj_weight = x_proj_weight + torch.cat([torch.zeros((self.time_step_rank, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_B * self.lora_B.adapter_up.weight @ self.lora_B.adapter_down.weight, torch.zeros((self.ssm_state_size, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device)], dim=0)
    if self.lora_C is not None:
        x_proj_weight = x_proj_weight + torch.cat([torch.zeros((self.time_step_rank+self.ssm_state_size, x_proj_weight.shape[1]), dtype=x_proj_weight.dtype, device=x_proj_weight.device), self.s_C * self.lora_C.adapter_up.weight @ self.lora_C.adapter_down.weight], dim=0)


    if self.scan_addition_num > 0:
        if self.scan_addition_pos=="suffix":
            x_proj_weight = torch.cat([x_proj_weight[:self.time_step_rank + self.ssm_state_size], self.x_proj_scan_addi.weight[:self.scan_addition_num], x_proj_weight[-self.ssm_state_size:], self.x_proj_scan_addi.weight[self.scan_addition_num:]], dim=0).contiguous()
        else:
            x_proj_weight = torch.cat([x_proj_weight[:self.time_step_rank], self.x_proj_scan_addi.weight[:self.scan_addition_num], x_proj_weight[self.time_step_rank: self.time_step_rank + self.ssm_state_size], self.x_proj_scan_addi.weight[self.scan_addition_num:], x_proj_weight[-self.ssm_state_size:]], dim=0).contiguous()
    else:
        x_proj_weight = x_proj_weight


    dt_proj_weight = self.dt_proj.weight
    dt_proj_bias = self.dt_proj.bias.float()
    if self.lora_dt is not None:
        dt_proj_weight = dt_proj_weight + self.s_dt * self.lora_dt.adapter_up.weight @ self.lora_dt.adapter_down.weight

    if self.learnable_bias is not None:
        dt_proj_bias = dt_proj_bias + self.learnable_bias

    if self.lora_conv1d_down is not None:
        wa = self.lora_conv1d_up.weight
        wb = self.lora_conv1d_down.weight
        conv1d_weight_delta = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
        conv1d_weight_delta = conv1d_weight_delta.view(self.conv1d.weight.shape)
        conv1d_weight = self.conv1d.weight + conv1d_weight_delta*self.s_conv1d
    elif self.learnable_conv1d_weight is not None:
        conv1d_weight = self.conv1d.weight + self.learnable_conv1d_weight
    else:
        conv1d_weight = self.conv1d.weight
    
    if peft_kwargs.get("inner_single_prefix") is not None: # 先頭にprefix (f,b共に同じprefix)
        single_prefix = peft_kwargs.get("inner_single_prefix").permute(0,2,1)
        projected_states = torch.cat([single_prefix, projected_states], dim=2)
        vertual_token_num = single_prefix.shape[2]
    else:
        pass


    if self.training or cache_params is None:  # Doesn't support outputting the states -> used for training
        contextualized_states = mamba_inner_fn(
            projected_states,
            conv1d_weight,
            self.conv1d.bias if self.use_conv_bias else None,
            x_proj_weight,
            dt_proj_weight,
            self.out_proj.weight if self.lora_out_proj is None else self.out_proj.weight + self.s * self.lora_out_proj.adapter_up.weight @ self.lora_out_proj.adapter_down.weight,
            self.out_proj.bias.float() if self.use_bias else None,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float() if self.learnable_D is None else (self.D+self.learnable_D).float(),
            delta_bias=dt_proj_bias,
            delta_softplus=True,
        )

        if "inner_single_prefix" in peft_kwargs.keys():
            contextualized_states = contextualized_states[:, vertual_token_num:]
        elif "outer_single_prefix" in peft_kwargs.keys():
            contextualized_states = contextualized_states[:, vertual_token_num:]
        
    else:
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Convolution sequence transformation
        conv_weights = conv1d_weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if cache_params is not None and cache_position[0] > 0:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx],
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.update_conv_state(self.layer_idx, conv_states, cache_position)
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = F.linear(hidden_states.transpose(1, 2), x_proj_weight)
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = dt_proj_weight @ time_step.transpose(1, 2)

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = dt_proj_bias
        if cache_params is not None and cache_position[0] > 0:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D.float() if self.learnable_D is None else (self.D+self.learnable_D).float(),
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float() if self.learnable_D is None else (self.D+self.learnable_D).float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(self.layer_idx, ssm_state)

        # 4. Final linear projection
        if self.lora_out_proj is not None:
            contextualized_states = F.linear(scan_outputs.transpose(1, 2),  self.out_proj.weight + self.s * self.lora_out_proj.adapter_up.weight @ self.lora_out_proj.adapter_down.weight, self.out_proj.bias)
        else:
            contextualized_states = F.linear(scan_outputs.transpose(1, 2),  self.out_proj.weight, self.out_proj.bias)

        if "inner_single_prefix" in peft_kwargs.keys():
            contextualized_states = contextualized_states[..., vertual_token_num:]
        elif "outer_single_prefix" in peft_kwargs.keys():
            contextualized_states = contextualized_states[..., vertual_token_num:]
    return contextualized_states


def mambamixer_forward(
    self,
    hidden_states,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **peft_kwargs
):
    if is_fast_path_available and "cuda" in self.x_proj.weight.device.type and not torch._dynamo.is_compiling():
        return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask, **peft_kwargs)
    else:
        raise NotImplementedError
    

def forward_block_peft(
    self,
    hidden_states,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **peft_kwargs
):
    residual = hidden_states
    hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
    if self.residual_in_fp32:
        residual = residual.to(torch.float32)

    hidden_states = self.mixer(
        hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask, **peft_kwargs
    )
    hidden_states = residual + hidden_states
    return hidden_states

def forward_mambamodel(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    cache_params = None,
    use_cache: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
):
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )
    
    if inputs_embeds is None:
        inputs_embeds = self.embeddings(input_ids)
    if self.lora_patch_embed is not None:
        inputs_embeds = inputs_embeds + self.s_patch_embed * self.lora_patch_embed(input_ids, self.embeddings.padding_idx, self.embeddings.max_norm, self.embeddings.norm_type, self.embeddings.scale_grad_by_freq, self.embeddings.sparse)

    if self.prompt_encoder is not None: # 先頭にprefix (f,b共に同じprefix)
        prompts = self.prompt_encoder(inputs_embeds.shape[0])
        if self.prompt_type == "prefix":
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        else:
            raise NotImplementedError

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if use_cache:
        if cache_params is None:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
        elif cache_position is None:
            # cases when we do manual forward instead of using `model.generate` which will initiate
            # `cache_position` and makes sure it is not None, throw error here instead of doing some
            # hack to conjecture the current cache position
            raise ValueError(
                "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                "be initialized for you automatically"
            )
    else:
        cache_params = None

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None

    if self.prefix_encoder is not None:
        prefixes = self.prefix_encoder(inputs_embeds.shape[0])

    for i_l, mixer_block in enumerate(self.layers):
        peft_kwargs={}
        if self.prefix_encoder is not None:
            peft_kwargs[self.prefix_type] = prefixes[...,i_l*self.prefix_ch_per_layer:(i_l+1)*self.prefix_ch_per_layer]
        
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask, **peft_kwargs
            )
        else:
            hidden_states = mixer_block(
                hidden_states,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=attention_mask,
                **peft_kwargs
            )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

    hidden_states = self.norm_f(hidden_states)

    if self.prompt_encoder is not None: # 先頭にprefix (f,b共に同じprefix)
        if self.prompt_type == "prefix":
            hidden_states = hidden_states[:, prompts.shape[1]:]
        else:
            raise NotImplementedError

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

    return MambaOutput(
        last_hidden_state=hidden_states,
        cache_params=cache_params if use_cache else None,
        hidden_states=all_hidden_states,
    )

# TODO: for use_cash
def prepare_inputs_for_generation(
    self,
    input_ids,
    inputs_embeds=None,
    use_cache=None,
    cache_params: Optional[MambaCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if use_cache:
        # `cache_position` should have been initialized in `generate`
        if cache_position is None:
            raise ValueError(
                "`cache_position` should not be None as it should have been initialized in "
                "`model.generate`, you are responsible for passing in a valid `cache_position` if "
                "you are calling `prepare_inputs_for_generation` directly with `use_cache=True`"
            )
        if cache_position[0] > 0:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            if attention_mask is not None:
                attention_mask = None

        else:
            # we initialize the `cache_position` to full size of `conv_states` at prefill stage
            # considering padding will be applied when input length is shorter, and truncation
            # will be applied when it is longer, so it will be equivalent to always have it match
            # the length of `cache_params.conv_states`, which is `config.conv_kernel`
            cache_position = torch.arange(0, self.config.conv_kernel, device=input_ids.device)

    if inputs_embeds is not None and cache_params is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs.update(
        {
            "cache_params": cache_params,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs



class Adapter(nn.Module):
    def __init__(self, in_channels, out_channels, dim, bit=32, use_act=False):
        super().__init__()

        if bit == 32:
            self.adapter_down = nn.Linear(in_channels, dim, bias=False)
            self.adapter_up = nn.Linear(dim, out_channels, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            raise NotImplementedError
            # self.adapter_down = QLinear(768, dim, bit)
            # self.adapter_up = QLinear(dim, 768, bit)
            # nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        if use_act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up

class EmbedAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, dim, bit=32):
        super().__init__()

        if bit == 32:
            weight_A = torch.randn((in_channels, dim))
            weight_B = torch.randn((dim, out_channels))
            self.adapter_down = nn.Parameter(weight_A)
            self.adapter_up = nn.Parameter(weight_B)
            nn.init.zeros_(self.adapter_up.data)
        else:
            raise NotImplementedError
            # self.adapter_down = QLinear(768, dim, bit)
            # self.adapter_up = QLinear(dim, 768, bit)
            # nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        self.dim = dim

    def forward(self, x, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
        x_down = F.embedding(
            x,
            self.adapter_down,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        x_up = x_down@self.adapter_up
        return x_up

# Based on https://github.com/huggingface/peft/blob/main/src/peft/tuners/prefix_tuning/model.py
class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, num_per_layer*layers*hidden)
    '''
    def __init__(self, prefix_projection=True, token_dim=768, num_virtual_tokens=20, encoder_hidden_size=768, num_layers=12, num_per_layer=2):
        super().__init__()
        self.prefix_projection = prefix_projection
        self.prompt_tokens = torch.arange(
            num_virtual_tokens
        ).long()
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = torch.nn.Sequential(
                torch.nn.Linear(token_dim, encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(encoder_hidden_size, num_layers * num_per_layer * token_dim),
            )
        else:
            self.embedding = torch.nn.Embedding(num_virtual_tokens, num_layers * num_per_layer * token_dim)

    def forward(self, batch_size):
        prefix = (
            self.prompt_tokens
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(self.embedding.weight.device)
        )
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    

def set_peft(model, 
             # Adaptformer
             adaptformer=False, dim_adaptf=32, s_adaptf=1, bit_adaptf=32, 
             # LoRA config out_proj (AdaptFormer)
             lora_out_proj=False, dim=32, s=1, bit=32, 
             # LoRA config in_proj (X and Z),
             lora_in_proj=False, dim_in_proj=32, s_in_proj=1, bit_in_proj=32,
             # LoRA config X,
             lora_X=False, dim_X=32, s_X=1, bit_X=32,
             # LoRA confg Z
             lora_Z=False, dim_Z=32, s_Z=1, bit_Z=32, 
             # LoRA config x_proj (d, B, and C),
             lora_x_proj=False, dim_x_proj=4, s_x_proj=1, bit_x_proj=32,
             # LoRA config d,
             lora_d=False, dim_d=4, s_d=1, bit_d=32,
             # LoRA config B,
             lora_B=False, dim_B=4, s_B=1, bit_B=32,
             # LoRA config C,
             lora_C=False, dim_C=4, s_C=1, bit_C=32,
             # LoRA config dt,
             lora_dt=False, dim_dt=4, s_dt=1, bit_dt=32,
             # LoRA config conv1d,
             lora_conv1d=False, dim_conv1d=32, s_conv1d=1, bit_conv1d=32,
             # LoRA config patch_embed Conv2d,
             lora_patch_embed=False, dim_patch_embed=32, s_patch_embed=1, bit_patch_embed=32,
             # prefix tuning config
             prefix_tuning=False, prefix_type="inner_single_prefix", prefix_projection=True, num_virtual_tokens=1, encoder_hidden_size=None,
             # prompt tuning config
             prompt_tuning=False, prompt_type="prefix", prompt_projection=True, prompt_num_tokens=2,
             # MambaならではのPEFTを考えてみた？Scanする数（state_d）を増やすという方向性
             additional_scan=False, scan_addition_num=1, scan_addition_pos="suffix", scan_A_constant=None, scan_A_copy_from_last=False, zero_init_x_proj=False,
             # Aのfinetuning
             learnable_A=False, learnable_A_v2=False,
             # Dのfinetuning
             learnable_D=False, learnable_D_v2=False,
             # conv1dのfinetuning
             learnable_conv1d=False, learnable_conv1d_v2=False,
             # cls_tokenのfinetuning
             learnable_cls_token=False, learnable_cls_token_v2=False,
             # pos_embedのfinetuning
             learnable_pos_embed=False, learnable_pos_embed_v2=False,
             # biasのfinetuning
             learnable_bias=False, learnable_bias_v2=False,
             ):
    
    if type(model) == MambaModel:
        if prefix_tuning:
            num_layers = len(model.layers)
            token_dim = model.embeddings.embedding_dim
            if prefix_type in ["inner_single_prefix", "outer_single_prefix"]:
                num_per_layer = 1
            else:
                raise NotImplementedError
            
            if prefix_type in ["inner_single_prefix"]:
                token_dim = model.layers[0].mixer.intermediate_size * 2
            encoder_hidden_size_ = encoder_hidden_size if encoder_hidden_size is not None else token_dim
            model.prefix_ch_per_layer = num_per_layer*token_dim
            model.prefix_encoder = PrefixEncoder(prefix_projection, token_dim, num_virtual_tokens, encoder_hidden_size_, num_layers, num_per_layer).to(model.embeddings.weight.device)
            model.prefix_type = prefix_type        
        else:
            model.prefix_encoder = None

        if prompt_tuning:
            token_dim = model.embeddings.embedding_dim
            model.prompt_encoder = PrefixEncoder(prompt_projection, token_dim, prompt_num_tokens, token_dim, 1, 1).to(model.embeddings.weight.device)
            model.prompt_type = prompt_type
        else:
            model.prompt_encoder = None
        
        if learnable_cls_token and learnable_cls_token_v2:
            model.learnable_cls_token= nn.Parameter(torch.zeros_like(model.cls_token.data))
        else:
            model.learnable_cls_token = None
        
        if learnable_pos_embed and learnable_pos_embed_v2:
            model.learnable_pos_embed= nn.Parameter(torch.zeros_like(model.pos_embed.data))
        else:
            model.learnable_pos_embed = None

        if lora_patch_embed:
            model.s_patch_embed=s_patch_embed
            model.lora_patch_embed = EmbedAdapter(model.embeddings.num_embeddings, model.embeddings.embedding_dim, dim_patch_embed, bit_patch_embed).to(model.embeddings.weight.device)
        else:
            model.lora_patch_embed = None 
            
        bound_method = forward_mambamodel.__get__(model, model.__class__)
        setattr(model, 'forward', bound_method)

    for layer in model.children():
        if type(layer) == MambaMixer:
            # print(f"add lora to {layer}")

            if adaptformer:
                in_ch, out_ch = layer.intermediate_size, layer.hidden_size
                layer.adaptf = Adapter(in_ch, out_ch, dim_adaptf, bit_adaptf, use_act=True).to(layer.out_proj.weight.device)
                layer.s_adaptf = s_adaptf
            else:
                layer.adaptf = None

            if lora_out_proj:
                in_ch, out_ch = layer.intermediate_size, layer.hidden_size
                layer.lora_out_proj = Adapter(in_ch, out_ch, dim, bit).to(layer.out_proj.weight.device)
                layer.s = s
            else:
                layer.lora_out_proj = None
            
            if lora_in_proj:
                in_ch, out_ch = layer.hidden_size, layer.intermediate_size*2
                layer.lora_in_proj = Adapter(in_ch, out_ch, dim_in_proj, bit_in_proj).to(layer.in_proj.weight.device)
                layer.s_in_proj = s_in_proj
            else:
                layer.lora_in_proj = None
            
            if lora_X:
                in_ch, out_ch = layer.hidden_size, layer.intermediate_size
                layer.lora_X = Adapter(in_ch, out_ch, dim_X, bit_X).to(layer.in_proj.weight.device)
                layer.s_X = s_X
            else:
                layer.lora_X = None
            
            if lora_Z:
                in_ch, out_ch = layer.hidden_size, layer.intermediate_size
                layer.lora_Z = Adapter(in_ch, out_ch, dim_Z, bit_Z).to(layer.in_proj.weight.device)
                layer.s_Z = s_Z
            else:
                layer.lora_Z = None
            
            if lora_x_proj:
                in_ch, out_ch = layer.intermediate_size, layer.time_step_rank+layer.ssm_state_size*2
                layer.lora_x_proj = Adapter(in_ch, out_ch, dim_x_proj, bit_x_proj).to(layer.x_proj.weight.device)
                layer.s_x_proj = s_x_proj
            else:
                layer.lora_x_proj = None
            
            if lora_d:
                in_ch, out_ch = layer.intermediate_size, layer.time_step_rank
                layer.lora_d = Adapter(in_ch, out_ch, dim_d, bit_d).to(layer.x_proj.weight.device)
                layer.s_d = s_d
            else:
                layer.lora_d = None
            
            if lora_B:
                in_ch, out_ch = layer.intermediate_size, layer.ssm_state_size
                layer.lora_B = Adapter(in_ch, out_ch, dim_B, bit_B).to(layer.x_proj.weight.device)
                layer.s_B = s_B
            else:
                layer.lora_B = None
            
            if lora_C:
                in_ch, out_ch = layer.intermediate_size, layer.ssm_state_size
                layer.lora_C = Adapter(in_ch, out_ch, dim_C, bit_C).to(layer.x_proj.weight.device)
                
                layer.s_C = s_C
            else:
                layer.lora_C = None
            
            if lora_dt:
                in_ch, out_ch = layer.time_step_rank, layer.intermediate_size
                layer.lora_dt = Adapter(in_ch, out_ch, dim_dt, bit_dt).to(layer.x_proj.weight.device)
                layer.s_dt = s_dt
            else:
                layer.lora_dt = None
            
            if lora_conv1d:
                print(layer.conv1d.groups)
                layer.lora_conv1d_down = nn.Conv1d(
                    in_channels=layer.conv1d.in_channels,
                    out_channels=dim_conv1d,
                    bias=False,
                    kernel_size=layer.conv1d.kernel_size,
                    groups=layer.conv1d.groups,
                    padding=layer.conv1d.padding,
                ).to(layer.conv1d.weight.device)
                layer.lora_conv1d_up = nn.Conv1d(
                    in_channels=dim_conv1d,
                    out_channels=layer.conv1d.out_channels,
                    bias=False,
                    kernel_size=1
                ).to(layer.conv1d.weight.device)
                layer.s_conv1d = s_conv1d
            elif learnable_conv1d and learnable_conv1d_v2:
                layer.lora_conv1d_down = None
                device = layer.conv1d.weight.device
                layer.learnable_conv1d_weight = nn.Parameter(torch.zeros_like(layer.conv1d.weight, device=device))
            else:
                layer.lora_conv1d_down = None
                layer.learnable_conv1d_weight = None
                

            if additional_scan:
                d_state = layer.ssm_state_size
                d_inner = layer.intermediate_size
                device = layer.A_log.device
                dtype= layer.A_log.dtype
                
                if scan_A_copy_from_last:
                    if scan_addition_pos == "suffix":
                        A_scan_addi_log= repeat(layer.A_log.data[:, -1], "d -> d n",
                            n=scan_addition_num,
                        ).contiguous()
                    else:
                        A_scan_addi_log= repeat(layer.A_log.data[:, 0], "d -> d n",
                            n=scan_addition_num,
                        ).contiguous()
                elif scan_A_constant is None:
                    A_scan_addi = repeat(
                        torch.arange(d_state+1, d_state+1+scan_addition_num, dtype=dtype, device=device),
                        "n -> d n",
                        d=d_inner,
                    ).contiguous()
                    A_scan_addi_log = torch.log(A_scan_addi)  # Keep A_log in fp32   
                else:
                    A_scan_addi = repeat(
                        scan_A_constant*torch.ones((scan_addition_num,), dtype=dtype, device=device),
                        "n -> d n",
                        d=d_inner,
                    ).contiguous()
                    A_scan_addi_log = torch.log(A_scan_addi)  # Keep A_log in fp32      
                layer.A_log_scan_addi = nn.Parameter(A_scan_addi_log)
                layer.A_log_scan_addi._no_weight_decay = True
                layer.x_proj_scan_addi = nn.Linear(
                    d_inner, scan_addition_num * 2, bias=False, dtype=dtype, device=device
                )

                layer.scan_addition_num=scan_addition_num
                layer.scan_addition_pos=scan_addition_pos

                if zero_init_x_proj:
                    nn.init.zeros_(layer.x_proj_scan_addi.weight)
            else:
                layer.scan_addition_num=-1
            
            if learnable_A and learnable_A_v2:
                device = layer.A_log.data.device
                layer.learnable_A_log = nn.Parameter(torch.zeros_like(layer.A_log.data, device=device))
            else:
                layer.learnable_A_log = None

            if learnable_D and learnable_D_v2:
                device = layer.D.data.device
                layer.learnable_D = nn.Parameter(torch.zeros_like(layer.D.data, device=device))
            else:
                layer.learnable_D = None
            
            if learnable_bias and learnable_bias_v2:
                device = layer.dt_proj.bias.device
                layer.learnable_bias = nn.Parameter(torch.zeros_like(layer.dt_proj.bias, device=device))
                layer.learnable_bias_b = nn.Parameter(torch.zeros_like(layer.dt_proj_b.bias, device=device))
            else:
                layer.learnable_bias = None
                layer.learnable_bias_b = None

            bound_method = mambamixer_cuda_kernels_forward.__get__(layer, layer.__class__)
            setattr(layer, 'cuda_kernels_forward', bound_method)
            bound_method = mambamixer_forward.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif type(layer) == MambaBlock:
            bound_method = forward_block_peft.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)

        
        if len(list(layer.children())) != 0:
            set_peft(layer,
                     # Adaptformer
                     adaptformer, dim_adaptf, s_adaptf, bit_adaptf,
                     lora_out_proj, dim, s, bit,
                     lora_in_proj, dim_in_proj, s_in_proj, bit_in_proj,
                     # LoRA config X,
                     lora_X, dim_X, s_X, bit_X,
                     # LoRA confg Z
                     lora_Z, dim_Z, s_Z, bit_Z, 
                     # LoRA config x_proj (d, B, and C),
                     lora_x_proj, dim_x_proj, s_x_proj, bit_x_proj,
                     # LoRA config d,
                     lora_d, dim_d, s_d, bit_d,
                     # LoRA config B,
                     lora_B, dim_B, s_B, bit_B,
                     # LoRA config C,
                     lora_C, dim_C, s_C, bit_C,
                     # LoRA config dt,
                     lora_dt, dim_dt, s_dt, bit_dt,
                     # LoRA config conv1d,
                     lora_conv1d, dim_conv1d, s_conv1d, bit_conv1d,
                     # LoRA config patch_embed Conv2d,
                     lora_patch_embed, dim_patch_embed, s_patch_embed, bit_patch_embed,
                     # Prefix tuning
                     prefix_tuning, prefix_type, prefix_projection, num_virtual_tokens, encoder_hidden_size,
                     # prompt tuning config
                     prompt_tuning, prompt_type, prompt_projection, prompt_num_tokens,
                     # Additional Scan
                     additional_scan, scan_addition_num, scan_addition_pos, scan_A_constant, scan_A_copy_from_last, zero_init_x_proj,
                     # Aのfinetuning
                     learnable_A, learnable_A_v2,
                     # Dのfinetuning
                     learnable_D, learnable_D_v2,
                     # conv1dのfinetuning
                     learnable_conv1d, learnable_conv1d_v2,
                     # cls_tokenのfinetuning
                     learnable_cls_token, learnable_cls_token_v2,
                     # pos_embedのfinetuning
                     learnable_pos_embed, learnable_pos_embed_v2,
                     # biasのfinetuning
                     learnable_bias, learnable_bias_v2,)


