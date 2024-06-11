""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on code in:
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision/tree/main/big_vision

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


class SmallPatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=1024,hidden_dim=1024,bias=True):
        super(SmallPatchEmbed, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, hidden_dim, kernel_size=16, stride=16, bias=bias)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=4, bias=bias)

        self.proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.num_patches=400
        self.patch_size=(64,64)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x) # 在每个小卷积层之后添加ReLU激活函数
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        return x
