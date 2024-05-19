import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.vit import *

class TransformerClassifier(nn.Module):
    def __init__(self, attr_num, dim=1024, pretrain_path=None):#checkpoint-last.pth
        #/data/rongyao_data/moco-v3-main/checkpoint-90.pth
        #tar -xvf /data/rongyao_data/moco-v3-main/checkpoint_0080.pth.tar -C /data/wuwentao/VTB/models
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(768, dim)

        self.vit = vit_large()
        # pretrain_path = '/Share/home/22054/VTB3333/checkpoint-146.pth'
        # self.vit.load_param(pretrain_path)
        #self.vit.load_param(pretrain_path)

        self.blocks = self.vit.blocks[-1:]
        self.norm = self.vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)

        self.vis_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.tex_embed = nn.Parameter(torch.zeros(1, 1, dim))

 

    def forward(self, imgs, word_vec, label=None):
        features = self.vit(imgs)
        
        word_embed = self.word_embed(word_vec).expand(features.shape[0], word_vec.shape[0], features.shape[-1])
        
        tex_embed = word_embed + self.tex_embed
        vis_embed = features + self.vis_embed

        x = torch.cat([tex_embed, vis_embed], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        logits = self.bn(logits)

        return logits