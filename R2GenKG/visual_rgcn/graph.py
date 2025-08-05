import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

class RGCN(nn.Module):
    def __init__(self, num_features, hidden_dim=256, output_dim=768, dropout=0.1, num_relations=3):
        super(RGCN, self).__init__()

        # 第一层 RGCN，输入维度是节点特征的维度，输出维度是 hidden_dim
        self.conv1 = RGCNConv(num_features, hidden_dim, num_relations)

        # 第二层 RGCN，输入维度是 hidden_dim，输出维度是 768
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations)

        # Dropout 层用于防止过拟合
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        # 第一层图卷积
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        
        # 第二层图卷积，输出维度为 768
        x = self.conv2(x, edge_index, edge_type)

        # 返回聚合后的节点特征（输出为维度为 768 的节点嵌入）
        return x


