import torch
import torch.nn as nn

class MultiScaleSelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_scales=5, max_nodes_per_scale=600, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.max_nodes_per_scale = max_nodes_per_scale

        # 可学习的尺度编码（5个尺度）
        self.scale_embedding = nn.Embedding(num_scales, embed_dim)
        
        # 可学习的位置编码（最多每个尺度600个节点）
        self.pos_embedding = nn.Embedding(max_nodes_per_scale, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, node_feats_list):
        """
        node_feats_list: List of tensors with shape [N_i, D] (each is one scale)
        Returns: 
            fused_output: [total_N, D]
            per_scale_outputs: list of tensors, each [N_i, D]
        """
        device = node_feats_list[0].device
        all_feats = []
        all_scales = []
        all_positions = []
        scale_ranges = []  # 记录每个尺度在融合特征中的起止索引

        start_idx = 0
        for scale_id, feats in enumerate(node_feats_list):
            num_nodes = feats.size(0)

            all_feats.append(feats)
            scale_ids = torch.full((num_nodes,), scale_id, device=device)
            all_scales.append(scale_ids)
            pos_ids = torch.arange(num_nodes, device=device)
            all_positions.append(pos_ids)

            scale_ranges.append((start_idx, start_idx + num_nodes))
            start_idx += num_nodes

        # 拼接所有节点
        x = torch.cat(all_feats, dim=0)           # [total_N, D]
        scale_ids = torch.cat(all_scales, dim=0)  # [total_N]
        pos_ids = torch.cat(all_positions, dim=0) # [total_N]

        # 加入尺度和位置编码
        x += self.scale_embedding(scale_ids)
        x += self.pos_embedding(pos_ids)

        # Transformer 编码
        x = x.unsqueeze(0)         # [1, total_N, D]
        x = self.encoder(x)        # [1, total_N, D]
        x = x.squeeze(0)           # [total_N, D]

        # 拆分为每个尺度的输出
        per_scale_outputs = [
            x[start:end] for (start, end) in scale_ranges
        ]

        return x, per_scale_outputs
