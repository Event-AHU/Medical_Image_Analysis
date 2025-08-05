import torch
import torch.nn as nn
import torch.nn.functional as F

# 可训练 Cross-Attention 模块
class CrossAttentionLookup(nn.Module):
    def __init__(self, query_dim=768, key_dim=768, hidden_dim=768):
        super(CrossAttentionLookup, self).__init__()
        # 可选：对 Query / Key / Value 做线性映射（提升学习能力）
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)

        self.scale = hidden_dim ** 0.5

    def forward(self, queries, key_bank):
        """
        queries: (B, N_query, D)  e.g. (B, 14, 768)
        key_bank: (N_key, D)      e.g. (6943, 768)
        """
        # 1. 映射维度
        Q = self.query_proj(queries)                 # (B, 14, D)
        K = self.key_proj(key_bank).T                # (D, 6943)
        V = self.value_proj(key_bank)                # (6943, D)

        # 2. 注意力打分: (B, 14, 6943)
        attn_scores = torch.matmul(Q, K) / self.scale

        # 3. Softmax 权重: (B, 14, 6943)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 4. 加权求和: (B, 14, 768)
        output = torch.matmul(attn_weights, V)

        return output  # shape: (B, 14, 768)


# B = 8  # batch size
# image_queries = torch.randn(B, 14, 768)
# disease_bank = torch.randn(6943, 768)
# target_output = torch.randn(B, 14, 768)  # 模拟训练目标

# # 初始化模型
# model = CrossAttentionLookup()

