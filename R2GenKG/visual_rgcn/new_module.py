import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context, attn = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + context
        q = q + self.mlp(self.ln_2(q))
        return q


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        #print(self.gamma, self.beta, x)
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)



class SimpleMLPWithGELU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLPWithGELU, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size)
            # nn.Linear(input_size, hidden_size),
            # nn.GELU(),
            # nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # 重塑 x 从 (B, 2048, 768) 到 (B * 768, 2048)
        original_shape = x.shape
        x = x.view(-1, original_shape[1])
        # 应用 MLP
        x = self.layers(x)
        # 恢复 x 的形状到 (B, 128, 768)
        x = x.view(original_shape[0], -1, original_shape[2])
        return x

# # 输入尺寸为768，中间层尺寸自选，输出尺寸为128
# mlp = SimpleMLPWithGELU(input_size=768, hidden_size=512, output_size=128)
#
# # 假设输入
# input_tensor = torch.randn(10, 2048, 768) # Shape: (B, 2048, 768)
#
# # 得到输出
# output_tensor = mlp(input_tensor) # 输出形状将是 (B, 128, 768)
# print(output_tensor.shape)  # 应为 (10, 128, 768)
