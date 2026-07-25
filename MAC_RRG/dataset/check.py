import torch

edge_type = torch.load('/wangx/home/E24301191/mycode/R2GenGPT-main/R2GenGPT-main/R_GCN/edge_type.pt')  # 加载边类型 (num_edges,)

print(edge_type.shape)  # 打印 edge_type 的形状
