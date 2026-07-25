import os
import json
import re
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ======================
# Config
# ======================
MODEL_NAME = "/wangx_nas/QYH/emilyalsentzer/Bio_ClinicalBERT/"
ENTITY_LINKS_JSON = "/data/qiaoyuhan/hhhh/Zretrival/entity_links.json"
OUT_DIR = "/data/qiaoyuhan/hhhh/Zretrival/features"   # 每个中心实体一个 .pt
TOPK = 10                    # 每个中心实体最多保留的邻居条数（按 count 降序）；None 表示全部
BATCH_SIZE = 64              # 文本批量编码大小
POOLING = "cls"              # "cls" 或 "mean"
DTYPE = torch.float32        # 可改为 torch.float16（推理用）以省显存

# os.makedirs(OUT_DIR, exist_ok=True)

# Tokenizer and Model Setup
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME).eval()  # 不需要将模型移到特定设备，默认在 CPU 上运行

# ======================
# Helpers
# ======================


@torch.no_grad()
def encode_batch(texts: List[str], device: torch.device, model, tokenizer) -> torch.Tensor:    
    """对一批字符串做编码 -> (N, hidden)"""
    encoded_list = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)  # Ensure the inputs are on the correct device
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        if POOLING == "mean":
            # 按 attention_mask 做 mean
            mask = inputs["attention_mask"].unsqueeze(-1)  # (B,L,1)
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom
        else:  # "cls"
            pooled = last_hidden[:, 0, :]
        encoded_list.append(pooled.detach().to(device).to(DTYPE))  # Ensure it's on the correct device
    return torch.cat(encoded_list, dim=0)

class TextEncoderCache:
    """简单缓存，避免重复文案反复编码"""
    def __init__(self):
        self.store: Dict[str, torch.Tensor] = {}

    def get_many(self, texts: List[str], device: torch.device , model, tokenizer) -> torch.Tensor:
        """获取多个文本的嵌入"""
        uniq = [t for t in texts if t not in self.store]
        if uniq:
            embs = encode_batch(uniq, device, model, tokenizer)  # 传递 device 参数
            for t, e in zip(uniq, embs):
                self.store[t] = e
        # 确保返回的嵌入都在指定的设备上
        return torch.stack([self.store[t] for t in texts], dim=0).to(device)  # 确保返回的嵌入也在正确的设备上

cache = TextEncoderCache()

def sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\-\.]+", "_", name)
    return name or "unk"

def build_features_for_entity(entity: str, neighbors: List[Dict], device: torch.device, model, tokenizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    entity: 中心实体 (str)
    neighbors: [{"entity": "...", "relation": "...", "count": N}, ...]
    返回：
      central_emb: (1, H)
      neighbor_embs: (N, H)
      edge_embs: (N, H)
    """
    # 取 TopK（可选）

    if TOPK is not None and len(neighbors) > TOPK:
        neighbors = sorted(neighbors, key=lambda x: x.get("count", 0), reverse=True)[:TOPK]

    # 中心实体
    central_emb = cache.get_many([entity], device, model, tokenizer).squeeze(0).unsqueeze(0).to(device)  # 将中心实体的嵌入移到指定设备

    if len(neighbors) == 0:
        # 空邻居时返回 (0, H) 张量，方便后续处理
        H = central_emb.shape[-1]
        neighbor_embs = torch.empty(0, H, dtype=DTYPE, device=device)
        edge_embs = torch.empty(0, H, dtype=DTYPE, device=device)
        return central_emb, neighbor_embs, edge_embs

    # 邻居 & 边关系
    neighbor_texts = [n["entity"] for n in neighbors]
    relation_texts = [n["relation"] for n in neighbors]

    neighbor_embs = cache.get_many(neighbor_texts, device, model, tokenizer).to(device)  # (N, H) and move to the correct device
    edge_embs = cache.get_many(relation_texts, device, model, tokenizer).to(device)      # (N, H) and move to the correct device

    return central_emb, neighbor_embs, edge_embs
