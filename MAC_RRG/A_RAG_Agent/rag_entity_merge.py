import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


def resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_doc_vecs_cache(
    cache_path: str,
    device: str,
    normalize: bool = True,
) -> torch.Tensor:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"doc_vecs cache not found: {cache_path}")

    obj = torch.load(cache_path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("doc_vecs", "embeddings", "vecs"):
            if key in obj:
                obj = obj[key]
                break

    if isinstance(obj, np.ndarray):
        doc_vecs = torch.from_numpy(obj)
    elif torch.is_tensor(obj):
        doc_vecs = obj
    else:
        raise TypeError(
            f"Unsupported cache type: {type(obj)}. "
            f"Expect torch.Tensor / np.ndarray / dict(with doc_vecs)."
        )

    doc_vecs = doc_vecs.float()
    if normalize:
        doc_vecs = F.normalize(doc_vecs, p=2, dim=1)

    return doc_vecs.to(device)


@torch.no_grad()
def encode_bge_m3(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    task: str,                 # "query" or "passage"
    batch_size: int = 32,
    max_length: int = 512,
    normalize: bool = True,
    return_torch: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    model.eval()
    all_vecs = []

    if task == "query":
        texts = [f"Represent this question for retrieving supporting documents: {t}" for t in texts]
    elif task == "passage":
        texts = [f"Represent this passage for answering the question: {t}" for t in texts]
    else:
        raise ValueError("task must be 'query' or 'passage'")

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = model(**inputs)
        vec = out.last_hidden_state[:, 0]  # CLS

        if normalize:
            vec = F.normalize(vec, p=2, dim=1)

        if return_torch:
            all_vecs.append(vec)
        else:
            all_vecs.append(vec.detach().cpu().numpy())

    if return_torch:
        return torch.cat(all_vecs, dim=0)
    return np.vstack(all_vecs)


def retrieve_topn_by_inner_product_torch(
    query_vec: torch.Tensor,  # (D,) normalized
    doc_vecs: torch.Tensor,   # (N, D) normalized
    top_n: int,
) -> List[int]:
    if doc_vecs.numel() == 0:
        return []
    scores = doc_vecs @ query_vec  # (N,)
    k = min(int(top_n), scores.shape[0])
    if k <= 0:
        return []
    _, idx = torch.topk(scores, k=k, largest=True)
    return idx.detach().cpu().tolist()


@torch.no_grad()
def rerank_topk_bge(
    query: str,
    docs: List[str],
    tokenizer,
    model,
    device: str,
    top_k: int = 5,
    batch_size: int = 16,
    max_length: int = 512,
) -> List[Tuple[int, float]]:
    model.eval()
    scores: List[float] = []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i + batch_size]
        inputs = tokenizer(
            [query] * len(batch_docs),
            batch_docs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**inputs).logits
        logits = logits.view(-1).float()
        batch_scores = torch.sigmoid(logits).detach().cpu().tolist()
        scores.extend(batch_scores)

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    indexed = indexed[: min(top_k, len(indexed))]
    return [(i, float(s)) for i, s in indexed]


def build_entity_query(report_text: str, entity: str, entity_type: Optional[str] = None) -> str:
    if entity_type:
        return f"{report_text}\n\nFocus entity: {entity} (type: {entity_type})."
    return f"{report_text}\n\nFocus entity: {entity}."


def search_one_entity(
    entity: str,
    report_text: str,
    chunks: List[str],
    emb_tok,
    emb_model,
    rr_tok,
    rr_model,
    device: str,
    doc_vecs: torch.Tensor,
    entity_type: Optional[str] = None,
    top_n: int = 100,
    top_k: int = 5,
    rerank_batch_size: int = 16,
    max_length: int = 512,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []

    query_text = build_entity_query(report_text, entity, entity_type)

    q_vec = encode_bge_m3(
        [query_text],
        tokenizer=emb_tok,
        model=emb_model,
        device=device,
        task="query",
        batch_size=1,
        max_length=max_length,
        normalize=True,
        return_torch=True,
    )[0]  # (D,)

    cand_idx = retrieve_topn_by_inner_product_torch(q_vec, doc_vecs, top_n=top_n)
    cand_chunks = [chunks[i] for i in cand_idx]

    reranked = rerank_topk_bge(
        query=query_text,
        docs=cand_chunks,
        tokenizer=rr_tok,
        model=rr_model,
        device=device,
        top_k=top_k,
        batch_size=rerank_batch_size,
        max_length=max_length,
    )

    out: List[Dict[str, Any]] = []
    for rank, (local_i, score) in enumerate(reranked, start=1):
        orig_idx = cand_idx[local_i]
        out.append(
            {
                "entity": entity,
                "entity_type": entity_type,
                "rank": rank,
                "idx": int(orig_idx),
                "score": float(score),
                "chunk": chunks[orig_idx],
            }
        )
    return out


class EntityWiseBGESearcher:
    """
    ✅初始化时只加载一次：
      - bge-m3 tokenizer/model
      - reranker tokenizer/model
      - doc_vecs_cache.pt
    后续 run(...) 多次不会重复加载
    """
    def __init__(
        self,
        bge_m3_path: str,
        bge_reranker_path: str,
        doc_vecs_cache_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        normalize_cache: bool = True,
    ):
        self.device = resolve_device(device)
        self.max_length = max_length

        print(f"[INFO] device={self.device}")
        print("[INFO] loading embedding model/tokenizer...")
        self.emb_tok = AutoTokenizer.from_pretrained(bge_m3_path)
        self.emb_model = AutoModel.from_pretrained(bge_m3_path).to(self.device)

        print("[INFO] loading reranker model/tokenizer...")
        self.rr_tok = AutoTokenizer.from_pretrained(bge_reranker_path)
        self.rr_model = AutoModelForSequenceClassification.from_pretrained(bge_reranker_path).to(self.device)

        print(f"[INFO] loading doc_vecs cache: {doc_vecs_cache_path}")
        self.doc_vecs = load_doc_vecs_cache(doc_vecs_cache_path, device=self.device, normalize=normalize_cache)

    def load_chunks(self, chunks_json_path: str) -> List[str]:
        with open(chunks_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        chunks = obj.get("chunks", [])
        if not chunks:
            raise ValueError("No chunks found in JSON.")
        if self.doc_vecs.shape[0] != len(chunks):
            raise ValueError(
                f"Cache N mismatch: doc_vecs={self.doc_vecs.shape[0]} vs chunks={len(chunks)}. "
                f"Please rebuild doc_vecs_cache.pt for this chunks.json."
            )
        return chunks

    def run(
        self,
        report_text: str,
        merged_entities: Dict[str, str],
        chunks_json_path: str,
        top_n: int = 100,
        top_k: int = 5,
        rerank_batch_size: int = 16,
    ) -> Dict[str, List[Dict[str, Any]]]:
        chunks = self.load_chunks(chunks_json_path)

        results: Dict[str, List[Dict[str, Any]]] = {}
        for entity, etype in merged_entities.items():
            results[entity] = search_one_entity(
                entity=entity,
                report_text=report_text,
                chunks=chunks,
                emb_tok=self.emb_tok,
                emb_model=self.emb_model,
                rr_tok=self.rr_tok,
                rr_model=self.rr_model,
                device=self.device,
                doc_vecs=self.doc_vecs,
                entity_type=etype,
                top_n=top_n,
                top_k=top_k,
                rerank_batch_size=rerank_batch_size,
                max_length=self.max_length,
            )
        return results


def merge_dedup_chunks_only(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
    best_by_idx: Dict[int, Dict[str, Any]] = {}
    for _, hits in results.items():
        for h in hits:
            idx = int(h["idx"])
            score = float(h.get("score", 0.0))
            chunk = h.get("chunk", "")
            if (idx not in best_by_idx) or (score > float(best_by_idx[idx]["score"])):
                best_by_idx[idx] = {"score": score, "chunk": chunk}

    sorted_items = sorted(best_by_idx.items(), key=lambda kv: float(kv[1]["score"]), reverse=True)
    chunks = [v["chunk"] for _, v in sorted_items]
    return {"chunks": chunks}


if __name__ == "__main__":
    report_text = (
        "pa and lateral views of the chest provided . there is no focal consolidation effusion or pneumothorax . "
        "the cardiomediastinal silhouette is normal . imaged osseous structures are intact . "
        "no free air below the right hemidiaphragm is seen . surgical clips in the left upper quadrant suggest prior cholecystectomy."
    )

    merged_entities = {
        "chest": "anatomy",
        "pleural": "anatomy",
        "effusion": "disorder",
        "pneumothorax": "disorder",
        "consolidation": "disorder",
        "cardiomegaly": "disorder",
        "aorta": "anatomy",
        "pulmonary": "anatomy",
        "vascular": "anatomy",
        "congestion": "disorder",
        "shoulder": "anatomy",
        "thoracic": "anatomy",
        "spine": "anatomy",
    }

    chunks_json_path = "/wangx/home/E24301191/mycode/MAC_RRG/A_RAG_Agent/chunks.json"
    bge_m3_path = "/wangx/home/E24301191/mycode/MAC_RRG/A_RAG_Agent/bge-m3"
    bge_reranker_path = "/wangx/home/E24301191/mycode/MAC_RRG/A_RAG_Agent/bge-reranker-v2-m3"
    doc_vecs_cache_path = "/wangx/home/E24301191/mycode/MAC_RRG/A_RAG_Agent/doc_vecs_cache.pt"

    # ✅只加载一次（慢就慢这一次）
    searcher = EntityWiseBGESearcher(
        bge_m3_path=bge_m3_path,
        bge_reranker_path=bge_reranker_path,
        doc_vecs_cache_path=doc_vecs_cache_path,
        device="cuda:0",
        max_length=512,
        normalize_cache=True,
    )

    # ✅后面你可以对多个 report 反复调用，不会再加载模型
    results = searcher.run(
        report_text=report_text,
        merged_entities=merged_entities,
        chunks_json_path=chunks_json_path,
        top_n=80,
        top_k=1,
        rerank_batch_size=8,
    )

    merged_chunks = merge_dedup_chunks_only(results)
    print(merged_chunks["chunks"])
