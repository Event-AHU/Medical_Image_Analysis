##

**MAC-RRG: Iterative Multi-Agent Collaboration for X-ray Radiology Report Generation**,
Futian Wang, Yuhan Qiao, Xiao Wang*, Dan Xu, Yuehang Li, Zhixiang Guo*, Jin Tang


#### **Abstract**
Despite the remarkable progress of LLM-based and knowledge graph-augmented Radiology Report Generation (RRG) methods, existing techniques still suffer from inherent defects. Conventional LLM-only models lack structured medical prior knowledge, resulting in frequent medical hallucinations and low diagnostic interpretability. Current knowledge graph-enhanced schemes adopt static one-round knowledge fusion with single-source knowledge, incapable of dynamic knowledge updating according to generation feedback. This paper proposes a novel multi-agent collaborative iterative framework for X-ray radiology report generation, termed MAC-RRG. Inspired by multi-agent technology, our framework constructs a closed-loop optimization paradigm based on task decoupling and collaborative reasoning. Specifically, the framework first generates a preliminary radiology report from input X-ray images via a vision encoder and a basic LLM. Subsequently, a multimodal knowledge graph (MM-KG) agent mines structured disease correlation and anatomical knowledge from medical knowledge graphs, while an auxiliary knowledge agent extracts unstructured domain knowledge from public medical databases. The multi-source knowledge acquired by dual agents is fused and embedded to guide the LLM in iteratively refining the initial report. Extensive quantitative and qualitative experiments on mainstream X-ray RRG datasets verify the superiority of our proposed method.

---

#### Framework

MAC-RRG follows a **draft → dual-agent knowledge mining → iterative refinement** pipeline:

```
Chest X-ray
    │
    ▼
Vision Encoder (Swin) + LLM  ──►  Draft Report (Draft_text)
    │
    ├──────────────────────────────┐
    ▼                              ▼
MM-KG Agent                    RAG Agent
(entity extraction,            (entity-wise BGE retrieval,
 KG subgraph, GAT encoding)     reranking, chunk encoding)
    │                              │
    └──────────┬───────────────────┘
               ▼
    Multi-source Knowledge Fusion
               │
               ▼
    LLM Refinement (Llama-2)  ──►  Final Radiology Report
```

**Key components:**

| Module | Role |
|--------|------|
| **Draft Generator** | Encodes the chest X-ray with Swin Transformer and produces a preliminary report via Llama-2 |
| **MM-KG Agent** (`A_MM_KG_Agent/`) | Extracts anatomy/disorder entities from the draft, retrieves related triples from a medical KG, and encodes the subgraph with a Graph Attention Embedder |
| **RAG Agent** (`A_RAG_Agent/`) | Performs entity-wise dense retrieval (BGE-M3) and reranking over public medical text chunks, then encodes retrieved knowledge with Bio_ClinicalBERT |
| **Knowledge Fusion & Refinement** | Concatenates visual, KG, and RAG embeddings as soft prompts to guide Llama-2 in refining the draft into the final report |

**Supported datasets:** IU X-ray (`iu_xray`), MIMIC-CXR (`mimic_cxr`), CheXpert Plus (`cheXpert_plus`).

---

#### Environment Configuration

**Requirements**

- Python 3.10+ (recommended)
- CUDA-capable GPU
- PyTorch (install according to your local CUDA version)

```bash
pip install -r requirements.txt
```

**Main dependencies** (`requirements.txt`):

```
torch
peft
tensorboardX
transformers==4.30.2
lightning==2.0.5
Pillow
numpy
gradio
```

**Pretrained assets** (update paths in `configs/config.py` and `model/R2GenGPT.py`):

| Component | Example Path / Model |
|-----------|----------------------|
| Vision backbone | `microsoft/swin-base-patch4-window7-224` |
| Language model | `meta-llama/Llama-2-7b-chat-hf` |
| Clinical text encoder | `emilyalsentzer/Bio_ClinicalBERT` |
| Retriever / Reranker | `A_RAG_Agent/bge-m3`, `A_RAG_Agent/bge-reranker-v2-m3` |
| KG resources | `A_MM_KG_Agent/res_dict_aliases.json`, `A_MM_KG_Agent/kg_relations.csv` |
| RAG corpus | `A_RAG_Agent/chunks.json`, `A_RAG_Agent/doc_vecs_cache.pt` |

**Data preparation**

1. Place images and annotation JSON under `data/` (or update `--annotation` / `--base_dir`).
2. Each sample should contain `id`, `image_path`, `report`, and **`Draft_text`** (Stage-1 draft report).
3. Replace absolute server paths in the config/scripts with your local paths before running.

---

#### Training and Testing

All experiments are launched via `train.py` with PyTorch Lightning. Example scripts are provided under `model/`.

**Training (IU X-ray)**

```bash
bash model/3-1.deep_run_iu.sh
```

Or:

```bash
python -u train.py \
  --dataset iu_xray \
  --annotation ./data/iu_xray/annotation.json \
  --base_dir ./data/iu_xray/images \
  --batch_size 8 \
  --val_batch_size 8 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ./save/iu_xray/v1 \
  --max_length 60 \
  --min_new_tokens 40 \
  --max_new_tokens 100 \
  --devices 1 \
  --max_epochs 30
```

**Training (CheXpert Plus)**

```bash
bash model/6-1.deep_run.sh
```

**Testing**

```bash
# IU X-ray
bash model/3-2.deep_test_iuxray.sh

# CheXpert Plus
bash model/6-1.deep_test.sh
```

Or:

```bash
python -u train.py \
  --test \
  --dataset iu_xray \
  --annotation ./data/iu_xray/annotation.json \
  --base_dir ./data/iu_xray/images \
  --delta_file /path/to/checkpoint.pth \
  --savedmodel_path ./save/iu_xray/test \
  --devices 1
```

**Evaluation metrics:** BLEU-1~4, ROUGE-L, METEOR, and CIDEr (implemented in `evalcap/`).

Hyperparameters and path settings are defined in `configs/config.py`.

---

#### Experimental Results

We evaluate MAC-RRG on mainstream X-ray RRG benchmarks (IU X-ray, MIMIC-CXR, and CheXpert Plus). Our method consistently improves NLG metrics over strong LLM-based and knowledge-augmented baselines, demonstrating the benefit of iterative multi-agent collaboration with multi-source knowledge fusion.

> Detailed quantitative tables, ablation studies, and qualitative case studies will be provided in the paper / supplementary material. You may also insert result figures (e.g., `figures/framework.png`, `figures/results.png`) here after release.

---

#### Acknowledgment

This project is built upon [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) and related open-source radiology report generation / medical NLP resources. We thank the authors of IU X-ray, MIMIC-CXR, CheXpert, Llama-2, Swin Transformer, Bio_ClinicalBERT, and BGE for releasing their datasets and models.

---

#### Citation

If you find this work useful for your research, please give us a star ⭐ and cite the following paper:

```bibtex
@article{wang2025macrrg,
  title   = {MAC-RRG: Iterative Multi-Agent Collaboration for X-ray Radiology Report Generation},
  author  = {Wang, Futian and Qiao, Yuhan and Wang, Xiao and Xu, Dan and Li, Yuehang and Guo, Zhixiang and Tang, Jin},
  journal = {},
  year    = {2025}
}
```

```
MAC-RRG: Iterative Multi-Agent Collaboration for X-ray Radiology Report Generation
Futian Wang, Yuhan Qiao, Xiao Wang*, Dan Xu, Yuehang Li, Zhixiang Guo*, Jin Tang
```
