# MAC-RRG

**Multi-Agent Collaboration for Radiology Report Generation**

A vision–language framework for Chest X-ray radiology report generation. Built on an R2GenGPT-style backbone, MAC-RRG introduces a 
**Knowledge Graph Agent (MM-KG)** and a **Retrieval-Augmented Generation Agent (RAG)**. Using a draft report (`Draft_text`) as the bridge, it fuses image features, knowledge-graph subgraphs, and external clinical text to produce more complete and clinically grounded reports.

## Method Overview

```
Chest X-ray ──► Swin Transformer ──┐
                                   │
Draft_text ──► Entity Extraction ──┬──► MM-KG Agent (GAT subgraph encoding) ──┤
                                   │                                          ├──► Concatenated embeds ──► Llama-2 ──► Report
                                   └──► RAG Agent (BGE retrieval + rerank) ────┘
```

1. **Visual encoding**: A Swin Transformer extracts image features and projects them into the LLM embedding space.
2. **MM-KG Agent**: Anatomy/disorder entities are extracted from the draft report; related triples are retrieved from a medical KG and encoded with a Graph Attention Embedder.
3. **RAG Agent**: Entity-wise retrieval is performed with BGE-M3, refined by a BGE reranker, and encoded with Bio_ClinicalBERT.
4. **Report generation**: Visual, RAG, and KG embeddings are concatenated and fed to Llama-2 via a prompt to generate the final report.

## Supported Datasets

| Dataset | `--dataset` | Description |
|---------|-------------|-------------|
| IU X-ray | `iu_xray` | Public small-scale chest X-ray report dataset |
| MIMIC-CXR | `mimic_cxr` | Large-scale ICU chest X-ray report dataset |
| CheXpert Plus | `cheXpert_plus` | Extended CheXpert variant |

Annotation JSON files should contain `id`, `image_path`, `report`, and **`Draft_text`** (a preliminary draft report used by both agents).

## Requirements

- Python 3.10+ (recommended)
- CUDA GPU (for training and inference)
- PyTorch (install according to your local CUDA version)

```bash
pip install -r requirements.txt
```

Main dependencies:

- `torch` / `lightning==2.0.5`
- `transformers==4.30.2` / `peft`
- `Pillow` / `numpy` / `tensorboardX` / `gradio`

You also need the following pretrained assets (paths are configured in `configs/config.py` and `model/R2GenGPT.py`):

| Component | Default / Example |
|-----------|-------------------|
| Vision backbone | `microsoft/swin-base-patch4-window7-224` |
| Language model | `meta-llama/Llama-2-7b-chat-hf` |
| Clinical text encoder | `emilyalsentzer/Bio_ClinicalBERT` |
| Retriever / Reranker | `A_RAG_Agent/bge-m3`, `A_RAG_Agent/bge-reranker-v2-m3` |

## Project Structure

```
MAC_RRG/
├── train.py                 # Entry point for train / validate / test
├── requirements.txt
├── configs/
│   └── config.py            # Hyperparameters and path settings
├── model/
│   ├── R2GenGPT.py          # Main model (vision + KG + RAG + Llama)
│   ├── 3-1.deep_run_iu.sh   # IU-Xray training example
│   ├── 3-2.deep_test_iuxray.sh
│   ├── 6-1.deep_run.sh      # CheXpert Plus training example
│   └── 6-1.deep_test.sh
├── dataset/                 # DataModule and parsers
├── A_MM_KG_Agent/           # Multimodal knowledge-graph agent
│   ├── kg_relations.csv
│   ├── res_dict_aliases.json
│   └── _*.py                # Entity extraction, subgraph building, GAT encoding
├── A_RAG_Agent/             # Retrieval-augmented agent
│   ├── chunks.json          # Knowledge-base text chunks
│   ├── doc_vecs_cache.pt    # Cached document vectors
│   ├── bge-m3 / bge-reranker-v2-m3
│   └── rag_entity_merge.py
├── evalcap/                 # BLEU / ROUGE / METEOR / CIDEr
├── lightning_tools/         # Callbacks, optimizer helpers, etc.
└── data/                    # Datasets (prepare locally)
```

## Data Preparation

1. Download the corresponding images and prepare annotation files, e.g.:
   - IU-Xray: `data/iu_xray/annotation.json`, images under `data/iu_xray/images`
   - MIMIC-CXR: `data/mimic_cxr/annotation.json`, images under `data/mimic_cxr/images`
2. Ensure each sample includes `Draft_text` (typically produced by a Stage-1 baseline model).
3. Make sure RAG resources are ready:
   - `A_RAG_Agent/chunks.json`
   - `A_RAG_Agent/doc_vecs_cache.pt`
   - Local BGE model directories
4. Make sure KG resources are ready:
   - `A_MM_KG_Agent/res_dict_aliases.json`
   - `A_MM_KG_Agent/kg_relations.csv`

> **Note**: `configs/config.py` and `model/R2GenGPT.py` contain absolute server paths (vision/LLM weights, Stage-1 checkpoints, etc.). Update them to your local paths before running.

## Quick Start

### Training

It is recommended to edit and run the provided scripts:

```bash
# IU-Xray
bash model/3-1.deep_run_iu.sh

# CheXpert Plus
bash model/6-1.deep_run.sh
```

Or equivalently:

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

See `configs/config.py` for full argument definitions. Common options:

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name |
| `--annotation` / `--base_dir` | Annotation file and image root |
| `--vision_model` / `--llama_model` | Paths to vision and LLM weights |
| `--freeze_vm` / `--llm_use_lora` / `--vis_use_lora` | Freeze vision encoder or enable LoRA |
| `--savedmodel_path` | Directory for logs and checkpoints |
| `--delta_file` | Checkpoint used for testing |
| `--devices` / `--precision` / `--max_epochs` | Distributed and training controls |

### Testing / Validation

```bash
# Test (requires delta_file)
python -u train.py \
  --test \
  --dataset iu_xray \
  --annotation ./data/iu_xray/annotation.json \
  --base_dir ./data/iu_xray/images \
  --delta_file /path/to/checkpoint.pth \
  --savedmodel_path ./save/iu_xray/test \
  --devices 1

# Validation only
python -u train.py --validate ...
```

You can also use:

```bash
bash model/3-2.deep_test_iuxray.sh
bash model/6-1.deep_test.sh
```

## Evaluation Metrics

The following metrics are computed automatically during validation and testing:

- **BLEU-1 ~ BLEU-4**
- **ROUGE-L**
- **METEOR**
- **CIDEr**

Implementations live in `evalcap/`. By default, `Bleu_4` and `CIDEr` are used as primary selection metrics (`--scorer_types`).

## Core Modules

### `A_MM_KG_Agent`

- Matches anatomy and disorder entities in the draft report (alias dictionary)
- Retrieves related triples / neighbor subgraphs from `kg_relations.csv`
- Encodes nodes and edges with Bio_ClinicalBERT, then produces graph soft prompts via `GraphAttentionEmbedder`

### `A_RAG_Agent`

- `EntityWiseBGESearcher`: entity-wise dense retrieval followed by reranking
- Deduplicates and merges retrieved chunks, then encodes them as external clinical soft prompts

### `model/R2GenGPT.py`

- Assembles visual, KG, and RAG features and concatenates them with image tokens
- Generates reports autoregressively with Llama-2
- Supports loading Stage-1 pretrained weights and `delta_file` checkpoints

## Acknowledgements

This project extends [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) and related radiology report generation / medical NLP work by integrating multi-agent knowledge augmentation for Chest X-ray report generation.

## License

For academic research use only. Please comply with the original licenses of the datasets (IU X-ray, MIMIC-CXR, CheXpert, etc.) and pretrained models (Llama-2, Swin, BGE, etc.) used in this project.
