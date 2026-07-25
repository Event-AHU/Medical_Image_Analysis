import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch.nn.functional as F


def encode_texts(texts, tokenizer, model, device, batch_size=16):

    all_embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state   # (B, L, 768)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        masked_hidden = last_hidden * attention_mask
        sum_hidden = masked_hidden.sum(1)
        lengths = attention_mask.sum(1)

        embeddings = sum_hidden / lengths  # mean pooling

        all_embeddings.append(embeddings.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)

    # normalize (optional but recommended for retrieval)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


if __name__ == "__main__":

    # ===== paths =====
    json_path = "/wangx/home/E24301191/mycode/MAC_RRG/A_RAG_Agent/merged_chunks.json"
    model_name = "/wangx_nas/QYH/emilyalsentzer/Bio_ClinicalBERT/"

    # ===== device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== load chunks =====
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]

    # print("Number of chunks:", len(chunks))

    # ===== load model =====
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    # ===== encode =====
    embeddings = encode_texts(
        chunks,
        tokenizer,
        model,
        device,
        batch_size=16
    )

    print("Embedding shape:", embeddings.shape)
