import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json, pickle

# ================= 配置 =================
REPORT_LABELS = np.array([
    'Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia',
    'Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices','No Finding'
])

json_path = 'mimic_label.json'
save_path = 'report_memory.pkl'

LLM_path = 'emilyalsentzer/Bio_ClinicalBERT'

batch_size = 32       
max_length = 256      

# ================= 加载数据 =================
with open(json_path, 'r') as f:
    datas = json.load(f)['test']

tokenizer = AutoTokenizer.from_pretrained(LLM_path)
tokenizer.bos_token_id = tokenizer.cls_token_id
text_encoder = AutoModel.from_pretrained(LLM_path).cuda()
text_encoder.eval()   # 固定模型为 eval 模式

REPORT_DICT = {label: [] for label in REPORT_LABELS}
LABEL_COUNT = {label: 0 for label in REPORT_LABELS}  

# ================= 批处理提取特征 =================
def get_text_features(reports):
    tokens = tokenizer(
        reports,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    ).to("cuda")

    with torch.no_grad():
        outputs = text_encoder(tokens["input_ids"], attention_mask=tokens["attention_mask"])
        hidden_states = outputs["last_hidden_state"]   
        cls_features = hidden_states[:, 0, :]          
    return cls_features.cpu().numpy()


# ================= 遍历数据 =================
all_reports, all_labels = [], []

for data in datas:
    all_reports.append(data['report'])
    all_labels.append(np.array(data['label']))

# 分批处理
for i in tqdm(range(0, len(all_reports), batch_size), desc="Processing reports"):
    batch_reports = all_reports[i:i+batch_size]
    batch_labels = all_labels[i:i+batch_size]

    features = get_text_features(batch_reports)  

    for j, labels in enumerate(batch_labels):
        selected_labels = REPORT_LABELS[labels == 1]
        for label in selected_labels:
            REPORT_DICT[label].append(features[j:j+1]) 
            LABEL_COUNT[label] += 1


# ================= 转换为 numpy =================
for label in REPORT_DICT:
    if REPORT_DICT[label]:  
        REPORT_DICT[label] = np.vstack(REPORT_DICT[label])
    else:
        REPORT_DICT[label] = np.empty((0, text_encoder.config.hidden_size))


# ================= 保存结果 =================
with open(save_path, 'wb') as f:
    pickle.dump(REPORT_DICT, f)

print("保存路径:", save_path)
print("\n每个标签的样本数量统计：")
for label, count in LABEL_COUNT.items():
    print(f"{label:25s} : {count}")