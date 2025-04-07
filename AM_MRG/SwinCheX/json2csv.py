import json
import pandas as pd

# 定义 JSON 文件的路径
json_file_path = '/wangx/home/E23201049fa/SwinCheX/mimic_chexpert_label.json'

# 定义输出 CSV 文件的路径
train_csv_path = '/wangx/home/E23201049fa/SwinCheX/configs/mimic_chexpert_train.csv'
val_csv_path = '/wangx/home/E23201049fa/SwinCheX/configs/mimic_chexpert_val.csv'
test_csv_path = '/wangx/home/E23201049fa/SwinCheX/configs/mimic_chexpert_test.csv'

# 读取 JSON 数据
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 定义 CSV 文件的列名
columns = [
    "Image Index", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", 
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", 
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

# 将数据转换为 DataFrame
def json_to_dataframe(json_list):
    # 提取 'image_path' 和 'label' 的值
    data_list = []
    for item in json_list:
        base_path = item['base_path']
        image_path = item['image_path'][0]
        image_path = [base_path + '/' + image_path]
        label = item['label']
        data_list.append(image_path + label)
    # 创建 DataFrame
    df = pd.DataFrame(data_list, columns=columns)
    return df

# 处理 train、val 和 test 数据
train_df = json_to_dataframe(data['train'])
val_df = json_to_dataframe(data['val'])
test_df = json_to_dataframe(data['test'])

# 保存为 CSV 文件
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"train.csv 已保存到 {train_csv_path}")
print(f"val.csv 已保存到 {val_csv_path}")
print(f"test.csv 已保存到 {test_csv_path}")
