# import json
# import os
# import shutil

# source_file = "/gpfs/home/22054/medical_xray_pretrain/ALL_DATA/b981053a21ff11eca49c2cea7fb3393a.jpg"
# # 目标文件夹路径
# destination_folder = '/gpfs/home/22054/image_A800/'
# # 复制图像到目标文件夹
# shutil.copy2(source_file, destination_folder)

# import os

# # 文件夹路径
# folder_path = '/gpfs/home/22054/medical_xray_pretrain/2023-07/'

# # 列出文件夹中的所有文件
# files = os.listdir(folder_path)

# # 获取文件数量
# num_files = len(files)

# print(f'文件夹中的文件数量为: {num_files}')

import json
import os
import shutil

# with open("/gpfs/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation.json", "r") as file:
    
#     data = json.load(file)
ann_path = "/gpfs/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation_20w.json"
data = json.loads(open(ann_path, 'r', encoding='utf-8').read())

train_list = data['train']
val_list = data['val']
test_list = data['test']

print(test_list[2726])