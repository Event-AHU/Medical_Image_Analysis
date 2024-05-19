import os  
  
# 指定要搜索的文件夹路径  
folder_path = '/Share/home/22054/pbc_dataset_medical/rsna-pneumonia-detection-challenge_classification/stage_2_train_images'  
  
# 指定要保存文件路径的txt文件路径  
output_file = 'train.txt'  
  
# 用于保存图像文件路径的列表  
image_paths = []  

print('kaishi')
  
# 遍历指定文件夹及其所有子文件夹  
for root, dirs, files in os.walk(folder_path):  
    for file in files:  
        # 检查文件是否为图像文件（这里假设图像文件的后缀为.jpg、.jpeg、.png）  
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):  
            # 构建完整的文件路径  
            file_path = os.path.join(root, file)  
            # 将文件路径添加到列表中  
            image_paths.append(file_path)  

print("jieshu")
  
# 将图像文件路径写入txt文件  
with open(output_file, 'w') as f:  
    for path in image_paths:  
        f.write(path + '\n')
