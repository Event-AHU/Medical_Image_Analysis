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

# print(len(test_list))
# print(test_list[0])


select_list = []
for i in range(0, 5000, 50):
    test_list[i]['test_list_id']=i
    select_list.append(test_list[i])

print(len(select_list))
print(select_list)

for i in range(len(select_list)):
    # # 源文件路径
    path = select_list[i]['image_path'][0]
    # source_file = path
    # # 目标文件夹路径
    # destination_folder = '/gpfs/home/22054/image_100/'
    # # 复制图像到目标文件夹
    # shutil.copy2(source_file, destination_folder)
    str_list = path.split('/')
    name = str_list[len(str_list)-1]
    name = '/gpfs/home/22054/image_100/'+name
    print('序号：',i*50,'名称：',name)
    

print('完成')
    
    


