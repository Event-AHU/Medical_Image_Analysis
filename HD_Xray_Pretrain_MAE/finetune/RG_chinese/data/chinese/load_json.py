import json
import os

# with open("/gpfs/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation.json", "r") as file:
    
#     data = json.load(file)
ann_path = "/gpfs/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation.json"
data = json.loads(open(ann_path, 'r', encoding='utf-8').read())

train_list = data['train']
val_list = data['val']
test_list = data['test']

not_find_train = []
not_find_val = []
not_find_test = []

for i in range(len(train_list)):
    # breakpoint()
    # print(train_list[i])
    str_list = train_list[i]['image_path'][0].split('/')
    image_name = str_list[len(str_list)-1]
    flod_name = str_list[len(str_list)-2]
    image_path = ["/gpfs/home/22054/medical_xray_pretrain/CID-class_20W/CID_ALL/" + image_name]
    train_list[i]['image_path'] = image_path
    file_path = "/gpfs/home/22054/medical_xray_pretrain/CID-class_20W/CID_ALL/" + image_name
    if not os.path.exists(file_path):
        not_find_train.append(i)
        print('无法找到图片', file_path)
        print(flod_name)
    
for i in range(len(val_list)):
    # breakpoint()
    # print(val_list[i])
    str_list = val_list[i]['image_path'][0].split('/')
    image_name = str_list[len(str_list)-1]
    flod_name = str_list[len(str_list)-2]
    image_path = ["/gpfs/home/22054/medical_xray_pretrain/CID-class_20W/CID_ALL/" + image_name]
    val_list[i]['image_path'] = image_path
    file_path = "/gpfs/home/22054/medical_xray_pretrain/CID-class_20W/CID_ALL/" + image_name
    if not os.path.exists(file_path):
        not_find_val.append(i)
        print('无法找到图片', file_path)
        print(flod_name)

for i in range(len(test_list)):
    # breakpoint()
    # print(test_list[i])
    str_list = test_list[i]['image_path'][0].split('/')
    image_name = str_list[len(str_list)-1]
    flod_name = str_list[len(str_list)-2]
    image_path = ["/gpfs/home/22054/medical_xray_pretrain/CID-class_20W/CID_ALL/" + image_name]
    test_list[i]['image_path'] = image_path
    file_path = "/gpfs/home/22054/medical_xray_pretrain/CID-class_20W/CID_ALL/" + image_name
    if not os.path.exists(file_path):
        not_find_test.append(i)
        print('无法找到图片', file_path)
        print(flod_name)
        
print(len(train_list))     
train_list.pop(not_find_train[0])
print(len(train_list))   

print(len(test_list))   
test_list.pop(not_find_test[0])
print(len(test_list)) 
# print(not_find_train)
# print(not_find_val)
# print(not_find_test)

all_dict = {}
all_dict['train'] = train_list
all_dict['val'] = val_list
all_dict['test'] = test_list

with open("/gpfs/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation_20w.json", "w") as file:
    # 将字典转换为json格式
    json_str = json.dumps(all_dict, ensure_ascii=False)
    
    # 将json字符串写入文件
    file.write(json_str)
# 关闭文件
file.close()
print('完成')


