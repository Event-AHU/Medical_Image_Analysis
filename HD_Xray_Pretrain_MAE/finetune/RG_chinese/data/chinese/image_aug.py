import cv2
import numpy as np
from tqdm import tqdm
import json

ann_path = "/gpfs/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation_20w.json"
data = json.loads(open(ann_path, 'r', encoding='utf-8').read())
train_list = data['train']
val_list = data['val']
test_list = data['test']

a = 0
b = 0
c = 0
select_list = []
for i in range(len(train_list)):
    image_finding = train_list[i]['image_finding']
    if "积液" in image_finding and "未见明显积液" not in image_finding and "未见积液" not in image_finding and "无积液" not in image_finding and "无胸腔积液" not in image_finding:
        a = a + 1
        select_list.append(train_list[i])
        # print(train_list[i]['image_finding'])
    elif "斑片状影" in image_finding and "未见明显斑片状影" not in image_finding:
        b = b + 1
        select_list.append(train_list[i])
        # print(train_list[i]['image_finding'])
    elif "膈肌消失" in image_finding:
        c = c + 1
        select_list.append(train_list[i])
        # print(train_list[i]['image_finding'])
        
# print(a)
# print(b)
# print(c)
# print(select_list)
# print(len(select_list))

save_path = '/gpfs/home/22054/medical_xray_pretrain/image_aug/'
aug_list = []
# aug_dict = {}

for index in tqdm(range(len(select_list))):
# for index in tqdm(range(2)):
    path = select_list[index]['image_path'][0]
    print(path)

    # 读取图像
    # path = '/gpfs/home/22054/image_A800/1.2.156.112605.50118241486.20220406001007.4.1456.3.jpg'
    image = cv2.imread(path)
    list = path.split("/")
    image_name = list[len(list)-1]
    image_name = image_name[: -4]

    # 随机裁剪
    for i in tqdm(range(10)):
        name1 = image_name
        h, w = image.shape[:2]
        start_x = np.random.randint(0, w // 2)
        start_y = np.random.randint(0, h // 2)
        end_x = np.random.randint(w // 2, w)
        end_y = np.random.randint(h // 2, h)
        cropped_image = image[start_y:end_y, start_x:end_x]
        cropped_image = cv2.resize(cropped_image, (w, h))  # 还原为原始图像大小
        name1 = name1 + f'_cropped_{i}.jpg'
        name1 = save_path + name1
        cv2.imwrite(name1, cropped_image)
        js1 = select_list[index].copy()
        # js1['id'] = js1['id'] + f'_cropped_{i}'
        js1['image_path'] = [name1]
        # print(js1)
        aug_list.append(js1)
        


    # 随机旋转
    # for i in range(10):
        name2 = image_name
        h, w = image.shape[:2]
        angle = np.random.randint(-100, 100)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        rotated_image = cv2.resize(rotated_image, (w, h))  # 还原为原始图像大小
        name2 = name2 + f'_rotated_{i}.jpg'
        name2 = save_path + name2
        cv2.imwrite(name2, rotated_image)
        js2 = select_list[index].copy()
        # js2['id'] = js2['id'] + f'_rotated_{i}'
        js2['image_path'] = [name2]
        # print(js2)
        aug_list.append(js2)
        

    # 随机翻转
    # for i in range(10):
        name3 = image_name
        flip_code = np.random.choice([0,1,-1])  # 随机选择垂直翻转、水平翻转和水平垂直同时翻转
        flipped_image = cv2.flip(image, flip_code)
        name3 = name3 + f'_flipped_{i}.jpg'
        name3 = save_path + name3
        cv2.imwrite(name3, flipped_image)
        js3 = select_list[index].copy()
        # js3['id'] = js3['id'] + f'_flipped_{i}'
        js3['image_path'] = [name3]
        # print(js3)
        aug_list.append(js3)

    # 调整亮度和对比度
    # for i in range(10):
        name4 = image_name
        alpha = np.random.uniform(0.5, 1.5)  # 随机选择一个亮度因子
        beta = np.random.randint(-50, 50)    # 随机选择一个对比度调节值
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        name4 = name4 + f'_adjusted_{i}.jpg'
        name4 = save_path + name4
        cv2.imwrite(name4, adjusted_image)
        js4 = select_list[index].copy()
        # js4['id'] = js4['id'] + f'_adjusted_{i}'
        js4['image_path'] = [name4]
        # print(js4)
        aug_list.append(js4)

    # 添加高斯噪声
    # for i in range(10):
        name5 = image_name
        mean = 0
        std_dev = np.random.randint(1, 50)
        noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        name5 = name5 + f'_noisy_{i}.jpg'
        name5 = save_path + name5
        cv2.imwrite(name5, noisy_image)
        js5 = select_list[index].copy()
        # js5['id'] = js5['id'] + f'_noisy_{i}'
        js5['image_path'] = [name5]
        # print(js5)
        aug_list.append(js5)

print(len(train_list))
train_list.extend(aug_list)
print(len(train_list))

data['train'] = train_list
with open("/gpfs/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation_20w_aug.json", "w") as file:
    # 将字典转换为json格式
    json_str = json.dumps(data, ensure_ascii=False)
    
    # 将json字符串写入文件
    file.write(json_str)
# 关闭文件
file.close()
print('完成')
        
        
        
        
        

    # print(aug_list)
    # print(len(aug_list))