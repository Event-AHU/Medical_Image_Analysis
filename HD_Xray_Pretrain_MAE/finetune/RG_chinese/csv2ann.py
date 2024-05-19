import os
import json
import random
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

src_path = 'data/chinese/all_30k_data.csv'
dst_path = 'data/chinese/'
suffix = '/media/20TB/hx/chestxray/liuming/'


def restructure_img_path(data):
    res_dict = {}
    values = list(data.values)
    res_dict['id'] = values[0]
    res_dict['image_finding'] = values[1]
    res_dict['impressions'] = values[2]
    if not values[3].startswith('/media/dataset'):
        res_dict['image_path'] = [suffix + values[3]]
    else:
        res_dict['image_path'] = [values[3]]
    return res_dict


def train_test_val_split(data, dst_path, ratio_train=0.7, ratio_val=0.1, ratio_test=0.2):
    random.shuffle(data)
    train, middle = train_test_split(data, test_size=1 - ratio_train)
    ratio = ratio_val / (1 - ratio_train)
    test, val = train_test_split(middle, test_size=ratio)

    for item in tqdm(train):
        item['split'] = 'train'

    for item in tqdm(val):
        item['split'] = 'val'
    
    for item in tqdm(test):
        item['split'] = 'test'

    res = {'train': train, 'val': val, 'test': test}

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    dst_path += 'chinese_annotation.json'
    with open(dst_path, encoding='utf-8', mode='w') as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == '__main__':
    csv_data = pd.read_csv(src_path, usecols=['uid', 'image_finding', 'impressions', 'id'])
    examples = []
    count = 0
    for index, row in tqdm(csv_data.iterrows()):
        if row['image_finding'] == '[]':
            count += 1
        else:
            examples.append(restructure_img_path(row))
    print(f'不规格数据{count}条！')
    train_test_val_split(examples, dst_path)
