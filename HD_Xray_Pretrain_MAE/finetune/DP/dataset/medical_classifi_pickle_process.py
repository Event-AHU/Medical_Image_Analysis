import os
import numpy as np
import random
import pickle
import json
from easydict import EasyDict
from xml.dom.minidom import parse

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
np.random.seed(0)
random.seed(0)

vehicleids_train = []
vehicleids_test = []

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def get_label_embeds(labels,label_SentenceTransformer_path):
    model = SentenceTransformer('all-mpnet-base-v2',cache_folder=label_SentenceTransformer_path)
    embeddings = model.encode(labels)
    return embeddings


data_json_path = './RSNA_Pneumonia'
dataset_folder = '/wangx/_dataset/rsna-pneumonia-detection-challenge_classification/'
# label_name_path = data_json_path+'/label_name.txt'
label_name_path = 'label_name.txt'
train_json_file_path = data_json_path+"/train.json"
test_json_file_path = data_json_path+"/test.json"
label_SentenceTransformer_path = r"allenai-specter"

dataset = EasyDict()    
stage = 'train'
# stage = 'test'

dataset.root = ''   #数据集数据地址 save_dir
dataset.description = 'rsna-pneumonia_100'
file_csv = data_json_path + f'/{stage}_list.txt'

# dataset.description = 'rsna-pneumonia_10'
# file_csv = data_json_path + f'/{stage}_10.txt'

dataset.description = 'rsna-pneumonia_1'
file_csv = data_json_path + f'/{stage}_1.txt'


data = pd.read_csv(file_csv,sep=' ',header= None).fillna(0)
import pandas as pd
data1 = pd.DataFrame(data.iloc[:,[0,1]])
data1.rename(columns={0:'img_path',1:'target'},inplace=True)
class_name = ['Lung Opacity']
data1['target'] = data1['target'].apply(lambda x : [x])
all_data_path =  f'{stage}_images_dir/'
data1['img_path'] = data1['img_path'].apply(lambda x : x.replace(f'stage_2_{stage}_images/',all_data_path).replace('.dcm','.png'))
# data1['target'] = data1['target'].apply(lambda x:list(x))
# 将 DataFrame 转换为字典列表  
file_path = data_json_path+f'/{stage}.json'
# file_path = data_path+'train.json'
data_list = data1.to_dict(orient='records')  
  
# 使用 json.dumps 来确保字符串被正确转义  
json_data = json.dumps(data_list)  
print(f"Class names have been saved to {file_path}")
# 将转义后的 JSON 数据保存到文件中  
with open(file_path, 'w') as f:  
    f.write(json_data)
    
file_path = data_json_path+ '/label_name.txt'
with open(file_path, 'w') as file:
    file.write('\n'.join(class_name))

print(f"Class names have been saved to {file_path}")



with open(train_json_file_path, 'r') as json_file:
    train_data_list = json.load(json_file)

with open(test_json_file_path, 'r') as json_file:
    test_data_list = json.load(json_file)

list_name_train = []
list_name_test = []
train_gt_list=[]
test_gt_list=[]
attr_words=[]

for item in train_data_list:
    a_train = item["target"]
    path = dataset_folder+ item["img_path"]
    #file_name_train = path.replace(prefix_to_remove, "/data/wangxiao/liuming/append_data")
    #file_name_train = os.path.basename(item["img_path"])  # 提取文件名部分
    list_name_train.append(path)
    train_gt_list.append(a_train)

for item in test_data_list:
    a_test = item["target"]
    path = dataset_folder+ item["img_path"]
    #file_name_test = path.replace(prefix_to_remove, "/data/wangxiao/liuming/append_data")
    #file_name_test = os.path.basename(item["img_path"])  # 提取文件名部分
    list_name_test.append(path)
    test_gt_list.append(a_test)

print(len(train_gt_list))
print(len(test_gt_list))

train_len = len(list_name_train)
test_len=len(list_name_test)
dataset.image_name = list_name_train + list_name_test

attr_file=open(os.path.join(data_json_path, label_name_path),'r',encoding='utf8')    #open(os.path.join(save_dir, 'train_label.xml'))
for attr in tqdm(attr_file.readlines()) :
    curLine=attr.strip('\n')
    attr_words.append(curLine)
dataset.attributes=attr_words 
dataset.attr_name = attr_words

dataset.label = np.concatenate((np.array(train_gt_list),np.array(test_gt_list)), axis=0)
assert dataset.label.shape == (train_len+test_len, len(attr_words))

dataset.attr_words = np.array(attr_words)
dataset.attr_vectors = get_label_embeds(attr_words,label_SentenceTransformer_path)

dataset.partition = EasyDict()
dataset.partition.test = np.arange(train_len, train_len+test_len)  # np.array(range(90000, 100000))
dataset.partition.trainval = np.arange(0, train_len)  # np.array(range(90000))
dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)
save_path = os.path.join(data_json_path, f'{dataset.description}_data.pkl')
with open(save_path, 'wb+') as f:
    pickle.dump(dataset, f)     #生成数据集的pkl文件
print(f'save to {save_path}')
# generate_data_description()
