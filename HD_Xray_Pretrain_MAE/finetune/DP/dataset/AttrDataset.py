import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image

from tools.function import get_pkl_rootpath
import torchvision.transforms as T


class MultiModalAttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None,data_path=None):

        # assert args.dataset in [ 'ourdata','vehicleid','vrai','ChestXray14'], \
        #     f'dataset name {args.dataset} is not exist'
        #data_path = "/DATA/wuwentao/data/vrai/dataset_vrai.pkl"

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label
        
        self.label_vector = dataset_info.attr_vectors
        self.label_word = dataset_info.attr_words

        self.words = self.label_word.tolist()
        # if split == 'trainval':
        # percent = 1
        # print(f'data: {percent}%')
        # index = (len(self.img_id) *percent)// 100
        # self.img_id, self.label, self.img_idx = self.img_id[:index], self.label[:index], self.img_idx[:index]
    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath).convert('L')
        
        if self.transform is not None:
            img = self.transform(img)
        # print(img.size, gt_label.shape, imgname,self.words)
        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        label_v = self.label_vector.astype(np.float32)
        
        return img, gt_label, imgname, label_v, self.words

    def __len__(self):
        return len(self.img_id)

def get_transform(args):
    height = args.height
    width = args.width
    #normalize = T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    normalize = T.Normalize(mean=[0.5], std=[0.5])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
