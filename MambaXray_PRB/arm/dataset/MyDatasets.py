from PIL import Image
from torch.utils.data import Dataset
import pickle
import os
import torch
from typing import Tuple, Optional, Union
class MyDatasets(Dataset):
    def __init__(self, data_path,transform = None, target_transform = None):
        dataset_info = pickle.load(open(data_path, 'rb+'))
        self.image_path = dataset_info.image_path
        self.impress_saw = dataset_info.report
        self.transform=transform
    def __getitem__(self, index)-> Tuple[torch.Tensor, ...]:
        impress_saw=self.impress_saw[index]
        img = Image.open(self.image_path[index])
        if self.transform is not None:
            img = self.transform(img) 
        return img,impress_saw
    def __len__(self):
        return len(self.image_path)