from PIL import Image
from torch.utils.data import Dataset
import pickle
import os
import torch
from typing import Tuple, Optional, Union
class MyDatasets(Dataset):
    def __init__(self, pkl_path, root_path,args,transform = None, target_transform = None):
        dataset_info = pickle.load(open(pkl_path, 'rb+'))
        self.image_names=dataset_info.image_name
        #self.labels=dataset_info.labels
        self.caption_impressions=dataset_info.caption_impressions
        self.caption_saw=dataset_info.captions_saw
        self.impress_saw=[self.caption_impressions[i]+self.caption_saw[i] for i in range(len(self.caption_impressions))]
        self.root=root_path
        self.transform=transform
    def __getitem__(self, index)-> Tuple[torch.Tensor, ...]:
        image_names= self.image_names[index]
        impress_saw=self.impress_saw[index]
        img = Image.open(os.path.join(self.root,image_names))
        if self.transform is not None:
            img = self.transform(img) 
        return img,impress_saw
    def __len__(self):
        return len(self.image_names)