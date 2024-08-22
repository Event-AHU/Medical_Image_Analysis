
import os
import json
import random
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
import pandas as pd

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        # The processor is used to transform to nomralize the images, not a visual encoder
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')

 
    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt",size=self.args.input_size).pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        
        elif self.dataset == "chinese":
            None
        # clean MIMIC-CXR reports
        elif self.dataset =='mimic_cxr':
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        else:
            None
        return report


    def parse(self, features):
        
        if self.dataset == "chinese":
            to_return = {'id': str(features['id'])}
            report = features.get("image_finding", "")
        else:
            to_return = {'id': features['id']}
            report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)

# from line_profiler import profile
class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        # self.meta = json.load(open(args.annotation, 'r'))
        self.meta = json.loads(open(args.annotation, 'r', encoding='utf-8').read())
        self.split = split
        if self.args.drop_unclear_report and split=='train':
            mm = pd.DataFrame(self.meta['train'])
            drop_unclear_report_before = len(mm)
            mm = mm[~mm['report'].str.contains('_')]
            mm = mm[mm['report'].apply(lambda x: len(x.split(' ')))>3]
            self.meta['train'] = mm.to_dict('records')
            drop_unclear_report_after = len(mm)
            print(f'len---drop_unclear_report_before: {drop_unclear_report_before}')
            print(f'len---drop_unclear_report_after: {drop_unclear_report_after}')

        if self.args.use_feature_mean is False and split=='train' and self.args.dataset == "mimic_cxr":
            mm = pd.DataFrame(self.meta['train'])
            gb =  mm.groupby('study_id')['image_path'].apply(lambda x: sum(x, [])).reset_index()
            self.merged_df = pd.merge(mm, gb, on='study_id',suffixes = ("_single", ""))
        
        # self.meta = self.meta[split][:100]
        self.meta = self.meta[split]
        # if self.args.dataset == "chinese" and split!='train':
        #     self.meta = self.meta[:len(self.meta)//10] # for quick test
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)
    # @profile
    def __getitem__(self, index):
        if self.args.use_feature_mean  is False and self.split=='train' and self.args.dataset == "mimic_cxr":
            img_path_num = len(self.merged_df.iloc[index]['image_path'])
            # print(self.meta[index]['image_path'],self.merged_df.iloc[index]['image_path'])
            if img_path_num ==2 :
                self.meta[index]['image_path']= self.merged_df.iloc[index]['image_path']
            elif img_path_num >2:
                self.meta[index]['image_path']= self.meta[index]['image_path']+ [random.choice(self.merged_df.iloc[index]['image_path'])]
            else:
                self.meta[index]['image_path']=self.meta[index]['image_path']+self.meta[index]['image_path']
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    
    if args.dev_form =='test':
        dev_dataset = ParseDataset(args, 'test')
    else:
        dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


