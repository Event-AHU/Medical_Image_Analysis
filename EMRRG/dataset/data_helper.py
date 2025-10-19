import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        # self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
        # self.vit_feature_extractor = AutoImageProcessor.from_pretrained('./swin_base_patch4_window7_224')
        # self.vit_feature_extractor = AutoImageProcessor.from_pretrained('/wangx/R2GenGPT-main/swin_base_patch4_window7_224')
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained('./swin_base_patch4_window7_224')

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt", size=self.args.input_size).pixel_values
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
            # report_cleaner = lambda t: re.sub('\d[.、](?!\d)', '', re.sub('\s+', '', t))
            # report_cleaner = lambda t: t.replace('/', '，').replace(' ', '').replace('"', '').replace(',', '，').replace(':', '：').replace(';', '，')
        # clean MIMIC-CXR reports
        else:
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

            # report = ' '.join(report.split()[:self.args.max_txt_len])
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


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        # self.meta = json.load(open(args.annotation, 'r'))
        self.meta = json.loads(open(args.annotation, 'r', encoding='utf-8').read())

        # self.meta = self.meta[split][:100]
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


