import os
import json
import re
# import jieba
from collections import Counter
from transformers import AutoTokenizer, BertTokenizer
 
# import torch.nn as nn 
# class Tokenizer(nn.Module):
class Tokenizer(object):
    def __init__(self, args):
        # super(Tokenizer, self).__init__()
        self.args = args
        self.ann_path = args.ann_path
        self.vocab_path = self.ann_path.replace('.json', '_vocab.json')
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        elif self.dataset_name == 'mimic_cxr':
            self.clean_report = self.clean_report_mimic_cxr
        else:
            self.clean_report = self.clean_report_chinese
        self.ann = json.loads(open(self.ann_path, mode='r', encoding='utf-8').read())
        self.tokenizer=AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        # self.tokenizer=BertTokenizer.from_pretrained('pretrain_models/chinese-roberta-wwm-ext')        # pretrain_models/bert
        self.token2idx, self.idx2token = self.create_vocabulary()
        
    def create_vocabulary(self):
        tokenizer = self.tokenizer
        token2idx = tokenizer.vocab    #vocab
        idx2token = {value:key for key, value in token2idx.items()} 
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        print(report)
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report.split()

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report.split()
        
    def clean_report_chinese(self, report):
        report_cleaner1 = lambda t: re.sub('\d[.、](?!\d)', '', re.sub('\s+', '', t))
        report_cleaner2 = lambda t: t.replace('/', '，').replace(' ', '').replace('"', '').replace(',', '，').replace(':', '：').replace(';', '，')
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        return self.tokenizer(self.clean_report(report), padding=True, truncation=True, return_tensors="pt", max_length=self.args.max_seq_length)

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx not in [self.args.bos_idx, self.args.eos_idx, self.args.pad_idx]:
                if i >= 1:
                    if self.dataset_name == 'iu_xray' or self.dataset_name == 'mimic_cxr':
                        txt += ' '
                    else:
                        # txt += ''
                        txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out