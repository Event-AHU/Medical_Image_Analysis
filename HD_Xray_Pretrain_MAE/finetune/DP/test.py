import os
import os
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

set_seed(605)
attr_words = [
    'yellow','orange', 'green', 'gray', 'red', 'blue', 'white','golden','brown','black',
    'sedan', 'suv', 'van','hatchback','mpv','pickup','bus','truck','estate'
]#color=10,type=9
def keshihua(gt_label, preds_probs,image_name):
    t=0
    pred_label = preds_probs > 0.45
    pred_attr=[[] for _ in range(len(pred_label))]
    for row_step,row in enumerate(pred_label) :
        for col_step,col in enumerate(row) :
            if col :
                pred_attr[row_step].append(attr_words[col_step])
    g_label = gt_label > 0.5
    gt_attr=[[] for _ in range(len(pred_label))]
    for row_step,row in enumerate(g_label) :
        for col_step,col in enumerate(row) :
            if col :
                pred_attr[row_step].append(attr_words[col_step])   
    #g_label = gt_label > 0.5
    name_pred = {key:value for key, value in zip(image_name, pred_attr)}
    for key,value in name_pred.items() :
        print(str(key)+':'+str(value))
    '''
    for i in range(0,len(gt_label)):
        pre_resout = pred_label[i]
        label = g_label[i]
        for j in range(0,len(pre_resout)):
            if pre_resout[j] is not label[j]:
                t = 1
        if t == 1:
            print(pre_resout)
            print(label)
            t =0
        '''


def main(args):
    #log_dir = '/DATA/wuwentao/data/VeRi' #os.path.join('/DATA/wuwentao/data/', args.dataset)/DATA/wuwentao/data/VehicleID/
    log_dir = '/DATA/wuwentao/data/VeRi'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    select_gpus(args.gpus)
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=64,#args.batchsize
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=64,#args.batchsize
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    model = TransformerClassifier(train_set.attr_num)
    model.load_state_dict(torch.load('/DATA/wuwentao/VTB-main/model_mae_lunkuo_clip/ckpt_2023-06-23_10_11_53_28.pth')['state_dicts'])
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    
    lr = args.lr
    epoch_num = args.epoch

    optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)

    best_metric, epoch = trainer(epoch=epoch_num,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 path=log_dir)
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.functional.classification import multilabel_auroc
def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, scheduler, path):
    for i in range(1, epoch+1):
        scheduler.step(i)
        '''
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )
        '''
        valid_loss, valid_gt, valid_probs,image_name = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )
        keshihua(valid_gt,valid_probs,image_name)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
        
        auroc_mean = multilabel_auroc(p_test, y_test, num_labels=args.num_classes,
                                 average="macro").cpu().numpy()
        auroc_individual = multilabel_auroc(p_test, y_test, num_labels=args.num_classes,
                                 average=None).cpu().numpy()
        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f},text_loss{:.4f}\n'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1,valid_loss))

        print('-' * 60)
        if i % args.epoch_save_ckpt:
            save_ckpt(model, os.path.join('/DATA/wuwentao/VTB-main/model', f'ckpt_{time_str()}_{i}.pth'), i, valid_result)

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
