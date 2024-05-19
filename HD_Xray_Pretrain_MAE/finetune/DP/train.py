import os
import os
import pprint
from collections import OrderedDict, defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
#from config import argument_parser
from dataset.AttrDataset import MultiModalAttrDataset, get_transform
from loss.CE_loss import *
from models.base_block import *
from tools.function import get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, select_gpus

from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

set_seed(605)

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, default="Medical")
    parser.add_argument("--dataset_pkl_path", type=str)
    parser.add_argument("--mode_save_path", type=str)
    parser.add_argument("--pretrain_path", type=str)
    
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--lr", type=float, default=4e-3)   #8e-3
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    
    # parser.add_argument('--gpus', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument("--epoch_save_ckpt", type=int, default=1)
    
    return parser

def main(args):
    #log_dir = '/DATA/wuwentao/data/VeRi' #os.path.join('/DATA/wuwentao/data/', args.dataset)/DATA/wuwentao/data/VehicleID/
    data_path = args.dataset_pkl_path 
    log_dir = os.path.join(args.mode_save_path, 'log_stdout')
    mode_save_path = args.mode_save_path
    if not os.path.exists(mode_save_path):
        os.mkdir(mode_save_path)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)    
        
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    # select_gpus(args.gpus)
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = MultiModalAttrDataset(args=args, split=args.train_split, transform=train_tsfm,data_path=data_path)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,#args.batchsize
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )

    valid_set = MultiModalAttrDataset(args=args, split=args.valid_split, transform=valid_tsfm,data_path=data_path)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,#args.batchsize
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    model = TransformerClassifier(train_set.attr_num,pretrain_path = args.pretrain_path)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = CEL_Sigmoid(sample_weight, attr_idx=train_set.attr_num)
    
    lr = args.lr
    epoch_num = args.epoch

    optimizer = make_optimizer(model, lr=lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(optimizer, num_epochs=epoch_num, lr=lr, warmup_t=5)

    best_loss = trainer(epoch=epoch_num,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 path=log_dir,num_classes_name=train_set.attr_id,mode_save_path=mode_save_path)

# from torchmetrics.classification import MultilabelAccuracy
# from torchmetrics.functional.classification import multilabel_auroc
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    # gt_np = gt.cpu().numpy()
    # pred_np = pred.cpu().numpy()
    gt_np = gt
    pred_np = pred
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, scheduler, path,num_classes_name,mode_save_path):
    best_loss = 100
    for cur_epoch in range(1, epoch+1):
        scheduler.step(cur_epoch)
        
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=cur_epoch,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss, valid_gt, valid_probs,name= valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)
        new_loss = float(valid_loss)
        if best_loss>new_loss:
            best_loss = new_loss
        print('Evaluation on test set,\nnew_loss:{:.4f},best_loss:{:.4f}'.format(valid_loss,best_loss))
        
        gt, pred = valid_gt, valid_probs
        target_class = num_classes_name
        AUROCs = compute_AUCs(gt, pred,len(target_class))
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        for i in range(len(target_class)):
            print('The AUROC of {} is {}'.format(target_class[i], AUROCs[i]))
        max_f1s = []
        accs = []
        for i in range(len(target_class)):   
            gt_np = gt[:, i]
            pred_np = pred[:, i]
            precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
            numerator = 2 * recall * precision
            denom = recall + precision
            f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
            max_f1 = np.max(f1_scores)
            max_f1_thresh = thresholds[np.argmax(f1_scores)]
            max_f1s.append(max_f1)
            accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))
            
        f1_avg = np.array(max_f1s).mean()    
        acc_avg = np.array(accs).mean()
        print('The average f1 is {F1_avg:.4f}'.format(F1_avg=f1_avg))
        print('The average ACC is {ACC_avg:.4f}'.format(ACC_avg=acc_avg))

        if(best_loss == new_loss):
            save_ckpt(model, os.path.join(mode_save_path, 'ckpt_ AUROCavg{:.4f}_cur_epoch{}.pth'.format(AUROC_avg,cur_epoch)), cur_epoch, valid_result)
            print('saveing model: ' + os.path.join(mode_save_path, 'ckpt_ AUROCavg{:.4f}_cur_epoch{}.pth'.format(AUROC_avg,cur_epoch)))
        print('-------------------------')
    return best_loss
if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)
