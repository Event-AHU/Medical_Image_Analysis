import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.utils import AverageMeter, to_scalar, time_str


def batch_trainer(epoch, model, train_loader, criterion, optimizer):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    lr = optimizer.param_groups[0]['lr']

    for step, (imgs, gt_label, imgname, label_v, label_n) in enumerate(train_loader):

        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        label_v = label_v[0].cuda()
        
        train_logits = model(imgs, label_v, gt_label)
        train_loss = criterion(train_logits, gt_label)

        train_loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))
        
        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())

        log_interval = 20
        '''
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {(time.time() - batch_time)/imgs.shape[0]:.4f}s ',
                  f'train_loss:{loss_meter.val:.4f}')
        '''

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs


def valid_trainer(model, valid_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    image_name_list=[]
    with torch.no_grad():
        for step, (imgs, gt_label, imgname, label_v, label_n) in enumerate(valid_loader):
            for elem in imgname:
                image_name_list.append(elem)
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            label_v = label_v[0].cuda()
            valid_logits = model(imgs, label_v)

            valid_loss = criterion(valid_logits, gt_label)
            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    return valid_loss, gt_label, preds_probs,image_name_list



