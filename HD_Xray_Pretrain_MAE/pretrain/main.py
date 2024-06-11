# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision.transforms import InterpolationMode
from tensorboardX import SummaryWriter


from dataset.MyDatasets import *


def get_args_parser():
    parser = argparse.ArgumentParser('our', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='mae_vit_large_patch16', type=str,
        choices=['mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'])
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--input_size', default=1280, type=int,
                        help='images input size')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--use_learnable_pos_emb', default=False,
                        type=str, help='masked strategy of video tokens/patches False')


    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")    #True
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.00025, type=float, help="""Learning rate at the end of  #0.0005
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the      #1e-6
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")       #训练中断后加载之前训好的模型
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    parser.add_argument('--mask_ratio_outer', default=0.85, type=float)       #训练中断后加载之前训好的模型
    parser.add_argument('--mask_ratio_iner', default=0.95, type=float)

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Learnable masking parameters
    parser.add_argument('--softmax_temp', type=float, default=1e-2, metavar='Learnable_Mask',
                        help='Softmax temp used to compute probability values for each patch')

    # Misc

    parser.add_argument('--data_path', default='/gpfs/home/22054/medical_xray_pretrain/ALL_DATA/image', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/gpfs/home/22054medical_xray_pretrain/pretrain/out", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=7, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    #datasets
    parser.add_argument('--pkl_path', type=str, default='/gpfs/home/22054/maeclip2-caption/104debug_sample_20230615/all_dataset.pkl')
    parser.add_argument('--root', type=str, default='/gpfs/home/22054/medical_xray_pretrain/ALL_DATA')
    #loss 
    parser.add_argument('--loss_weight_sim', type=float, default=0.2)
    return parser

def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    device = torch.device(args.device)
    
    # ============ preparing data ... ============
    transform_train_ran = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),   # 3 is bicubic
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5] )]) 
    transform_train = transforms.Compose([
            transforms.Resize([args.input_size,args.input_size]),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5] )])  

    dataset_train_ran=MyDatasets(args.pkl_path,args.root,args, transform=transform_train_ran)
    dataset_train=MyDatasets(args.pkl_path,args.root,args, transform=transform_train)
    sampler_ran = torch.utils.data.DistributedSampler(dataset_train_ran, shuffle=True)
    sampler_rgb = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)

    data_loader_train_ran = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_ran,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_rgb,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Data loaded: there are {len(dataset_train)} images.")

    student = models.__dict__[args.arch](norm_pix_loss=args.norm_pix_loss)
    
    student = utils.MultiCropWrapper(student)

    student.to(device)

    student_without_ddp = student
    
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)

        # we need DDP wrapper to have synchro batch norms working...我们需要 DDP 包装器才能使同步批处理规范正常工作...
        student_without_ddp = student.module
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False,find_unused_parameters=True) 

    print(f"Student and Teacher are built: they are both {args.arch} network.")


    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)

    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1 :
            param.requires_grad=False
        else:
            pass
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        # 创建优化器对象
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader_train),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader_train),
    )
                  
    print(f"Loss, optimizer and schedulers ready.")


    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
        )
        
        checkpoint = torch.load(os.path.join(args.output_dir, args.load_from), map_location='cpu')
        student_dict = checkpoint['model']
        new_state_dict = {}
        for key, value in student_dict.items():
            new_key = 'module.' + key
            new_state_dict[new_key] = value
        student.load_state_dict(new_state_dict)
        
    start_epoch = to_restore["epoch"]
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"可用的GPU数量：{num_gpus}")

    # 获取当前进程使用的GPU索引
    current_gpu = torch.cuda.current_device()
    print(f"当前进程使用的GPU索引：{current_gpu}")

    start_time = time.time()
    print("Starting our training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        data_loader_train_ran.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(student,
            data_loader_train_ran,data_loader_train,optimizer, device,lr_schedule, wd_schedule,
            epoch, fp16_scaler,args)
        
        save_dict = {
            'student': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'all_loss':train_stats['loss'],
            'mae_loss':train_stats['mae_loss']
        }
        loss_scaler = dict( mae =train_stats['mae_loss'], loss= train_stats['loss'])
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.output_dir and (epoch % args.epochs == 0 or epoch == args.epochs -1):
            utils.save_model(
                args=args, model=student, model_without_ddp=student_without_ddp, optimizer=optimizer,
                epoch=epoch,loss_scaler=loss_scaler)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        
        
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, data_loader_ran, data_loader, optimizer, device: torch.device,lr_schedule, wd_schedule,epoch,
                    fp16_scaler, args):
  
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    
    for (it, (images_ran,_)),( _, (images,_)) in zip(enumerate(metric_logger.log_every(data_loader_ran, 20, header)),enumerate(metric_logger.log_every(data_loader, 20, header))):

        it = len(data_loader) * epoch + it  # global training iteration

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        # move images to gpu   
        images_ran = images_ran.to(device, non_blocking=True)   
        images = images.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            mae_loss,masks = student(images_ran, mask_type = 0, mask_ratio_outer = args.mask_ratio_outer, mask_ratio_iner = args.mask_ratio_iner)
            mae_loss,masks = student(images, mask_type = 1, mask_ratio_outer = args.mask_ratio_outer, mask_ratio_iner = args.mask_ratio_iner)

            #loss1= student_loss
            mae_loss = ((mae_loss * masks).sum() / masks.sum())
            

        loss= mae_loss
        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(mae_loss=mae_loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return return_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Xray', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
