import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel

from modules.datasets import MimiccxrSingleImageDataset
from torchvision import transforms
from modules.tester import Tester
import torch.distributed as dist

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def parse_agrs():
    parser = argparse.ArgumentParser()
 
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='../data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--img_size', type=int, default=224, help='')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr', 'chinese'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    # parser.add_argument('--mae_pretrained', type=str, default='pretrain_models/VL_16_224_checkpoint-90.pth',    # vit_base_p16_224.pth  VL_16_224_checkpoint-27.pth
    #                     help='the visual extractor to be used.')
    parser.add_argument('--mae_pretrained', type=str, default='/gpfs/home/22054/maeclip/checkpoint4/checkpoint-146.pth',    # vit_base_p16_224.pth  VL_16_224_checkpoint-27.pth
                        help='the visual extractor to be used.') 
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=1024, help='the dimension of the patch features.')    # base  768    /    large  1024
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=101, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=102, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    # parser.add_argument('--resume', type=str, default="/gpfs/home/22054/R2Gen-mae-224/results/mae_large_224_90_bs16_roberta_9223_ss50_20w_aug/model_best.pth",
    # help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--resume', type=str, default=None,
    help='whether to resume the training from existing checkpoints.')
    
    parser.add_argument('--distributed', type=bool, default=False, help='Whether distributed training is enabled')
    parser.add_argument('--load', type=str, help='whether to load a pre-trained model.')
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device =torch.device('cpu')
    if(args.distributed):
        
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)        
        device = torch.device(f'cuda:{local_rank}')
        # 初始化进程组
        dist.init_process_group(backend='nccl', init_method='env://')

    # fix random seeds
    print(args)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)


    # create tokenizer
    tokenizer = Tokenizer(args)

    train_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    # create data loader
    # train_dataset = MimiccxrSingleImageDataset(args, tokenizer, split='train', transform=train_transform)
    # train_sampler = DistributedSampler(train_dataset)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
    # next(iter(train_dataloader))
    # 1/0
    
    # val_dataset = MimiccxrSingleImageDataset(args, tokenizer, split='val', transform=val_transform)
    # val_sampler = DistributedSampler(val_dataset)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    
    # test_dataset = MimiccxrSingleImageDataset(args, tokenizer, split='test', transform=val_transform)
    # test_sampler = DistributedSampler(test_dataset)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    # exper_dataloader = R2DataLoader(args, tokenizer, split='experiment', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)
    model = model.to(device)
    
    if(args.distributed):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    # trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    # trainer.train()
    tester = Tester(model, criterion, metrics, args, test_dataloader)
    tester.plot()


if __name__ == '__main__':
    main()
