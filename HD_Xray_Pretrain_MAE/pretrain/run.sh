#!/bin/bash
#SBATCH --job-name=Mae_Vim
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=7
#SBATCH -p GPUFEE07
#SBATCH -o log/%j.log 
#SBATCH -e log/%j.err
echo $(hostname) $CUDA_VISIBLE_DEVICES
srun python -u -m torch.distributed.launch --nproc_per_node=8 main.py --arch='mae_vit_large_patch16' --lr 0.000025 --load_from /gpfs/home/22054/maeclip/checkpoint4/checkpoint-20.pth