#!/bin/bash
  
source activate R2Gen
CUDA_VISIBLE_DEVICES=0 \
python main_train.py \
--image_dir /wangx/soft_link/wangx_new_soft_link/iu_xray/images/ \
--ann_path /wangx/soft_link/wangx_new_soft_link/iu_xray/annotation.json \
--img_size 384 \
--dataset_name iu_xray \
--max_seq_length 30 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir ./results/run_iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--num_workers 3 \
--mae_pretrained /wangx/R2Gen_mae_github/checkpoint-83.pth