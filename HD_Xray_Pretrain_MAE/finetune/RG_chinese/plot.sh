#!/bin/bash
  
source activate R2Gen
CUDA_VISIBLE_DEVICES=0 \
python plot.py \
--ann_path /Share/home/22054/R2Gen-mae-224/data/chinese/chinese_annotation_20w.json \
--dataset_name chinese \
--img_size 224 \
--max_seq_length 100 \
--n_gpu 1 \
--batch_size 1 \
--epochs 100 \
--save_dir results/mae_large_224_90_bs16_roberta_9223_ss50_20w \
--step_size 1 \
--gamma 0.8 \
--seed 9223 \
--lr_ve 5e-5 \
--mae_pretrained /wangx/R2Gen-mae-224/checkpoint-146.pth \
--load ./results/mae_large_224_90_bs16_roberta_9223_ss50_20w/model_best_60.pth \
--beam_size 1