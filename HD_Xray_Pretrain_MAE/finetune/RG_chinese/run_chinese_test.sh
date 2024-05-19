#!/bin/bash
  
source activate R2Gen
CUDA_VISIBLE_DEVICES=0 \
python main.py \
--ann_path ./data/chinese/chinese_annotation_20w.json \
--resume ./results/mae_large_224_90_bs16_roberta_9223_ss50_20w/model_best.pth \
--test True \
--dataset_name chinese \
--img_size 224 \
--max_seq_length 100 \
--n_gpu 1 \
--batch_size 100 \
--epochs 100 \
--save_dir ./results/mae_large_224_90_bs16_roberta_9223_ss50_20w \
--step_size 50 \
--gamma 0.8 \
--seed 9223 \
--mae_pretrained /wangx/R2Gen_mae_github/checkpoint-83.pth