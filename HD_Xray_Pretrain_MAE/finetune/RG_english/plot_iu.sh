#!/bin/bash
  
source activate R2Gen
CUDA_VISIBLE_DEVICES=0 \
python main_plot.py \
--image_dir /Share/home/22054/pbc_dataset_medical/iu_xray/images/ \
--ann_path /Share/home/22054/pbc_dataset_medical/iu_xray/annotation.json \
--img_size 384 \
--dataset_name iu_xray \
--max_seq_length 30 \
--threshold 3 \
--batch_size 1 \
--epochs 30 \
--save_dir results/plot \
--step_size 100 \
--gamma 0.1 \
--seed 9223 \
--beam_size 1 \
--load ./results/run_iu_xray/model_best.pth \
--mae_pretrained /wangx/R2Gen_mae_github/checkpoint-83.pth
