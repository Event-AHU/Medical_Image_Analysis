#!/bin/bash
  
source activate R2Gen
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--dataset_pkl_path ./dataset/RSNA_Pneumonia/rsna-pneumonia_1_data.pkl \
--batchsize 200  \
--epoch 100 \
--height 224 \
--width 224 \
--mode_save_path rsna-pneumonia_model_1p \
--pretrain_path /wangx/checkpoint-83.pth