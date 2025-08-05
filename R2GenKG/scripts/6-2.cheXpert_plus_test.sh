#!/bin/bash

dataset="cheXpert_plus"
annotation="/wangx/_dataset/cheXpert_plus/annotation.json"
base_dir="/wangx/_dataset/cheXpert_plus/PNG/"
delta_file="save/$dataset/$version/checkpoint.pth"

version="v1_deep"
savepath="save/$dataset/$version"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 8 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt
