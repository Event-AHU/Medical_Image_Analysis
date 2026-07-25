#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

dataset="cheXpert_plus"
annotation="/wangx/home/E24301191/mycode/MAC_RRG/annotation_with_draft_CXP.json"
base_dir="/wangx/_dataset/cheXpert_plus/PNG/"

delta_file="xxxxxx"

version="v1_deep"
savepath="/wangx_nas/QYH/cheXpert_plus/1e-4/v1_deep/test"


python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 6 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 2 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt