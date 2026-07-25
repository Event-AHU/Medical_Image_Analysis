#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

dataset="iu_xray"
annotation="/wangx/home/E24301191/mycode/MAC_RRG/data/iu_xray/annotation.json"
base_dir="/wangx/home/E24301191/mycode/MAC_RRG/data/iu_xray/images"
delta_file="xxxxx"

version="v1_deep"
savepath="/wangx_nas/QYH/IU_Xray/test/1e-4/v1_deep/test"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 6 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt