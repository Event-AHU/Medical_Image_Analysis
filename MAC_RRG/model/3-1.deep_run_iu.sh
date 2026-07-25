#!/bin/bash
export CUDA_VISIBLE_DEVICES=4\

dataset="iu_xray"
annotation="/wangx/home/E24301191/mycode/MAC_RRG/data/iu_xray/annotation.json"
base_dir="./data/iu_xray/images"

version="v1_deep"
savepath="/wangx_nas/QYH/IU_Xray/1e-4/$version"




python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 8 \
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
    --max_epochs 30 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt
