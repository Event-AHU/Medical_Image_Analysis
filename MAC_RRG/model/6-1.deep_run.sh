#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

dataset="cheXpert_plus"
annotation="/wangx/home/E24301191/mycode/MAC_RRG/annotation_with_draft_CXP.json"
base_dir="/wangx/_dataset/cheXpert_plus/PNG/"

version="v1_deep"
savepath="/wangx_nas/QYH/CheXpert_plus/1e-4/v1_deep/"



if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size  8 \
    --val_batch_size 8 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 10 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    2>&1 |tee -a ${savepath}/log.txt
