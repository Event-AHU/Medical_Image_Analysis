#!/bin/bash
export CUDA_VISIBLE_DEVICES=5\

# mimic_cxr
dataset="mimic_cxr"
annotation="configs/chexpert_annotation.json"
base_dir="/cheXpert_plus/PNG/"

version="chexpert"
savepath="save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
python -u train_downstream.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 6  \
    --val_batch_size 6 \
    --vision_model configs/MambaXrayCLIP-B.pth \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 6 \
    --devices 2 \
    --max_epochs 11 \
    --limit_val_batches 1 \
    --val_check_interval 1 \
    --num_sanity_val_steps 1 \
    --strategy deepspeed \
    --lora_X \
    --dim_X 16 \
    --s_X 1.0 \
    2>&1 |tee -a ${savepath}/log.txt

    