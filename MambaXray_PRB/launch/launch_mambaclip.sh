#!/bin/bash
export CUDA_VISIBLE_DEVICES=7\

# mimic_cxr
dataset="mimic_cxr"
annotation="configs/clip_annotation.json"
base_dir="/"

# mimic_cxr
version="MambaCLIP"
savepath="save/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train_clip.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 48  \
    --vision_model configs/Pretrain-L.pth \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --num_workers 6 \
    --devices 1 \
    --max_epochs 51 \
    --strategy deepspeed \
    2>&1 |tee -a ${savepath}/log.txt

    