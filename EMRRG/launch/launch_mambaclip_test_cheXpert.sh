#!/bin/bash
export CUDA_VISIBLE_DEVICES=5\

dataset="mimic_cxr"
annotation="configs/chexpert_annotation.json"
base_dir="cheXpert_plus/PNG/"
version="chexpert_test"


load_model="configs/EMRRG-CXP.pth"
savepath="save/$dataset/$version"
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train_downstream.py \
  --test \
  --dataset ${dataset} \
  --annotation ${annotation} \
  --base_dir ${base_dir} \
  --delta_file ${load_model} \
  --vision_model configs/MambaXrayCLIP-B.pth \
  --test_batch_size 8 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --max_length 100 \
  --min_new_tokens 80 \
  --max_new_tokens 120 \
  --repetition_penalty 2.0 \
  --length_penalty 2.0 \
  --num_workers 6 \
  --devices 1 \
  2>&1 |tee -a ${savepath}/log.txt


