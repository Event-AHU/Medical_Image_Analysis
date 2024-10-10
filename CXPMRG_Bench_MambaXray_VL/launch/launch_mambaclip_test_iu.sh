#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

dataset="iu_xray"
annotation="configs/iu_annotation.json"
base_dir="iu_xray/images/"
version="iu_test"

# load_model="save/iu_xray/iu/checkpoints/checkpoint.pth"  
load_model="configs/IU-Finetune-L.pth"
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
  --vision_model configs/MambaXrayCLIP-L.pth \
  --test_batch_size 16 \
  --beam_size 5 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --max_length 60 \
  --min_new_tokens 40 \
  --max_new_tokens 100 \
  --repetition_penalty 2.0 \
  --length_penalty 2.0 \
  --num_workers 6 \
  --devices 1 \
  2>&1 |tee -a ${savepath}/log.txt

