#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

dataset="iu_xray"
annotation="iu_xray/annotation.json"
base_dir="iu_xray/images"
version="iu_test"

load_model="save/iu_xray/iu/checkpoints/checkpoint.pth"  
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
  --test_batch_size 4 \
  --beam_size 5 \
  --freeze_vm True \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --max_length 60 \
  --min_new_tokens 40 \
  --max_new_tokens 100 \
  --repetition_penalty 2.0 \
  --length_penalty 2.0 \
  --num_workers 6 \
  --devices 1 \
  --lora_X \
  --dim_X 16 \
  --s_X 1.0 \
  2>&1 |tee -a ${savepath}/log.txt

