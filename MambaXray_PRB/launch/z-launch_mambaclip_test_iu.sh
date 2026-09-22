#!/bin/bash
source ./java.sh
export CUDA_VISIBLE_DEVICES=3

dataset="iu_xray"
annotation="./imagesize192/maxpatchs12/CoT_iu_patchs.json"
base_dir="./iu-xray/images/"
version="test_iu_base"

load_model="./save/iu_xray/iu_base/checkpoints/checkpoint.pth"
savepath="save/$dataset/$version"
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

nohup python -u train_downstream.py \
  --test \
  --dataset ${dataset} \
  --annotation ${annotation} \
  --base_dir ${base_dir} \
  --delta_file ${load_model} \
  --vision_model configs/MambaXrayCLIP-B.pth \
  --test_batch_size 20 \
  --beam_size 5 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --thinking_min_new_tokens 0 \
  --thinking_max_new_tokens 120 \
  --max_length 60 \
  --min_new_tokens 40 \
  --max_new_tokens 100 \
  --repetition_penalty 2.0 \
  --length_penalty 2.0 \
  --num_workers 6 \
  --devices 1 \
  > ${savepath}/log.txt 2>&1 &
    
echo "PID: $!"
echo "tail -f ${savepath}/log.txt"

