#!/bin/bash
source ./java.sh
export CUDA_VISIBLE_DEVICES=2 \

dataset="mimic_cxr"
annotation="./imagesize192/maxpatchs12/CoT_cheXpert_plus_patchs.json"
base_dir="./cheXpert_plus/PNG/"
version="test_chexpert_base"

load_model="./save/mimic_cxr/chexpert_base/checkpoints/checkpoint.pth"
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
  --test_batch_size 12 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --thinking_min_new_tokens 0 \
  --thinking_max_new_tokens 200 \
  --max_length 100 \
  --min_new_tokens 80 \
  --max_new_tokens 120 \
  --repetition_penalty 2.0 \
  --length_penalty 2.0 \
  --num_workers 6 \
  --devices 1 \
  > ${savepath}/log.txt 2>&1 &
    
echo "PID: $!"
echo "tail -f ${savepath}/log.txt"


