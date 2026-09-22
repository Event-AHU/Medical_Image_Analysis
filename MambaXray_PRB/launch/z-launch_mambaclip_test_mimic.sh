#!/bin/bash
source ./java.sh
export CUDA_VISIBLE_DEVICES=0 \

dataset="mimic_cxr"
annotation="./imagesize224/maxpatchs12/CoT_mimic_patchs.json"
base_dir="./mimic_cxr_pretrain/images/"
version="test_mimic_large"

load_model="./save/mimic_cxr/mimic_large/checkpoints/checkpoint.pth"
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
  --vision_model configs/MambaXrayCLIP-L.pth \
  --test_batch_size 12 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --thinking_min_new_tokens 0 \
  --thinking_max_new_tokens 700 \
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


