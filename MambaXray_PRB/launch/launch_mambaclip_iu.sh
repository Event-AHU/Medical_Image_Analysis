#!/bin/bash
source ./java.sh
export CUDA_VISIBLE_DEVICES=6 \

dataset="iu_xray"
annotation="./imagesize192/maxpatchs12/CoT_iu_patchs.json"
base_dir="./iu-xray/images/"

version="iu_xray_base"
savepath="save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

nohup python -u train_downstream.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 20 \
    --val_batch_size 20 \
    --vision_model configs/MambaXrayCLIP-B.pth \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --learning_rate 0.00009 \
    --thinking_length 200 \
    --thinking_min_new_tokens 0 \
    --thinking_max_new_tokens 120 \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 6 \
    --devices 1 \
    --max_epochs 30 \
    --limit_val_batches 1.0 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 1 \
    --strategy deepspeed \
    > ${savepath}/log.txt 2>&1 &
    
echo "PID: $!"
echo "tail -f ${savepath}/log.txt"
