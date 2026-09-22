#!/bin/bash
source ./java.sh
export CUDA_VISIBLE_DEVICES=6 \

dataset="mimic_cxr"
annotation="./imagesize192/maxpatchs12/CoT_cheXpert_plus_patchs.json"
base_dir="./cheXpert_plus/PNG/"

version="chexpert_base"
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
    --batch_size 6 \
    --accumulate_grad_batches 3 \
    --val_batch_size 12 \
    --vision_model configs/MambaXrayCLIP-B.pth \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --learning_rate 0.00009 \
    --thinking_length 200 \
    --thinking_min_new_tokens 0 \
    --thinking_max_new_tokens 200 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 6 \
    --devices 1 \
    --max_epochs 10 \
    --limit_val_batches 1.0 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 1 \
    --strategy deepspeed \
    > ${savepath}/log.txt 2>&1 &
    
echo "PID: $!"
echo "tail -f ${savepath}/log.txt"
