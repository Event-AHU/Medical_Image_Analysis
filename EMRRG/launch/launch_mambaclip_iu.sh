#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 \

# mimic_cxr
dataset="iu_xray"
annotation="/iu_xray/annotation.json"
base_dir="/iu_xray/images"

# mimic_cxr
version="iu"
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
    --batch_size 4  \
    --val_batch_size 6 \
    --vision_model configs/MambaXrayCLIP-B.pth \
    --freeze_vm True \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --beam_size 5 \
    --num_workers 6 \
    --devices 1 \
    --max_epochs 21 \
    --limit_val_batches 1.0 \
    --val_check_interval 2.0 \
    --num_sanity_val_steps 2 \
    --strategy deepspeed \
    --lora_X \
    --dim_X 16 \
    --s_X 1.0 \
    2>&1 |tee -a ${savepath}/log.txt

    