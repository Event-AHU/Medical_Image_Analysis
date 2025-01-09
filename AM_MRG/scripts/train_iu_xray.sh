#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# iu_xray
dataset="iu_xray"
annotation="annotation.json"
base_dir="iu_xray/images"

# mimic_cxr
version="train"
savepath="save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
# max_batch_size 32-->70G
python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --stageone_path 'configs/Stage1ckpt.pth' \
    --class_activation_maps_path 'configs/CAM.pkl' \
    --llm_path 'Meta/Llama-2-7b-chat-hf' \
    --batch_size 2  \
    --val_batch_size 2 \
    --vision_model configs/MambaXrayCLIP-L.pth \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --beam_size 3 \
    --num_workers 4 \
    --devices 1 \
    --max_epochs 30 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --num_sanity_val_steps 2 \
    --strategy deepspeed \
    2>&1 |tee -a ${savepath}/log.txt

    