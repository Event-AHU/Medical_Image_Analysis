#!/bin/bash

dataset="mimic_cxr"
annotation="path/to/mimic_cxr/annotation.json"
base_dir="path/to/mimic_cxr/images"

CUDA_VISIBLE_DEVICES=4 python demo.py \
    --dataset ${dataset} \
    --delta_file path/to/save/checkpoint.pth \
    --beam_size 3 \
    --demo True \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --context_pair 3 \
    --chosen vmamba \
    --llm llama2 \
    --vision_model ./VMamba/vssm_base_0229_ckpt_epoch_237.pth \
    --llama_model meta-llama/Llama-2-7b-chat-hf \
    --batch_size 36  \
    --val_batch_size 36 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --devices 1 \
    --max_epochs 20 \
    --limit_val_batches 1 \
    --val_check_interval 1 \
    --num_sanity_val_steps 0 \
    --input-size 224 \
    --strategy deepspeed 
