#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# batch_size 128 arm_large_pz16 26G     400
torchrun --standalone --nproc_per_node=1 --master_port 4399 pretrain/main_pretrain.py \
    --batch_size 12 \
    --input_size 192 \
    --model arm_large_pz16 \
    --norm_pix_loss \
    --epochs 101 \
    --warmup_epochs 5 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path all_data_mimic.pkl \
    --output_dir pretrain/outputs/test/ \
    --log_dir pretrain/outputs/test/
