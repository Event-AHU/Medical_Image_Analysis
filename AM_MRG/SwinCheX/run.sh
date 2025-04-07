#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 1 --master_port 1235 main.py \
    --local_rank 0\
    --cfg configs/swin_large_patch4_window7_224.yaml \
    --trainset / \
    --validset / \
    --testset / \
    --qformer True \
    --train_csv_path configs/mimic_chexpert_train.csv \
    --valid_csv_path configs/mimic_chexpert_val.csv \
    --test_csv_path configs/mimic_chexpert_test.csv \
    --batch-size 2 \
    --output output \
    --tag train_qformer \
    --num_mlp_heads 3 