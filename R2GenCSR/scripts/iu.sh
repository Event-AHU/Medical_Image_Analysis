#!/bin/bash

dataset="iu_xray"
annotation="path/to/iu_xray/annotation.json"
base_dir="path/to/iu_xray/images"
if [ $# -eq 0 ]; then
    echo "If you need to test the model, please provide the absolute path of the model to be tested after training with this script."
fi
load_model=$1
if [[ $load_model != "" ]]; then
    echo "++++++++test++++++++"
    savepath="${load_model}_test"
    echo "The running files will be saved to $savepath"
    test_mode="--test \
              --beam_size 5\
              --delta_file ${load_model} \
              --test_batch_size 32 \
              "
else
    echo "++++++++train++++++++"
    script_name=$(basename "$0" .sh)
    savepath="./save/$dataset/$script_name"
    echo "The running files will be saved to $savepath"
    test_mode=""
fi
mkdir -p "$savepath"
script_name=$(basename "$0")
script_dir=$(dirname "$0")
source_script="${script_dir}/${script_name}"
cp "$source_script" "${savepath}"
CUDA_VISIBLE_DEVICES=1 python train.py \
     $test_mode \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --context_pair_seed 4096 \
    --beam_size 5 \
    --context_pair 3 \
    --chosen vmamba \
    --llm qwen \
    --vision_model ./VMamba/vssm_base_0229_ckpt_epoch_237.pth \
    --llama_model Qwen/Qwen1.5-1.8B-Chat \
    --batch_size 32  \
    --val_batch_size 32 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 60 \
    --min_new_tokens 40 \
    --max_new_tokens 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 6 \
    --devices 1 \
    --max_epochs 25 \
    --limit_val_batches 1 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 0 \
    --input-size 224 \
    --strategy deepspeed \
    2>&1 |tee -a ${savepath}/log.txt

