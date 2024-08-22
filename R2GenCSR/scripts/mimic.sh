#!/bin/bash

dataset="mimic_cxr"
annotation="path/to/mimic_cxr/annotation.json"
base_dir="path/to/mimic_cxr/images"
if [ $# -eq 0 ]; then
    echo "If you need to test the model, please provide the absolute path of the model to be tested after training with this script."
fi
load_model=$1
if [[ $load_model != "" ]]; then
    echo "++++++++test++++++++"
    savepath="${load_model}_test"
    echo "The running files will be saved to $savepath"
    test_mode="--test \
              --beam_size 3\
              --delta_file ${load_model} \
              --test_batch_size 20 \
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
CUDA_VISIBLE_DEVICES=0 python train.py \
    $test_mode \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --context_pair 3 \
    --chosen vmamba \
    --llm llama2 \
    --vision_model ./VMamba/vssm_base_0229_ckpt_epoch_237.pth \
    --llama_model meta-llama/Llama-2-7b-chat-hf\
    --batch_size 36  \
    --val_batch_size 36 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
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
    --strategy deepspeed \
    2>&1 |tee -a ${savepath}/log.txt

