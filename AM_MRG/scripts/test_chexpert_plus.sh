#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# use mimic_cxr
dataset="mimic_cxr"
annotation="annotation.json"
base_dir="cheXpert_plus/PNG/"
version="test"

load_model="save/mimic_cxr/train/checkpoint.pth"  

savepath="save/$dataset/$version"
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
python -u train.py \
  --test \
  --dataset ${dataset} \
  --annotation ${annotation} \
  --base_dir ${base_dir} \
  --delta_file ${load_model} \
  --stageone_path 'configs/Stage1ckpt.pth' \
  --class_activation_maps_path 'configs/CAM.pkl' \
  --llm_path 'Meta/Llama-2-7b-chat-hf' \
  --vision_model configs/MambaXrayCLIP-L.pth \
  --test_batch_size 2 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --max_length 100 \
  --min_new_tokens 80 \
  --max_new_tokens 120 \
  --repetition_penalty 2.0 \
  --length_penalty 2.0 \
  --num_workers 6 \
  --devices 1 \
  2>&1 |tee -a ${savepath}/log.txt

