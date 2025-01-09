#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# mimic_cxr
dataset="iu_xray"
annotation="annotation.json"
base_dir="iu_xray/images"
version="test"

load_model="save/iu_xray/train/checkpoint.pth"  
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
  --beam_size 3 \
  --freeze_vm False \
  --vis_use_lora False \
  --savedmodel_path ${savepath} \
  --max_length 60 \
  --min_new_tokens 40 \
  --max_new_tokens 100 \
  --repetition_penalty 2.0 \
  --length_penalty 2.0 \
  --num_workers 4 \
  --devices 1 \
  2>&1 |tee -a ${savepath}/log.txt

