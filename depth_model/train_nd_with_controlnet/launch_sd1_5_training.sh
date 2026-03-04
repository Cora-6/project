#! /usr/bin/env bash

export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="/data/nd_model_7_lora"
export WANDB_API_KEY=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


export CUDA_VISIBLE_DEVICES=0
accelerate launch train_with_cnet.py \
 --mixed_precision="fp16" \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="/data/train" \
 --resolution=64 \
 --report_to="wandb" \
 --learning_rate=1e-5 \
 --train_batch_size=8 \
 --num_train_epochs=48 \
 --gradient_accumulation_steps=8 \
 --gradient_checkpointing \
 --dataloader_num_workers 4 #\
#  --resume_from_checkpoint='/data/nd_model_4_wrap_cnet/checkpoint-9000'
#  --validation_image "/data/result_sphere/stitch/2025-03-07_11-20-44/render_h_transformed_hsi.mat" "/home/viplab/diffusers/examples/controlnet/mytry/dataset/result/2025-02-10_11-01-49"
#  --validation_prompt "city view"\
#  --max_train_steps=6000 \
