export MODEL_DIR="hotshotco/SDXL-512" #"stabilityai/stable-diffusion-xl-base-1.0"
# export OUTPUT_DIR="/data/mv_model_0_with_drop"
export OUTPUT_DIR="/data/try_time"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before running this script}"

export CUDA_VISIBLE_DEVICES=0
accelerate launch train_controlnet_sdxl_warp.py\
 --checkpointing_steps=500 \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="/data/train" \
 --resolution=512 \
 --report_to="wandb" \
 --learning_rate=1e-5 \
 --num_train_epochs=48 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=16 \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --proportion_empty_prompts=0.1 \
 --resume_from_checkpoint='latest'
#  --validation_image "/data/result_sphere/stitch/2025-03-07_11-20-44/render_h_transformed_hsi.mat" "/home/viplab/diffusers/examples/controlnet/mytry/dataset/result/2025-02-10_11-01-49"
#  --validation_prompt "city view"\
#  --max_train_steps=6000 \
