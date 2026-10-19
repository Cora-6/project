#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before running this script}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-hotshotco/SDXL-512}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR before running this script}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:?Set TRAIN_ENTRYPOINT before running this script}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

accelerate_args=()
if [[ -n "${ACCELERATE_GPU_IDS:-}" ]]; then
  accelerate_args+=(--gpu_ids "$ACCELERATE_GPU_IDS")
fi

accelerate launch "${accelerate_args[@]}" "$PROJECT_ROOT/$TRAIN_ENTRYPOINT" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  --pretrained_model_name_or_path="$MODEL_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --dataset_name="${DATASET_DIR:-/data/train}" \
  --resolution=512 \
  --report_to=wandb \
  --learning_rate=1e-5 \
  --num_train_epochs=48 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=16 \
  --gradient_checkpointing \
  --dataloader_num_workers=16 \
  --proportion_empty_prompts=0.1 \
  --resume_from_checkpoint=latest \
  "$@"
