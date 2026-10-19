#!/usr/bin/env bash
set -euo pipefail

export TRAIN_ENTRYPOINT="mv_model_no_deformable/train_controlnet_sdxl_warp.py"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/mv_model_0_with_drop_no_deformable}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"
export ACCELERATE_GPU_IDS="${ACCELERATE_GPU_IDS:-1}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"

exec "$(dirname "$0")/../mv_model_common/launch_training.sh" "$@"
