#!/usr/bin/env bash
set -euo pipefail

export TRAIN_ENTRYPOINT="mv_model_no_hsi/train_controlnet_sdxl_warp.py"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/mv_model_no_hsi_0_with_drop}"
export CUDA_DEVICES="${CUDA_DEVICES:-1}"
export ACCELERATE_GPU_IDS="${ACCELERATE_GPU_IDS:-1}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-2500}"

exec "$(dirname "$0")/../mv_model_common/launch_training.sh" "$@"
