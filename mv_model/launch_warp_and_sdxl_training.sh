#!/usr/bin/env bash
set -euo pipefail

export TRAIN_ENTRYPOINT="mv_model/train_controlnet_sdxl_warp.py"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/try_time}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"

exec "$(dirname "$0")/../mv_model_common/launch_training.sh" "$@"
