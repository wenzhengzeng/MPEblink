#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/instblink_plus/abla_full_model.py}
GPUS=${GPUS:-2}
PRETRAINED=${PRETRAINED:-pretrained_models/tevit_r50.pth}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

export CUDA_VISIBLE_DEVICES
PORT=${PORT:-29502} bash tools/dist_train.sh \
    "$CONFIG" \
    "$GPUS" \
    --no-validate \
    --cfg-options load_from="$PRETRAINED" \
    "$@"
