#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/instblink_plus/abla_full_model.py}
CHECKPOINT=${CHECKPOINT:-pretrained_models/instblink_plus.pth}
DEVICE=${DEVICE:-cuda:0}
RAW_RESULT=${RAW_RESULT:-results/test_results/instblink_plus_raw.json}
CONVERTED_RESULT=${CONVERTED_RESULT:-results/test_results/instblink_plus.json}

: "${JSON:?Set JSON to the MPEblink2 test annotation file}"
: "${ROOT:?Set ROOT to the MPEblink2 test frame directory}"

python tools/test_instblink_plus_eye_q_only_nms.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --json "$JSON" \
    --root "$ROOT" \
    --device "$DEVICE" \
    --output "$RAW_RESULT" \
    "$@"

python tools/blink_result_convertor.py \
    "$RAW_RESULT" \
    "$CONVERTED_RESULT"

python tools/eval_mpeblink.py \
    --json "$JSON" \
    --results "$CONVERTED_RESULT"
