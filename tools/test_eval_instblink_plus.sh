#！/bin/bash

CUDA_VISIBLE_DEVICES=0 python tools/test_instblink_plus_eye_q_only_nms.py configs/instblink_plus/abla_full_model.py pretrained_models/instblink_plus.pth --json "/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/annotations/test.json" --root "/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/test_rawframes/"

python tools/blink_result_convertor.py

python tools/eval_mpeblink.py

