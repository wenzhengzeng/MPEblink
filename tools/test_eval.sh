#！/bin/bash

CUDA_VISIBLE_DEVICES=0 python tools/test_mpeblink.py configs/instblink/instblink_r50.py pretrained_models/instblink_r50.pth --json "/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/annotations/test.json" --root "/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/test_rawframes/"

python tools/blink_result_convertor.py

python tools/eval_mpeblink.py

