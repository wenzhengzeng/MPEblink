#！/bin/bash

CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=2 --master_port=29502 tools/train.py configs/instblink/instblink_r50.py --seed 0 --launcher pytorch --no-validate --cfg-options load_from=pretrained_models/tevit_r50.pth

CUDA_VISIBLE_DEVICES=2 python tools/test_mpeblink.py configs/instblink/instblink_r50.py work_dirs/instblink_r50/iter_10000.pth --json "/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/annotations/test.json" --root "/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/test_rawframes/"

python tools/blink_result_convertor.py

python tools/eval_mpeblink.py
