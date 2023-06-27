#！/bin/bash

CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=2 --master_port=29502 tools/train.py configs/instblink/instblink_r50.py --seed 0 --launcher pytorch --no-validate --cfg-options load_from=pretrained_models/tevit_r50.pth