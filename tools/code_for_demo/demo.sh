#！/bin/bash

python tools/code_for_demo/video2img_info.py
python tools/code_for_demo/video_inference.py
python tools/code_for_demo/blink_convertor.py
python tools/code_for_demo/result_visual.py

rm -rf demo_video/intermediate_results