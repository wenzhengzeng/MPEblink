import os
import cv2
import json
from tqdm import tqdm

source_root = 'demo_video'
source_video_root = os.path.join(source_root, 'source_video')
source_video_name_list = os.listdir('demo_video/source_video')
inter_path = os.path.join(source_root, 'intermediate_results')
os.makedirs(inter_path, exist_ok=True)


info = {}
info['videos'] = []

for source_video_name in tqdm(source_video_name_list):

    source_video_path = os.path.join(source_video_root, source_video_name)

    capture = cv2.VideoCapture(source_video_path)
    img_dir = os.path.join(inter_path, source_video_name.split('.')[0])

    os.makedirs(img_dir, exist_ok=True)

    file_names = []
    img_index = 0
    cur_height = 0 
    cur_width = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        save_img_name = f'{str(img_index).rjust(5,"0")}.png'
        cv2.imwrite(os.path.join(img_dir, save_img_name), frame)
        file_names.append(os.path.join(source_video_name.split('.')[0], save_img_name))
        if img_index == 0 :
            cur_height = frame.shape[0]
            cur_width = frame.shape[1]
        img_index +=1
        
    cur_video = {'file_names': file_names, 'id': source_video_name.split('.')[0], 'height': cur_height, 'width': cur_width}
    info['videos'].append(cur_video)
json_path = os.path.join(inter_path, 'info.json')
json.dump(info, open(json_path, 'w'))
