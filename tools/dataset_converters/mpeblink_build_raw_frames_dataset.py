import os
import time
import cv2
import json
from tqdm import tqdm
from argparse import ArgumentParser
# generate raw frames and gt json files from video dataset 
parser = ArgumentParser()

parser.add_argument('--root', default="/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/", help='Path to dataset root')
args = parser.parse_args()

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

video_dataset_root = args.root

split_dirs = ['validate', 'train','test']
for split_dir in split_dirs:
    
    split_dataset_root = os.path.join(video_dataset_root, split_dir)
    if not os.path.exists(split_dataset_root):
        continue
    rawframes_dataset_root = os.path.join(video_dataset_root, f'{split_dir}_rawframes')
    video_list = os.listdir(split_dataset_root)
    video_list = list(map(int, video_list))
    video_list.sort()
    dataset = {}

    info = {'info': {'description': 'MPEblink Dataset', 'url': '1', 'version': '1', 'year': '2022', 'contributor': 'Wenzheng Zeng, Sicheng Wei, Jinfang Gan, Xintao Zhang', 'data_created': time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}   }
    licenses = {'licenses': 'only for research'}
    categories = {'categories': [{'supercategory': 'object', 'id': 1, 'name': 'person_face'}] }

    videos = []
    annotations = []
    anno_id = 1
    
    target_width = 640
    target_height = 360
    for video_sample in tqdm(video_list):

        video_path = os.path.join(split_dataset_root, str(video_sample), 'video.mp4')
        anno_path = os.path.join(split_dataset_root, str(video_sample), 'annote.json')
        origin_anno = json.load(open(anno_path, 'r'))

        height = origin_anno.pop('height')
        width = origin_anno.pop('width')
        length = origin_anno.pop('length')
        scale_w = target_width / width
        scale_h = target_height / height

        # record video-related information and store the individual frames
        video = {'height': target_height, 'width': target_width, 'length': length}
        file_names = []
        camera = cv2.VideoCapture(video_path)
        save_dir = os.path.join(rawframes_dataset_root, str(video_sample))
        os.makedirs(save_dir, exist_ok=True)
        img_index = 0
        while True:
            res, image = camera.read()
            if not res: 
                break
            relative_img_path = str(video_sample)+'/'+ str(img_index).rjust(5,'0') + '.png'
            image = cv2.resize(image, (target_width, target_height))
            cv2.imwrite(os.path.join(rawframes_dataset_root, relative_img_path), image)
            file_names.append(relative_img_path)
            img_index += 1
        camera.release()
        video.update({'file_names': file_names})
        video.update({'id': video_sample})
        videos.append(video)

        # record annotation-related information
        for person in origin_anno:
            anno = {'height': target_height, 'width': target_width, 'length': 1, 'category_id': 1}
            anno_blink = []
            # resize the gt annotation according to the shape of resized frames (640*360 in our experiment)
            for index in range(0,length):
                if origin_anno[person]['bbox'][index] == None:
                    anno_blink.append(None) 
                    continue
                else:
                    origin_anno[person]['bbox'][index][0] = origin_anno[person]['bbox'][index][0] * scale_w
                    origin_anno[person]['bbox'][index][1] = origin_anno[person]['bbox'][index][1] * scale_h
                    origin_anno[person]['bbox'][index][2] = origin_anno[person]['bbox'][index][2] * scale_w
                    origin_anno[person]['bbox'][index][3] = origin_anno[person]['bbox'][index][3] * scale_h
                    for landmark_index in range(0, 68):
                        origin_anno[person]['landmark'][index][landmark_index][0] = origin_anno[person]['landmark'][index][landmark_index][0] * scale_w
                        origin_anno[person]['landmark'][index][landmark_index][1] = origin_anno[person]['landmark'][index][landmark_index][1] * scale_h
                blink_sign = 0
                for blink_index in range(0,len(origin_anno[person]['blink'])):
                    if index >= origin_anno[person]['blink'][blink_index][0] and index <=origin_anno[person]['blink'][blink_index][1]:
                        blink_sign = 1
                        break
                anno_blink.append(blink_sign)
            anno.update({'bboxes': origin_anno[person]['bbox']})
            anno.update({'landmark': origin_anno[person]['landmark']})
            anno.update({'blinks':origin_anno[person]['blink']})
            anno.update({'blinks_binary': anno_blink})
            anno.update({'video_id': video_sample})
            anno.update({'id': anno_id})

            anno_id += 1
            
            annotations.append(anno)

    dataset.update(info)
    dataset.update(licenses)
    dataset.update({'videos': videos})
    dataset.update(categories)
    dataset.update({'annotations': annotations})

    final_json_root = os.path.join(video_dataset_root, 'annotations')
    os.makedirs(final_json_root, exist_ok=True)
    json.dump(dataset, open(os.path.join(final_json_root, f'{split_dir}.json'), 'w'))
    print('Done')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





