import os
import cv2
from tqdm import tqdm
import json

def visual_pred(data_root):

    visualization_save_path = os.path.join(data_root,'visual_result')
    video_info_list = json.load(open(os.path.join(data_root, 'intermediate_results','info.json'), 'r'))
    mpeblink_det = json.load(open(os.path.join(data_root, 'intermediate_results','results_blink_converted.json'), 'r'))
    os.makedirs(visualization_save_path,exist_ok=True)
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    color = [(176, 196, 222), (255, 0, 255), (30, 144, 255), (250, 128, 114), (238, 232, 170),
             (255, 20, 147), (123, 104, 238), (255, 192, 203), (105, 105, 105), (85, 107, 47),
             (205, 133, 63), (0, 0, 128), (50, 205, 50), (127, 0, 127), (176, 48, 96),
             (128, 0, 0), (72, 61, 139), (0, 128, 0), (60, 179, 113), (0, 139, 139),
             (255, 0, 0), (255, 140, 0), (255, 215, 0), (0, 255, 0), (148, 0, 211),
             (0, 250, 154), (220, 20, 60), (0, 255, 255), (0, 191, 255), (0, 0, 255),
             (173, 255, 47), (218, 112, 214)]
    color_blink = (0, 255, 255)
    
    for video_info in tqdm(video_info_list['videos']):

        cur_video_id = video_info['id']
        focused_det = [det for det in mpeblink_det if det['video_id']==cur_video_id]
        
        f = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(visualization_save_path,f'demo_{cur_video_id}.mp4'), f, 24, (video_info['width'],video_info['height']))
        frame_index = -1

        blink_count_list = [0]*len(focused_det)    # to store the number of blinks per instance
        for img_path in video_info['file_names']:    
            img = cv2.imread(os.path.join(data_root,'intermediate_results', img_path))
            person_index = 0
            frame_index+=1

            for person in focused_det:
                bbox = person['bboxes'].pop(0)  

                if bbox == None:
                    person_index += 1
                    continue
                draw_color = color[person_index]
                for blink_event in person['blinks_converted']:
                    if frame_index>=blink_event[0] and frame_index<=blink_event[1]:
                        draw_color = color_blink
                        if frame_index == blink_event[1]:
                            blink_count_list[person_index]+=1
                        break

                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),draw_color, 5)
                cv2.putText(img,f'P{person_index} blink{blink_count_list[person_index]}',(int(bbox[0]),max(0,int(bbox[1])-10)),font,1.5,color[person_index],2)
                person_index+=1

            videoWriter.write(img)
        videoWriter.release()

if __name__ == '__main__':

    data_root = "demo_video"
    visual_pred(data_root)