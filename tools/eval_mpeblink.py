from argparse import ArgumentParser
from mmcv import DictAction
from mmdet.datasets.mpeblink_api import MPEblink
from mmdet.datasets.mpeblink_eval_api import MPEblinkEval

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--json',
        default="/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/annotations/test.json", 
        help='Path to annotation json file')
    parser.add_argument(
        '--root', default="/data/data4/zengwenzheng/data/dataset_building/mpeblink_cvpr2023/test_rawframes/", help='Path to image file') 
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main(args):
    mpeblink = MPEblink(args.json)
    mpeblink_dets = mpeblink.loadRes('results/results_blink_converted.json')
    vid_ids = mpeblink.getVidIds()
    for res_type in ['bbox']:
        iou_type = res_type
        mpeblink_eval = MPEblinkEval(mpeblink, mpeblink_dets, iou_type)
        mpeblink_eval.params.vidIds = vid_ids
        mpeblink_eval.evaluate()
        mpeblink_eval.accumulate()
        mpeblink_eval.action_ap()
        mpeblink_eval.summarize()

if __name__ == '__main__':
    args = parse_args()
    main(args)