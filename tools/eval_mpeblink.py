from argparse import ArgumentParser

from mmdet.datasets.mpeblink_api import MPEblink
from mmdet.datasets.mpeblink_eval_api import MPEblinkEval


def parse_args():
    parser = ArgumentParser(description='Evaluate MPEblink predictions.')
    parser.add_argument(
        '--json',
        default=('/data/data4/zengwenzheng/data/dataset_building/'
                 'mpeblink_cvpr2023/annotations/test.json'),
        help='Path to the MPEblink annotation JSON file.')
    parser.add_argument(
        '--results',
        default='results/results_blink_converted.json',
        help='Path to the converted prediction JSON file.')
    return parser.parse_args()


def main(args):
    mpeblink = MPEblink(args.json)
    detections = mpeblink.loadRes(args.results)
    evaluator = MPEblinkEval(mpeblink, detections, 'bbox')
    evaluator.params.vidIds = mpeblink.getVidIds()
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.action_ap()
    evaluator.summarize()


if __name__ == '__main__':
    main(parse_args())
