import json
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from mmcv import ConfigDict, DictAction
from mmcv.parallel import collate, scatter
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.core import build_assigner
from mmdet.datasets.pipelines import Compose


def parse_args():
    parser = ArgumentParser(description='Run InstBlink++ video inference.')
    parser.add_argument('config', help='Config file.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument(
        '--json', required=True, help='MPEblink test annotation JSON file.')
    parser.add_argument(
        '--root', required=True, help='Root directory containing test frames.')
    parser.add_argument(
        '--output',
        help='Output JSON file. A generated path is used by default.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    parser.add_argument(
        '--no-nms',
        action='store_false',
        dest='nms',
        help='Disable temporal NMS.')
    parser.add_argument(
        '--clip-len',
        type=int,
        help=
        'Frames per forward pass. Defaults to config.my_infer_cfg.clip_len.')
    parser.add_argument(
        '--stride',
        type=int,
        help='Clip stride. Defaults to half of clip length.')
    parser.add_argument(
        '--match-threshold',
        type=float,
        default=0.2,
        help='Minimum similarity used to link tracks between clips.')
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.5,
        help='Similarity threshold used by temporal NMS.')
    parser.add_argument(
        '--person-threshold',
        type=float,
        help='Detection threshold. Defaults to the value in the config.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings using key=value pairs.')
    return parser.parse_args()


def build_matcher():
    matcher_config = ConfigDict(
        dict(
            type='FaceLinkerCalculator',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='InferenceBBoxL1Cost', weight=5.0),
            iou_cost=dict(type='IoUCost', iou_mode='iou', weight=1.0)))
    return build_assigner(matcher_config)


def build_clip_ranges(video_length, clip_length, stride):
    if video_length <= clip_length:
        return [(0, video_length)]

    last_start = video_length - clip_length
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + clip_length) for start in starts]


def prepare_clip(file_names, root, pipeline, device):

    def prepare_frame(file_name):
        data = dict(img_info=dict(filename=file_name), img_prefix=root)
        return pipeline(data)

    workers = min(8, len(file_names))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        samples = list(executor.map(prepare_frame, file_names))

    batch = collate(samples, samples_per_gpu=len(samples))
    batch['img_metas'] = batch['img_metas'].data
    batch['img'] = batch['img'].data
    return scatter(batch, [device])[0]


def stack_clip_results(det_bboxes, eye_bboxes, blink_scores):
    det_bboxes = torch.stack(det_bboxes).permute(1, 0, 2)
    eye_bboxes = torch.stack(eye_bboxes).permute(1, 0, 2)
    blink_scores = torch.stack(blink_scores).permute(1, 0, 2)
    return det_bboxes, eye_bboxes, blink_scores


def filter_queries(det_bboxes, eye_bboxes, blink_scores, threshold):
    keep = det_bboxes[:, 0, -1] > threshold
    return det_bboxes[keep], eye_bboxes[keep], blink_scores[keep]


def temporal_nms(assigner, det_bboxes, eye_bboxes, blink_scores, img_meta,
                 threshold):
    if det_bboxes.shape[0] == 0:
        return det_bboxes, eye_bboxes, blink_scores

    order = det_bboxes[:, 0, -1].sort(descending=True)[1]
    det_bboxes = det_bboxes[order]
    eye_bboxes = eye_bboxes[order]
    blink_scores = blink_scores[order]

    similarities = assigner.assign(
        det_bboxes.permute(1, 0, 2), det_bboxes.permute(1, 0, 2), img_meta)
    suppressed = torch.zeros(
        det_bboxes.shape[0], dtype=torch.bool, device=similarities.device)
    keep = []
    for index in range(det_bboxes.shape[0]):
        if suppressed[index]:
            continue
        keep.append(index)
        suppressed |= similarities[index] > threshold
        suppressed[index] = False
    return det_bboxes[keep], eye_bboxes[keep], blink_scores[keep]


def append_new_tracks(video_bboxes, video_eye_bboxes, video_blinks, new_bboxes,
                      new_eye_bboxes, new_blinks, new_indices):
    if not new_indices:
        return video_bboxes, video_eye_bboxes, video_blinks

    new_indices = torch.as_tensor(
        new_indices, device=new_bboxes.device, dtype=torch.long)
    prefix_length = video_bboxes.shape[1] - new_bboxes.shape[1]
    num_new = len(new_indices)
    bbox_prefix = video_bboxes.new_zeros((num_new, prefix_length, 5))
    blink_prefix = video_blinks.new_zeros((num_new, prefix_length, 1))

    appended_bboxes = torch.cat([bbox_prefix, new_bboxes[new_indices]], dim=1)
    appended_eye_bboxes = torch.cat(
        [bbox_prefix.clone(), new_eye_bboxes[new_indices]], dim=1)
    appended_blinks = torch.cat([blink_prefix, new_blinks[new_indices]], dim=1)

    video_bboxes = torch.cat([video_bboxes, appended_bboxes], dim=0)
    video_eye_bboxes = torch.cat([video_eye_bboxes, appended_eye_bboxes],
                                 dim=0)
    video_blinks = torch.cat([video_blinks, appended_blinks], dim=0)
    return video_bboxes, video_eye_bboxes, video_blinks


def merge_clip_tracks(assigner, video_bboxes, video_eye_bboxes, video_blinks,
                      new_bboxes, new_eye_bboxes, new_blinks, overlap,
                      img_meta, match_threshold):
    extension = new_bboxes.shape[1] - overlap
    previous_overlap = video_bboxes[:, -overlap:] if overlap else \
        video_bboxes[:, :0]

    bbox_padding = video_bboxes.new_zeros(
        (video_bboxes.shape[0], extension, 5))
    blink_padding = video_blinks.new_zeros(
        (video_blinks.shape[0], extension, 1))
    video_bboxes = torch.cat([video_bboxes, bbox_padding], dim=1)
    video_eye_bboxes = torch.cat(
        [video_eye_bboxes, bbox_padding.clone()], dim=1)
    video_blinks = torch.cat([video_blinks, blink_padding], dim=1)

    if new_bboxes.shape[0] == 0:
        return video_bboxes, video_eye_bboxes, video_blinks
    if previous_overlap.shape[0] == 0:
        return append_new_tracks(video_bboxes, video_eye_bboxes, video_blinks,
                                 new_bboxes, new_eye_bboxes, new_blinks,
                                 list(range(new_bboxes.shape[0])))
    if overlap == 0:
        return append_new_tracks(video_bboxes, video_eye_bboxes, video_blinks,
                                 new_bboxes, new_eye_bboxes, new_blinks,
                                 list(range(new_bboxes.shape[0])))

    similarities = assigner.assign(
        previous_overlap.permute(1, 0, 2),
        new_bboxes[:, :overlap].permute(1, 0, 2), img_meta)
    matched_new = set()
    for _ in range(min(similarities.shape)):
        flat_index = int(similarities.argmax().item())
        old_index = flat_index // similarities.shape[1]
        new_index = flat_index % similarities.shape[1]
        score = float(similarities[old_index, new_index].item())
        similarities[old_index, :] = -10000
        similarities[:, new_index] = -10000
        if score < match_threshold:
            continue

        matched_new.add(new_index)
        if overlap:
            overlap_slice = slice(-(extension + overlap), -extension)
            video_bboxes[
                old_index,
                overlap_slice] = (video_bboxes[old_index, overlap_slice] +
                                  new_bboxes[new_index, :overlap]) / 2
            video_eye_bboxes[
                old_index,
                overlap_slice] = (video_eye_bboxes[old_index, overlap_slice] +
                                  new_eye_bboxes[new_index, :overlap]) / 2
            video_blinks[
                old_index,
                overlap_slice] = (video_blinks[old_index, overlap_slice] +
                                  new_blinks[new_index, :overlap]) / 2
        video_bboxes[old_index, -extension:] = \
            new_bboxes[new_index, overlap:]
        video_eye_bboxes[old_index, -extension:] = \
            new_eye_bboxes[new_index, overlap:]
        video_blinks[old_index, -extension:] = \
            new_blinks[new_index, overlap:]

    unmatched = [
        index for index in range(new_bboxes.shape[0])
        if index not in matched_new
    ]
    return append_new_tracks(video_bboxes, video_eye_bboxes, video_blinks,
                             new_bboxes, new_eye_bboxes, new_blinks, unmatched)


def xyxy_to_xywh(box):
    if sum(box) == 0:
        return None
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def serialize_tracks(video_id, det_bboxes, eye_bboxes, blink_scores):
    results = []
    det_bboxes = det_bboxes.permute(1, 0, 2)
    eye_bboxes = eye_bboxes.permute(1, 0, 2)
    blink_scores = blink_scores.permute(1, 0, 2)

    for instance_index in range(det_bboxes.shape[1]):
        frame_scores = det_bboxes[:, instance_index, -1]
        positive_scores = frame_scores[frame_scores > 0]
        score = positive_scores.mean().item() if positive_scores.numel() else 0
        result = dict(
            video_id=video_id,
            score=score,
            category_id=1,
            bboxes=[],
            eye_bboxes=[],
            blink_scores=[],
            score_per_img=[])

        for frame_index in range(det_bboxes.shape[0]):
            face_box = det_bboxes[frame_index,
                                  instance_index, :-1].detach().cpu().tolist()
            eye_box = eye_bboxes[frame_index,
                                 instance_index, :-1].detach().cpu().tolist()
            result['bboxes'].append(xyxy_to_xywh(face_box))
            result['eye_bboxes'].append(xyxy_to_xywh(eye_box))
            result['blink_scores'].append(blink_scores[frame_index,
                                                       instance_index].item())
            result['score_per_img'].append(frame_scores[frame_index].item())
        results.append(result)
    return results


def main(args):
    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=args.cfg_options)
    config = model.cfg
    pipeline = Compose(config.data.test.pipeline)
    matcher = build_matcher()

    clip_length = args.clip_len or config.my_infer_cfg.clip_len
    stride = args.stride or max(1, clip_length // 2)
    person_threshold = (
        args.person_threshold if args.person_threshold is not None else
        config.my_infer_cfg.person_threshold)

    with open(args.json, encoding='utf-8') as annotation_file:
        annotations = json.load(annotation_file)

    results = []
    for video in tqdm(annotations['videos']):
        file_names = video['file_names']
        clip_ranges = build_clip_ranges(len(file_names), clip_length, stride)
        video_bboxes = None
        video_eye_bboxes = None
        video_blinks = None
        previous_end = 0

        for start, end in clip_ranges:
            batch = prepare_clip(file_names[start:end], args.root, pipeline,
                                 args.device)
            with torch.no_grad():
                (det_bboxes, _), eye_bboxes, blink_scores = model(
                    return_loss=False, rescale=True, format=False, **batch)

            det_bboxes, eye_bboxes, blink_scores = stack_clip_results(
                det_bboxes, eye_bboxes, blink_scores)
            det_bboxes, eye_bboxes, blink_scores = filter_queries(
                det_bboxes, eye_bboxes, blink_scores, person_threshold)
            if args.nms:
                det_bboxes, eye_bboxes, blink_scores = temporal_nms(
                    matcher, det_bboxes, eye_bboxes, blink_scores,
                    batch['img_metas'][0][0], args.nms_threshold)

            if video_bboxes is None:
                video_bboxes = det_bboxes
                video_eye_bboxes = eye_bboxes
                video_blinks = blink_scores
            else:
                overlap = max(0, previous_end - start)
                video_bboxes, video_eye_bboxes, video_blinks = \
                    merge_clip_tracks(
                        matcher,
                        video_bboxes,
                        video_eye_bboxes,
                        video_blinks,
                        det_bboxes,
                        eye_bboxes,
                        blink_scores,
                        overlap,
                        batch['img_metas'][0][0],
                        args.match_threshold)
            previous_end = end

        if video_bboxes is not None and video_bboxes.shape[0] > 0:
            results.extend(
                serialize_tracks(video['id'], video_bboxes, video_eye_bboxes,
                                 video_blinks))

    if args.output:
        output_path = Path(args.output)
    else:
        suffix = '_NMS' if args.nms else ''
        output_path = Path('results/test_results') / (
            f'{Path(args.config).stem}{suffix}_{Path(args.json).name}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(results, output_file)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main(parse_args())
