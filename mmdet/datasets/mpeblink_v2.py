import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from .builder import DATASETS
from .custom import CustomDataset
from .mpeblink_api import MPEblink
from .pipelines import Compose


@DATASETS.register_module()
class MPEblinkV2Dataset(CustomDataset):
    """MPEblink2 video dataset with face, eye-region and blink labels."""

    CLASSES = ('person_face', )
    EYE_REGION_LANDMARKS = (17, 21, 22, 26, 36, 40, 41, 45, 46, 47, 29)

    def __init__(self,
                 ann_file,
                 pipeline,
                 clip_length,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 with_eye_bbox=False,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.ann_file = ann_file
        self.clip_length = clip_length
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.with_eye_bbox = with_eye_bbox
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        self.data_infos_found = set(self.data_infos)
        self.pipeline = Compose(pipeline)

    def load_annotations(self, ann_file):
        self.mpeblink = MPEblink(ann_file)
        self.cat_ids = self.mpeblink.getCatIds()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        vid_ids = self.mpeblink.getVidIds()

        vid_infos = []
        for i in vid_ids:
            info = self.mpeblink.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos.append(info)
        self.vid_infos = vid_infos

        img_ids = []
        vid2frame = defaultdict(list)
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                img_ids.append((idx, frame_id))
                vid2frame[idx].append(frame_id)
        self.vid2frame = vid2frame
        return img_ids

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i, (vid, frame_id) in enumerate(self.data_infos):
            video_info = self.vid_infos[vid]
            if video_info['width'] / video_info['height'] > 1:
                self.flag[i] = 1

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = []

        if self.filter_empty_gt:
            for i, (vid, frame_id) in enumerate(self.data_infos):
                vid_id = self.vid_infos[vid]['id']
                ann_ids = self.mpeblink.getAnnIds(vidIds=[vid_id])
                ann_info = self.mpeblink.loadAnns(ann_ids)
                anns = [
                    ann['bboxes'][frame_id] for ann in ann_info
                    if ann['bboxes'][frame_id] is not None
                ]
                if anns:
                    ids_with_ann.append(1)
                else:
                    ids_with_ann.append(0)
        for i, (vid, frame_id) in enumerate(self.data_infos):
            if self.filter_empty_gt and not ids_with_ann[i]:
                continue
            if min(self.vid_infos[vid]['width'],
                   self.vid_infos[vid]['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_img_info(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_info = self.vid_infos[vid]
        img_info = dict(
            file_name=vid_info['file_names'][frame_id],
            filename=vid_info['filenames'][frame_id],
            width=vid_info['width'],
            height=vid_info['height'],
            frame_id=frame_id)
        return img_info

    def get_ann_info(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_id = self.vid_infos[vid]['id']
        ann_ids = self.mpeblink.getAnnIds(vidIds=[vid_id])
        ann_info = self.mpeblink.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def get_cat_ids(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_id = self.vid_infos[vid]['id']
        ann_ids = self.mpeblink.getAnnIds(vidIds=[vid_id])
        ann_info = self.mpeblink.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    @staticmethod
    def get_min_max_position(coordinates, indices):
        x_values = [coordinates[i][0] for i in indices]
        y_values = [coordinates[i][1] for i in indices]
        return min(x_values), max(x_values), min(y_values), max(y_values)

    def _parse_ann_info(self, ann_info, frame_id):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        gt_blinks = []
        gt_eye_bboxes = []

        for ann in ann_info:
            bbox = ann['bboxes'][frame_id]
            if bbox is None:
                continue
            x1, y1, w, h = bbox
            bbox = [x1, y1, x1 + w, y1 + h]

            if self.with_eye_bbox:
                min_x, max_x, min_y, max_y = self.get_min_max_position(
                    ann['landmark'][frame_id], self.EYE_REGION_LANDMARKS)
                eye_bbox = [min_x, min_y, max_x + 1, max_y + 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'] - 1)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_blinks.append(ann['blinks_binary'][frame_id])
                if self.with_eye_bbox:
                    gt_eye_bboxes.append(eye_bbox)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_blinks = np.array(gt_blinks, dtype=np.int64)
            gt_ids = np.array(gt_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_blinks = np.array([], dtype=np.int64)
            gt_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        parsed = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            blinks=gt_blinks,
            bboxes_ignore=gt_bboxes_ignore,
            ids=gt_ids)
        if self.with_eye_bbox:
            parsed['eye_bboxes'] = np.array(
                gt_eye_bboxes, dtype=np.float32).reshape(-1, 4)
        return parsed

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        if self.test_mode:
            raise NotImplementedError
        while True:
            data = self.prepare_train_clip(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_clip(self, idx):
        vid, frame_id = self.data_infos[idx]
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_idxs = []
        for i in sample_range:
            valid_idx = (vid, i)
            if valid_idx in self.data_infos_found:
                valid_idxs.append(valid_idx)
        assert len(valid_idxs) > 0
        frame_interval = 2
        index_pre = [
            (vid, frame_id - frame_interval * i)
            for i in range(1, self.clip_length // 2 + 1)
            if (frame_id - frame_interval * i) >= valid_idxs[0][1] and (
                vid, frame_id - frame_interval * i) in valid_idxs
        ]
        pre_res = [(vid, valid_idxs[0][1])
                   for i in range(0, self.clip_length // 2 - len(index_pre))]
        index_pre = index_pre + pre_res
        index_post = [
            (vid, frame_id + frame_interval * i)
            for i in range(1, self.clip_length // 2 + 1)
            if (frame_id + frame_interval * i) <= valid_idxs[-1][1] and (
                vid, frame_id + frame_interval * i) in valid_idxs
        ]
        post_res = [(vid, valid_idxs[-1][1])
                    for i in range(0, self.clip_length // 2 - len(index_post))]
        index_post += post_res
        index_except_center = index_pre + index_post
        valid_idxs = [idx] + [
            self.data_infos.index(_) for _ in index_except_center
        ]
        valid_idxs.sort()

        clip = []

        for _ in valid_idxs:
            clip.append(self.prepare_train_img(_))

            if _ == valid_idxs[0]:
                for tsfm in self.pipeline.transforms:
                    if hasattr(tsfm, 'isfix'):
                        tsfm.isfix = True
            elif _ == valid_idxs[-1]:
                for tsfm in self.pipeline.transforms:
                    if hasattr(tsfm, 'isfix'):
                        tsfm.isfix = False

        data = {}
        for key in clip[0]:
            stacked = []
            stacked += [_[key].data for _ in clip]
            if isinstance(stacked[0], torch.Tensor) and clip[0][key].stack:
                stacked = torch.stack(stacked, dim=0)
            data[key] = DC(
                stacked,
                clip[0][key].stack,
                clip[0][key].padding_value,
                cpu_only=clip[0][key].cpu_only)
        return data

    def prepare_test_clip(self, idx):
        raise NotImplementedError

    def prepare_train_img(self, idx):
        img_info = self.get_img_info(idx)
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def evaluate(self,
                 results,
                 metric,
                 results_file='results.json',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False):
        self.result2json(results, results_file)

    def result2json(self, results, results_file):
        json_results = []
        vid_objs = {}
        for idx in range(len(self)):
            # assume results is ordered
            vid, frame_id = self.data_infos[idx]
            if idx == len(self) - 1:
                is_last = True
            else:
                _, frame_id_next = self.data_infos[idx + 1]
                is_last = frame_id_next == 0
            det, seg, id = results[idx]
            labels = []
            for i in range(len(det)):
                labels += [i for _ in range(len(det[i]))]
            det = np.vstack(det)
            segm = []
            for i in seg:
                segm += i
            ids = []
            for i in id:
                ids += i
            seg = segm
            id = ids

            for obj_index in range(len(id)):
                bbox = det[obj_index]
                segm = seg[obj_index]
                label = labels[obj_index]
                obj_id = id[obj_index]
                if obj_id not in vid_objs:
                    vid_objs[obj_id] = {'scores': [], 'cats': [], 'segms': {}}
                vid_objs[obj_id]['scores'].append(bbox[4])
                vid_objs[obj_id]['cats'].append(label)
                segm['counts'] = segm['counts'].decode()
                vid_objs[obj_id]['segms'][frame_id] = segm
            if is_last:
                # store results of  the current video
                for obj_id, obj in vid_objs.items():
                    data = dict()

                    data['video_id'] = vid + 1
                    data['score'] = np.array(obj['scores']).mean().item()
                    # majority voting for sequence category
                    data['category_id'] = np.bincount(np.array(
                        obj['cats'])).argmax().item() + 1
                    vid_seg = []
                    for fid in range(frame_id + 1):
                        if fid in obj['segms']:
                            vid_seg.append(obj['segms'][fid])
                        else:
                            vid_seg.append(None)
                    data['segmentations'] = vid_seg
                    json_results.append(data)
                vid_objs = {}
        mmcv.dump(json_results, results_file)
