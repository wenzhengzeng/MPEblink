# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .queryinst import QueryInst


@DETECTORS.register_module()
class InstBlinkPlus(QueryInst):
    """InstBlink++ detector with paired instance and eye queries."""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def extract_feat(self, batch_size, clip_length, img):
        if hasattr(self.backbone, 'msg_tokens'):
            features = self.backbone(batch_size, clip_length, img)
        else:
            features = self.backbone(img)
        if self.with_neck:
            features = self.neck(features)
        return features

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_blinks,
                      gt_eye_bboxes,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_ids=None,
                      proposals=None,
                      **kwargs):
        assert proposals is None, (
            'InstBlink++ does not support external proposals')

        batch_size, clip_length = img.shape[:2]
        img = img.reshape(batch_size * clip_length, *img.shape[2:])
        img_metas = [frame for clip in img_metas for frame in clip]
        gt_bboxes = [frame for clip in gt_bboxes for frame in clip]
        gt_labels = [frame for clip in gt_labels for frame in clip]
        gt_blinks = [frame for clip in gt_blinks for frame in clip]
        gt_eye_bboxes = [frame for clip in gt_eye_bboxes for frame in clip]
        gt_ids = [frame for clip in gt_ids for frame in clip]

        features = self.extract_feat(batch_size, clip_length, img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(features, img_metas)
        return self.roi_head.forward_train(
            batch_size,
            clip_length,
            features,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_blinks,
            gt_eye_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            gt_ids=gt_ids,
            imgs_whwh=imgs_whwh)

    def simple_test(self, img, img_metas, rescale=False, format=False):
        batch_size, clip_length = 1, img.size(0)
        features = self.extract_feat(batch_size, clip_length, img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(features, img_metas)
        return self.roi_head.simple_test(
            features,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale,
            format=format)
