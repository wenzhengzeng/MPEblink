import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList
from .sparse_roi_head import SparseRoIHead
from ..builder import HEADS, build_head, build_roi_extractor
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh

@HEADS.register_module()
class InstBlinkRoIHead(SparseRoIHead):

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=None,
                 mask_roi_extractor=None,
                 bbox_head=None,
                 mask_head=None,
                 blink_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(InstBlinkRoIHead, self).__init__(num_stages,
            stage_loss_weights,
            proposal_feature_channel,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        if blink_head is not None:
            self.init_blink_head(mask_roi_extractor, blink_head)

    @property
    def with_blink(self):
        """bool: whether the RoI head contains a `blink_head`"""
        return hasattr(self, 'blink_head') and self.blink_head is not None
    
    def init_blink_head(self, mask_roi_extractor, blink_head):
        """Initialize blink head.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            blink_head (dict): Config of blink head.
        """
        self.blink_head = nn.ModuleList()
        if not isinstance(blink_head, list):
            blink_head = [blink_head for _ in range(self.num_stages)]
        assert len(blink_head) == self.num_stages
        for head in blink_head:
            self.blink_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor


    def _bbox_forward(self, stage, x, rois, object_feats, img_metas, clip_length):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois) # [b*t*num_proposal, 256, w_feat, h_feat] roi-align
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats, clip_length)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            rois.new_zeros(len(rois)),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            decode_bbox_pred=torch.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list]

        ) 

        return bbox_results
    
    def _blink_forward(self, stage, attn_feats):
        """Mask head forward function used in both training and testing."""
        blink_head = self.blink_head[stage]
        # do not support caffe_c4 model anymore
        blink_pred = blink_head(attn_feats)

        blink_results = dict(blink_pred=blink_pred)
        return blink_results

    def _blink_forward_train(self, stage, attn_feats, sampling_results,
                            gt_blinks, rcnn_train_cfg):
        """Run forward function and calculate loss for mask head in
        training."""
        attn_feats = torch.cat([
            feats[res.pos_inds]
            for (feats, res) in zip(attn_feats, sampling_results)
        ])
        blink_results = self._blink_forward(stage, attn_feats)

        blink_targets = self.blink_head[stage].get_targets(
            sampling_results, gt_blinks, rcnn_train_cfg)

        loss_blink = self.blink_head[stage].loss(blink_results['blink_pred'],
                                               blink_targets)
        blink_results.update(loss_blink)
        return blink_results

    def forward_train(self,
                      B,
                      T,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_blinks,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None,
                      gt_ids=None):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_blinks: eyeblink scores corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """

        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1) # [b*t,1,4] --> [b*t,num_proposals(100),4]
        all_stage_bbox_results = []
        proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(
                stage, x, rois, object_feats, img_metas, clip_length=T)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            for i in range(B):
                normolize_bbox_ccwh = []
                for j in range(T):
                    normolize_bbox_ccwh.append(
                        bbox_xyxy_to_cxcywh(proposal_list[i * T + j] /
                                            imgs_whwh[i * T]))
                assign_result = self.bbox_assigner[stage].assign(
                    normolize_bbox_ccwh,            
                    cls_pred_list[i * T:i * T + T],     
                    gt_bboxes[i * T:i * T + T],
                    gt_labels[i * T:i * T + T],
                    img_metas[i * T],
                    gt_ids=gt_ids[i * T:i * T + T])
                sampling_result = []
                for j in range(T):  # for each frame
                    sampling_result.append(self.bbox_sampler[stage].sample(
                        assign_result[j], proposal_list[i * T + j],gt_bboxes[i * T + j]
                        ))  
                sampling_results.extend(sampling_result)    
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True) # bbox_targets: (labels,label_weights,bbox_targets,bbox_weights)，for subsequent loss calculation. labels: [b*t*num_proposals], bbox: [b*t*num_proposals,4]
            cls_score = bbox_results['cls_score']   # [b*t,num_proposal,num_class]
            decode_bbox_pred = bbox_results['decode_bbox_pred'] # [b*t*num_proposal, 4]

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh)

            if self.with_blink:
                blink_results = self._blink_forward_train(
                    stage, bbox_results['object_feats'], sampling_results,
                    gt_blinks, self.train_cfg[stage])
                single_stage_loss['loss_blink'] = blink_results['loss_blink']

            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                    self.stage_loss_weights[stage]
            object_feats = bbox_results['object_feats']

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False,
                    format=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has a mask branch,
            it is a list[tuple] that contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposal_list = [proposal_boxes[i] for i in range(num_imgs)]    # [t,num_proposal,4]
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        object_feats = proposal_features
        if all([proposal.shape[0] == 0 for proposal in proposal_list]):
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for i in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs
            return bbox_results

        for stage in range(self.num_stages):
            rois = bbox2roi(proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas, clip_length=len(img_metas))
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []
        attn_feats = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]
        # During inference, only the results of the last iteration are used
        cls_score_mean = cls_score.mean(dim=0) # The classification results of each frame are averaged and used as the classification score of the whole query
        scores_per_img, topk_indices = cls_score_mean.flatten(0, 1).topk(
            self.test_cfg.max_per_img, sorted=False) # [num_qeury,num_class] --> [num_query*num_class]
        for img_id in range(num_imgs):    # for each frame                     
            labels_per_img = topk_indices % num_classes # get category
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                      num_classes]
            attn_feats_per_img = bbox_results['object_feats'][img_id][
                topk_indices // num_classes] 
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor) # bbox divides directly by scale_factor, [x1,y1,x2,y2]
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1)) # concat bbox and score
            det_labels.append(labels_per_img)
            attn_feats.append(attn_feats_per_img)

        if format:
            bbox_results = [
                bbox2result(det_bboxes[i], det_labels[i], num_classes)
                for i in range(num_imgs)
            ]
        else:
            bbox_results = (det_bboxes, det_labels)

        if self.with_blink:
           
            attn_feats = torch.cat(attn_feats, dim=0)

            blink_results = self._blink_forward(stage, attn_feats)
            blink_results['blink_pred'] = blink_results['blink_pred'].reshape(
                num_imgs, -1, *blink_results['blink_pred'].size()[1:])
            final_blink_results = []
            blink_pred = blink_results['blink_pred']
            blink_pred = blink_pred.sigmoid()
            for img_id in range(num_imgs):  # for each frame
              
                blink_pred_per_img = blink_pred[img_id]
                
                final_blink_results.append(blink_pred_per_img)


        if self.with_blink:
           
            return bbox_results, final_blink_results

        return bbox_results
