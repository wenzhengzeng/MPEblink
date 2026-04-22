# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)


@HEADS.register_module()
class BlinkFusionHead(BaseModule):
    r"""Modified from Dynamic Mask Head for
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:

        in_channels (int): Input feature channels.
            Defaults to 256.
        loss_blink (dict): The config for blink loss.
    """

    def __init__(self,
                 in_channels=256,
                 loss_blink=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=5.0),
                 **kwargs):
        init_cfg = None
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(BlinkFusionHead, self).__init__(init_cfg)

        self.fp16_enabled = False
        self.in_channels = in_channels
        self.loss_blink = build_loss(loss_blink)


        self.inst_blink_fcs = nn.ModuleList()
        for _ in range(0,2):
            self.inst_blink_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.inst_blink_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.inst_blink_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))
        self.inst_fc_blink = nn.Linear(in_channels, 1)


        self.eye_blink_fcs = nn.ModuleList()
        for _ in range(0,2):
            self.eye_blink_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            self.eye_blink_fcs.append(build_norm_layer(dict(type='LN'), in_channels)[1])
            self.eye_blink_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))
        self.eye_fc_blink = nn.Linear(in_channels, 1)






    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.fc_blink.bias, bias_init)

    @auto_fp16()
    def forward(self, inst_feat, eye_feat):
        """Forward function of BlinkHead.

        Args:
            
            proposal_feat (Tensor): query feature
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            blink_score (Tensor): Predicted blink score with shape
                (batch_size*num_proposals, 1).
        """


        for blink_layer in self.inst_blink_fcs:
            blink_feat = blink_layer(inst_feat)     
        inst_blink_score = self.inst_fc_blink(inst_feat)

        for blink_layer in self.eye_blink_fcs:
            blink_feat = blink_layer(eye_feat)     
        eye_blink_score = self.eye_fc_blink(blink_feat)


        return inst_blink_score, eye_blink_score

    @force_fp32(apply_to=('blink_pred', ))
    def loss(self, blink_pred, blink_targets, reduction_override=None):
        num_pos = torch.tensor(blink_pred.size()[0],dtype=float).to(blink_pred.device)
        avg_factor = reduce_mean(num_pos)
        loss = dict()
        
        blink_targets = 1 - blink_targets
        loss_blink = self.loss_blink(
            blink_pred,
            blink_targets,
            avg_factor=avg_factor,
            reduction_override=reduction_override)
        loss['loss_blink'] = loss_blink
        return loss

    def get_targets(self, sampling_results, gt_blinks, rcnn_train_cfg):

        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]

        blink_targets = torch.cat([gt_blink[pos_assigned_gt_ind] for (gt_blink, pos_assigned_gt_ind) in zip(gt_blinks, pos_assigned_gt_inds)])
        return blink_targets
