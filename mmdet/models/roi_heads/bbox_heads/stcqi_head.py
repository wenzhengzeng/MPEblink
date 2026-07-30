import torch
from mmcv.runner import auto_fp16

from mmdet.models.builder import HEADS

from .dii_head import DIIHead


@HEADS.register_module()
class STCQIHead(DIIHead):
    """Spatio-temporal head for paired instance and eye queries."""

    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine proposals and split the paired query branches."""
        all_bboxes = super().refine_bboxes(rois, labels, bbox_preds,
                                           pos_is_gts, img_metas)
        if not self.with_eye_query:
            return all_bboxes

        num_proposals = bbox_preds.shape[0] // len(img_metas) // 2
        instance_bboxes = [bboxes[:num_proposals] for bboxes in all_bboxes]
        eye_bboxes = [bboxes[num_proposals:] for bboxes in all_bboxes]
        return instance_bboxes, eye_bboxes, all_bboxes

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, clip_length, stage=0):
        """Update paired queries with spatial, clue and temporal attention."""
        num_frames, total_proposals, feature_dim = proposal_feat.shape
        assert total_proposals % 2 == 0
        assert num_frames % clip_length == 0
        num_proposals = total_proposals // 2

        instance_feat = proposal_feat[:, :num_proposals]
        eye_feat = proposal_feat[:, num_proposals:]

        instance_feat = instance_feat.permute(1, 0, 2)
        instance_feat = self.attention_norm(self.attention(instance_feat))
        instance_feat = instance_feat.permute(1, 0, 2)
        proposal_feat = torch.cat([instance_feat, eye_feat], dim=1)

        proposal_feat = proposal_feat.reshape(num_frames, 2, num_proposals,
                                              feature_dim)
        proposal_feat = proposal_feat.permute(1, 0, 2, 3)
        proposal_feat = proposal_feat.reshape(2, num_frames * num_proposals,
                                              feature_dim)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        proposal_feat = proposal_feat.reshape(2, num_frames, num_proposals,
                                              feature_dim)
        proposal_feat = proposal_feat.permute(1, 0, 2, 3)
        proposal_feat = proposal_feat.reshape(num_frames, total_proposals,
                                              feature_dim)

        batch_size = num_frames // clip_length
        proposal_feat = proposal_feat.reshape(batch_size, clip_length,
                                              total_proposals, feature_dim)
        proposal_feat = proposal_feat.permute(1, 0, 2, 3)
        proposal_feat = proposal_feat.reshape(clip_length,
                                              batch_size * total_proposals,
                                              feature_dim)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        proposal_feat = proposal_feat.reshape(clip_length, batch_size,
                                              total_proposals, feature_dim)
        proposal_feat = proposal_feat.permute(1, 0, 2, 3)
        attn_feats = proposal_feat.reshape(num_frames, total_proposals,
                                           feature_dim)

        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        interactive_feat = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + \
            self.instance_interactive_conv_dropout(interactive_feat)
        object_feat = self.instance_interactive_conv_norm(proposal_feat)
        object_feat = self.ffn_norm(self.ffn(object_feat))
        object_feat = object_feat.reshape(num_frames, total_proposals,
                                          feature_dim)

        instance_feat = object_feat[:, :num_proposals].reshape(
            num_frames * num_proposals, feature_dim)
        eye_feat = object_feat[:, num_proposals:].reshape(
            num_frames * num_proposals, feature_dim)

        cls_feat = instance_feat
        reg_feat = instance_feat
        for layer in self.cls_fcs:
            cls_feat = layer(cls_feat)
        for layer in self.reg_fcs:
            reg_feat = layer(reg_feat)

        cls_channels = (
            self.num_classes
            if self.loss_cls.use_sigmoid else self.num_classes + 1)
        cls_score = self.fc_cls(cls_feat).reshape(num_frames, num_proposals,
                                                  cls_channels)
        bbox_delta = self.fc_reg(reg_feat).reshape(num_frames, num_proposals,
                                                   4)

        eye_cls_feat = eye_feat
        eye_reg_feat = eye_feat
        for layer in self.eye_cls_fcs:
            eye_cls_feat = layer(eye_cls_feat)
        for layer in self.eye_reg_fcs:
            eye_reg_feat = layer(eye_reg_feat)

        eye_cls_score = self.eye_fc_cls(eye_cls_feat).reshape(
            num_frames, num_proposals, cls_channels)
        eye_bbox_delta = self.eye_fc_reg(eye_reg_feat).reshape(
            num_frames, num_proposals, 4)

        cls_score = torch.cat([cls_score, eye_cls_score], dim=1)
        bbox_delta = torch.cat([bbox_delta, eye_bbox_delta], dim=1)
        return cls_score, bbox_delta, object_feat, attn_feats
