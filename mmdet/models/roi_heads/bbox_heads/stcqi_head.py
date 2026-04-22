from mmdet.models.builder import HEADS
from .dii_head import DIIHead
from mmcv.runner import auto_fp16
import torch

@HEADS.register_module()
class STCQIHead(DIIHead):
    def __init__(self, *args, **kwargs):
        super(STCQIHead, self).__init__(*args, **kwargs)
    
    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, clip_length, stage = 0):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals, d = proposal_feat.shape
        num_proposals = num_proposals//2


        proposal_feat_inst = proposal_feat[:, :num_proposals, :]
        proposal_feat_eye = proposal_feat[:, num_proposals:, :]

        proposal_feat_inst = proposal_feat_inst.permute(1, 0, 2)
        proposal_feat_inst = self.attention_norm(self.attention(proposal_feat_inst))
        proposal_feat_inst = proposal_feat_inst.permute(1, 0, 2)

        proposal_feat = torch.cat([proposal_feat_inst, proposal_feat_eye], dim = 1)




        proposal_feat = proposal_feat.reshape(N, 2, num_proposals, d) 
        proposal_feat = proposal_feat.permute(1,0,2,3) # [b*t, 2, num_q,256] --> [2, b*t, num_q, 256]
        proposal_feat = proposal_feat.reshape(2, N*num_proposals, d)    # [2, b*t*num_q, 256]
        proposal_feat = self.attention_norm(self.attention(proposal_feat))

        proposal_feat = proposal_feat.reshape(2, N, num_proposals, d)    # [2, b*t, num_q, 256]
        proposal_feat = proposal_feat.permute(1,0,2,3)      # [b*t, 2, num_q, 256]
        proposal_feat = proposal_feat.reshape(N, 2*num_proposals, d)  # [b*t, 2*num_q, 256]



        proposal_feat = proposal_feat.resize(N // clip_length, clip_length,
                                             2*num_proposals,
                                             d).permute(1, 0, 2, 3) # [b*t,num_proposals,256] --> [t,b,num_proposals,256]
        proposal_feat = proposal_feat.resize(clip_length,
                                             N * 2*num_proposals // clip_length,
                                             d)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        proposal_feat = proposal_feat.resize(clip_length, N // clip_length,
                                             2*num_proposals,
                                             d).permute(1, 0, 2, 3) # [t,b*num_proposals,256] --> [b,t,num_proposals,256]
        proposal_feat = proposal_feat.resize(N, 2*num_proposals, d)
        
        attn_feats = proposal_feat

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels) # [b*t,num_proposals,256] --> [b*t*num_proposals,256]
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        obj_feat = obj_feat.reshape(N, 2*num_proposals, d)

        obj_feat_inst = obj_feat[:, :num_proposals, :].reshape(N*num_proposals, d)
        obj_feat_eye = obj_feat[:, num_proposals:, :].reshape(N*num_proposals, d)

        cls_feat = obj_feat_inst # [b*t*num_proposal, 256]
        reg_feat = obj_feat_inst
        
        eye_cls_feat = obj_feat_eye # [b*t*num_proposal, 256]
        eye_reg_feat = obj_feat_eye


        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:  #  3* fc+layer_norm+relu
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(
            N, num_proposals, self.num_classes
            if self.loss_cls.use_sigmoid else self.num_classes + 1) # [b*t,num_proposals,num_class]
        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, 4) # [b*t,num_proposals,4]


        for eye_cls_layer in self.eye_cls_fcs:
            eye_cls_feat = eye_cls_layer(eye_cls_feat)
        for eye_reg_layer in self.eye_reg_fcs:  #  3* fc+layer_norm+relu
            eye_reg_feat = eye_reg_layer(eye_reg_feat)

        eye_cls_score = self.eye_fc_cls(eye_cls_feat).view(
            N, num_proposals, self.num_classes
            if self.loss_cls.use_sigmoid else self.num_classes + 1) # [b*t,num_proposals,num_class]
        eye_bbox_delta = self.eye_fc_reg(eye_reg_feat).view(N, num_proposals, 4) # [b*t,num_proposals,4]


        cls_score = torch.cat([cls_score, eye_cls_score], dim = 1)
        bbox_delta = torch.cat([bbox_delta, eye_bbox_delta], dim = 1)


        return cls_score, bbox_delta, obj_feat, attn_feats
