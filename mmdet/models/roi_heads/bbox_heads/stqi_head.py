from mmdet.models.builder import HEADS
from .dii_head import DIIHead
from mmcv.runner import auto_fp16

@HEADS.register_module()
class STQIHead(DIIHead):
    def __init__(self, *args, **kwargs):
        super(STQIHead, self).__init__(*args, **kwargs)
    
    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, clip_length):
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
        # sptial self-attention 
        proposal_feat = proposal_feat.permute(1, 0, 2) # [b*t,num_proposals,256] --> [num_proposals,b*t,256]  regards b*t as batch and do self attention across proposals within the same frame
        proposal_feat = self.attention_norm(self.attention(proposal_feat))

        proposal_feat = proposal_feat.permute(1, 0, 2) # [num_proposals,b*t,256] --> [b*t,num_proposals,256]


        # Temporal self-atten
        proposal_feat = proposal_feat.resize(N // clip_length, clip_length,
                                             num_proposals,
                                             d).permute(1, 0, 2, 3) # [b*t,num_proposals,256] --> [t,b,num_proposals,256]
        proposal_feat = proposal_feat.resize(clip_length,
                                             N * num_proposals // clip_length,
                                             d) # [t,b,num_proposals,256] --> [t,b*num_proposals,256], regards b*num_proposal as batch and do self attention across frame (time) within one query
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        proposal_feat = proposal_feat.resize(clip_length, N // clip_length,
                                             num_proposals,
                                             d).permute(1, 0, 2, 3) # [t,b*num_proposals,256] --> [b,t,num_proposals,256]
        proposal_feat = proposal_feat.resize(N, num_proposals, d) # [b,t,num_proposals,256] --> [b*t,num_proposals,256] 
        
        attn_feats = proposal_feat

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels) # [b*t,num_proposals,256] --> [b*t*num_proposals,256]
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat) # dynamic conv, take proposal_feat as filter, apply on the roi_feat
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic) # Residual connection，similar to detr decoder
        obj_feat = self.instance_interactive_conv_norm(proposal_feat) # a layer norm

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat # [b*t*num_proposal, 256]
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:  # fc+layer_norm+relu
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:  #  3* fc+layer_norm+relu
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(
            N, num_proposals, self.num_classes
            if self.loss_cls.use_sigmoid else self.num_classes + 1) # [b*t,num_proposals,num_class]
        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, 4) # [b*t,num_proposals,4]
        return cls_score, bbox_delta, obj_feat.view(
            N, num_proposals, self.in_channels), attn_feats
