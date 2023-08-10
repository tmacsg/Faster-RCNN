from typing import List, Optional, Dict, Tuple
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import roi_pool
import config
from utils import data_sample
from einops import rearrange
from modules import backbone, roi_head, rpn

class FasterRCNN(nn.Module):
    def __init__(self, 
                 backbone: nn.Module=backbone.mobilenetv2_backbone(),
                 rpn_loss_weight: float=1.0):
        super().__init__()       
        self.rpn_loss_weight = rpn_loss_weight
        self.backbone = backbone
        self.out_channels = self.backbone.out_channels        
        self.num_anchors = len(config.anchor_size_ratios) * len(config.anchor_aspect_ratios)        
        self.rpn_head = rpn.RPNHead(self.out_channels, self.num_anchors)
        self.rpn = rpn.RPN(self.rpn_head)        
        self.roi_head = roi_head.ROIHead(in_channel=self.out_channels)  
        self.roi_head_loss = roi_head.ROIHeadLoss()
        self.roi_head_post_process = roi_head.ROIHeadPostprocess()     
        
    def forward(self, images, resized_image_sizes, targets=None):
        batch_size = images.shape[0]
        image_size = images.shape[-2:]       
        feature = self.backbone(images)
        rpn_proposals, rpn_loss = self.rpn(feature, image_size, targets)              
        if self.training:
            roi_proposals, roi_labels, roi_reg_targets = data_sample.sample_proposals_batch(rpn_proposals, targets)              
        else:
            roi_proposals = rpn_proposals
            roi_labels = None
            roi_reg_targets = None       
        bbox_deltas, class_logits = self.roi_head(feature, roi_proposals)  
        bbox_deltas = rearrange(bbox_deltas, '(b m) n->b m n', b=batch_size)
        class_logits = rearrange(class_logits, '(b m) n->b m n', b=batch_size)
        
        detections = []
        roi_loss = None
        fastrcnn_loss = None
        if self.training:
            roi_loss = self.roi_head_loss(class_logits, bbox_deltas, roi_labels, roi_reg_targets)
            fastrcnn_loss = roi_loss + self.rpn_loss_weight * rpn_loss
        else:
            detections = self.roi_head_post_process(class_logits, 
                                                    bbox_deltas, 
                                                    roi_proposals,
                                                    resized_image_sizes)  
        return detections, fastrcnn_loss