from typing import Tuple
from torch import Tensor
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import anchor, box, data_sample
from einops import rearrange, repeat

class RPN(nn.Module):
    def __init__(self, head=None):
        super().__init__()
        self.loss_fn = RPNLoss()
        self.head = head
               
    def forward(self, features, image_size, targets=None):
        device = features.device
        batch_size = features.shape[0]
        logits, bbox_reg = self.head(features)
        objectness = F.sigmoid(logits)

        anchors = anchor.create_anchors(image_size, 
                                        features.shape[-2:],
                                        config.anchor_size_ratios,
                                        config.anchor_aspect_ratios)
                     
        anchors = repeat(anchors, 'm n->repeat m n', repeat=batch_size).to(device)
        objectness = rearrange(objectness, 'b c h w->b (h w c)')        
        bbox_reg = rearrange(bbox_reg, 'b (n c) h w->b (h w n) c', c=4)
        
        proposals = box.decode_boxes(anchors.reshape(-1,4), bbox_reg.data.reshape(-1,4))
        proposals = box.clip_boxes(proposals, image_size)
        proposals = rearrange(proposals, '(b m) n->b m n', b=batch_size)
        filtered_proposals = data_sample.filter_proposals(proposals, objectness.data)
                
        rpn_loss = None                        
        if self.training:
            matched_boxes, matched_labels = box.match_proposals(anchors, targets)
            bbox_reg_target = box.encode_boxes(matched_boxes.reshape(-1,4), anchors.reshape(-1,4))
            bbox_reg_target = bbox_reg_target.reshape(batch_size, -1, 4)       
            rpn_loss = self.loss_fn(bbox_reg, objectness, bbox_reg_target, matched_labels)
        return filtered_proposals, rpn_loss
    
class RPNHead(nn.Module):
    """Add a RPN head with classification and regression
    
    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, feature: Tensor) -> Tuple[Tensor,Tensor]:
        """
        Args:
            feature (torch.Tensor): feature from backbone, shape (N,C,H,W)
        Returns:
            Tuple[logits, bbox_reg]: classification and bbox predictions
        """
        t = F.relu(self.conv(feature))
        logits = self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)
        return logits, bbox_reg
    
class RPNLoss(nn.Module):
    def __init__(self, reg_loss_weight: float=1.0):
        super().__init__()
        self.reg_loss_weight = reg_loss_weight
        self.regression_loss = nn.SmoothL1Loss()
        self.classification_loss = nn.BCELoss()
        
    def forward(self, bbox_reg: Tensor, 
                obejctness: Tensor, bbox_reg_target: Tensor, 
                matched_labels: Tensor) -> Tuple[Tensor,Tensor]:
        """Compute regression and classification loss for RPN 

        Args:
            bbox_reg (Tensor): shape [B,N,4]
            obejctness (Tensor): shape [B,N]
            bbox_reg_target (Tensor): shape [B,N,4]
            matched_labels (Tensor): shape [B,N]

        Returns:
            Tuple[reg_loss,cls_loss]: 
        """
        device = bbox_reg.device
        batch_size = bbox_reg.shape[0]
        reg_loss = []
        cls_loss = []
        for i in range(batch_size):
            cur_bbox_reg, cur_obejctness = bbox_reg[i], obejctness[i]
            cur_bbox_reg_target, cur_matched_labels = bbox_reg_target[i], matched_labels[i]            
            obj_boxes_indexes, bg_boxes_indexes = data_sample.sample_proposals(cur_matched_labels,
                                                                               config.rpn_batch_size_per_image,
                                                                               config.rpn_positive_fraction)            
            total_sampled_indexes = obj_boxes_indexes + bg_boxes_indexes
                        
            cur_matched_labels[torch.where(cur_matched_labels > 0)[0]] = 1
            if len(obj_boxes_indexes) == 0:
                cur_reg_loss = torch.tensor([0.]).to(device)
            else:
                cur_reg_loss = self.regression_loss(cur_bbox_reg[obj_boxes_indexes], 
                                                    cur_bbox_reg_target[obj_boxes_indexes])                
            cur_cls_loss = self.classification_loss(cur_obejctness[total_sampled_indexes], 
                                                    cur_matched_labels[total_sampled_indexes].float())           
            reg_loss.append(cur_reg_loss)
            cls_loss.append(cur_cls_loss)
        reg_loss = sum(reg_loss) / batch_size
        cls_loss = sum(cls_loss) / batch_size
        return cls_loss + self.reg_loss_weight * reg_loss