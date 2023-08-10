from typing import Tuple
from torch import Tensor
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from utils import data_sample, box
from einops import rearrange

class ROIHead(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.roi_pool = ROIPool()
  
        resolution = self.roi_pool.output_size
        representation_size = 1024
        self.predictor = FastRCNNPredictor(
            in_channel * resolution ** 2,
            representation_size
        )
      
    def forward(self, feature, proposals):
        output = self.roi_pool(feature, proposals)
        bbox_deltas, logits = self.predictor(output)
        return  bbox_deltas, logits
        
class ROIPool(nn.Module):
    def __init__(self, output_size: int=7):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, feature, proposals): # b n 4
        """

        Args:
            feature (Tensor): shape [B,C,H,W]
            proposals (Tensor): shape [B,N,4]

        Returns:
            Tensor: shape [B*N,C,output_size,output_size]
        """
        batch_size = feature.shape[0]
        proposal_list = [proposals[i].float() for i in range(batch_size)]
        output = roi_align(input = feature, 
                  boxes = proposal_list, 
                  output_size = self.output_size,
                  spatial_scale=config.ROI_SPATIAL_RATIO,
                  aligned = True)
        return output

    
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """
    def __init__(self, in_channels, representation_size, num_classes=config.NUM_CLASS):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas, logits
    
class ROIHeadLoss(nn.Module):
    def __init__(self, reg_loss_weight: float=1.0):
        super().__init__()
        self.reg_loss_weight = reg_loss_weight
        self.regression_loss = nn.SmoothL1Loss()
        self.classification_loss = nn.CrossEntropyLoss()
        
    def forward(self, class_logits: Tensor, 
                bbox_reg: Tensor, 
                labels: Tensor, 
                bbox_reg_targets: Tensor) -> Tuple[Tensor,Tensor]: 
        """Compute regression and classification loss for RPN 

        Args:
            class_logits (Tensor): shape [B,N,21]
            bbox_reg (Tensor): shape [B,N,21*4]
            labels (Tensor): shape [B,N]
            bbox_reg_targets (Tensor): shape [B,N,4]

        Returns:
            Tuple[reg_loss,cls_loss]: 
        """
        device = class_logits.device
        cls_loss = self.classification_loss(class_logits.view(-1,class_logits.shape[-1]), 
                                            labels.reshape(-1))
        class_probs, class_idxes = torch.max(class_logits, axis=2)        
        class_idxes = class_idxes.reshape(-1)
        obj_indexes = torch.where(class_idxes > 0)[0]
        
        if len(obj_indexes) == 0:
            reg_loss = torch.tensor([0.]).to(device)
        else:
            obj_bbox_reg_targets = bbox_reg_targets.reshape(-1,4)[obj_indexes]
            obj_bbox_reg = bbox_reg.view(-1,bbox_reg.shape[-1])[obj_indexes]
            obj_class_idxes = class_idxes[obj_indexes]
            indices = torch.tensor([[4*idx, 4*idx+1, 4*idx+2, 4*idx+3] for idx in obj_class_idxes]).to(device)        
            obj_bbox_reg_matched = obj_bbox_reg.gather(dim=1, index=indices)
            reg_loss = self.regression_loss(obj_bbox_reg_matched, obj_bbox_reg_targets)

        return cls_loss + self.reg_loss_weight * reg_loss
    
class ROIHeadPostprocess(nn.Module):
    def __init__(self, 
                 score_thresh: float=config.box_score_thresh,
                 nms_thresh: float=config.box_nms_thresh):
        super().__init__()
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
               
    def forward(self, 
                class_logits: Tensor, 
                bbox_deltas: Tensor,
                roi_proposals: Tensor,
                resized_image_sizes):  
        """
        Args:
            class_logits (Tensor): shape [B,N,NUM_CLASS]
            bbox_deltas (Tensor): shape [B,N,NUM_CLASS*4]
            roi_proposals (Tensor): shape [B,N,4]
            
        Returns:

        """
        device = class_logits.device
        batch_size = class_logits.shape[0]
        class_scores = F.softmax(class_logits, dim=2)
        class_probs, class_idxes = torch.max(class_scores, axis=2)  
        
        detections = []
        for i in range(batch_size):
            cur_class_idxes = class_idxes[i]
            cur_class_probs = class_probs[i]
            cur_roi_proposals = roi_proposals[i]
            cur_bbox_deltas = bbox_deltas[i]            
            cur_obj_idxes = torch.where(torch.logical_and(cur_class_idxes > 0, cur_class_probs > self.score_thresh))[0]          
            if len(cur_obj_idxes) == 0:
                detections.append({"boxes": torch.empty(0).to(device), 
                                   "labels": torch.empty(0).to(device), 
                                   "scores": torch.empty(0).to(device)})
            else:            
                cur_class_idxes = cur_class_idxes[cur_obj_idxes]
                cur_class_probs = cur_class_probs[cur_obj_idxes]
                cur_roi_proposals = cur_roi_proposals[cur_obj_idxes]
                cur_bbox_deltas = cur_bbox_deltas[cur_obj_idxes]
            
                indices = torch.tensor([[4*idx, 4*idx+1, 4*idx+2, 4*idx+3] for idx in cur_class_idxes]).to(device) 
                cur_bbox_deltas_matched = cur_bbox_deltas.gather(dim=1, index=indices)
                cur_roi_proposals = box.decode_boxes(cur_roi_proposals, cur_bbox_deltas_matched)
                cur_roi_proposals = box.clip_boxes(cur_roi_proposals, resized_image_sizes[i])                
                cur_boxes, cur_socres, cur_labels = box.batched_nms(cur_roi_proposals.float(),
                                                                    cur_class_probs,
                                                                    cur_class_idxes,
                                                                    self.nms_thresh)
                detections.append({"boxes": cur_boxes[0:config.box_detections_per_img], 
                                "labels": cur_labels[0:config.box_detections_per_img], 
                                "scores": cur_socres[0:config.box_detections_per_img]})            
        return detections
