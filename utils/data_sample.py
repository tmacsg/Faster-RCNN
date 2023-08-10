from typing import Tuple, Dict
import torch
import config
import random
from utils import box
from torchvision.ops import nms

def sample_proposals(labels: torch.Tensor, 
                     num_sample_per_image: int=config.rpn_batch_size_per_image, 
                     fraction: float=config.rpn_positive_fraction):
    """Sample object and background indexes 

    Args:
        labels (torch.Tensor): shape [N,]
        num_sample_per_image (int, optional):  Defaults to config.rpn_batch_size_per_image.
        fraction (float, optional):  Defaults to config.rpn_positive_fraction.

    Returns:
        Tuple[obj_boxes_indexes, bg_boxes_indexes]: 
    """
    num_obj_per_image = int(num_sample_per_image * fraction)
    obj_indexes = torch.where(labels > 0)[0] 
    bg_indexes = torch.where(labels == 0)[0] 
    if num_obj_per_image > len(obj_indexes):
            obj_boxes_indexes = obj_indexes.tolist()
    else:
        obj_boxes_indexes = random.sample(obj_indexes.tolist(), num_obj_per_image)
    
    bg_count = num_sample_per_image - len(obj_boxes_indexes)  

    if bg_count > len(bg_indexes):
         bg_boxes_indexes = bg_indexes.tolist()
         obj_count = num_sample_per_image - len(bg_boxes_indexes) 
         obj_boxes_indexes = random.sample(obj_indexes.tolist(), obj_count)
    else:
        bg_boxes_indexes = random.sample(bg_indexes.tolist(), bg_count)
    return obj_boxes_indexes, bg_boxes_indexes
    
def sample_proposals_batch(proposals: torch.Tensor, 
                           targets: Dict,
                           fg_iou_thresh: float=config.box_fg_iou_thresh,
                           bg_iou_thresh: float=config.box_bg_iou_thresh,
                           num_proposals_per_image: int=config.box_batch_size_per_image,
                           positive_fraction: float=config.box_positive_fraction):
    """

    Args:
        proposals (torch.Tensor)
        targets (Dict)
        fg_iou_thresh (float, optional):  Defaults to config.box_fg_iou_thresh.
        bg_iou_thresh (float, optional):  Defaults to config.box_bg_iou_thresh.
        num_proposals_per_image (int, optional):  Defaults to config.box_batch_size_per_image.
        positive_fraction (float, optional):  Defaults to config.box_positive_fraction.

    Returns:
        Tuple[filtered_proposals, filtered_labels, filter_reg_targets]:
    """
    batch_size = proposals.shape[0]
    matched_boxes, matched_labels = box.match_proposals(proposals, 
                                                        targets,
                                                        fg_iou_thresh,
                                                        bg_iou_thresh)
    
    reg_targets = box.encode_boxes(matched_boxes.reshape(-1,4), proposals.reshape(-1,4))
    reg_targets = reg_targets.reshape(batch_size, -1, 4) 
    
    filtered_proposals = []
    filtered_labels = []
    filter_reg_targets = []
    for i in range(batch_size):
        cur_matched_labels = matched_labels[i]
        
        cur_obj_boxes_indexes, cur_bg_boxes_indexes = sample_proposals(cur_matched_labels, 
                                                                       num_proposals_per_image, 
                                                                       positive_fraction)
        cur_indexes = cur_obj_boxes_indexes + cur_bg_boxes_indexes
        filtered_proposals.append(proposals[i][cur_indexes])
        filtered_labels.append(matched_labels[i][cur_indexes])
        filter_reg_targets.append(reg_targets[i][cur_indexes])
        
    filtered_proposals = torch.stack(filtered_proposals)
    filtered_labels = torch.stack(filtered_labels)
    filter_reg_targets = torch.stack(filter_reg_targets)
    return filtered_proposals, filtered_labels, filter_reg_targets    


def filter_proposals(proposals: torch.Tensor, 
                     objectness: torch.Tensor,
                     num_proposer_pre_nms: int=config.rpn_pre_nms_top_n, 
                     num_proposer_post_nms: int=config.rpn_post_nms_top_n,
                     nms_thresh: float=config.rpn_nms_thresh):  
    """Filter propsals via objectness scores, then perform nms on them

    Args:
        proposals (torch.Tensor): 
        objectness (torch.Tensor): 
        num_proposer_pre_nms (int, optional): Defaults to config.rpn_pre_nms_top_n.
        num_proposer_post_nms (int, optional): Defaults to config.rpn_post_nms_top_n.
        nms_thresh (float, optional): Defaults to config.rpn_nms_thresh.

    Returns:
        filtered_proposals (torch.Tensor):
    """
    
    filtered_proposals = []
    proposal_cnts = []
    batch_size = proposals.shape[0]
    for i in range(batch_size):
        cur_proposal = proposals[i]
        cur_objectness = objectness[i]
        cur_filtered_proposals = filter_proposals_single(cur_proposal, 
                                                  cur_objectness,
                                                  num_proposer_pre_nms,
                                                  num_proposer_post_nms,
                                                  nms_thresh)        
        filtered_proposals.append(cur_filtered_proposals)
        proposal_cnts.append(len(cur_filtered_proposals))

    min_count = min(proposal_cnts)
    filtered_proposals = [filtered_proposal[:min_count] 
                          for filtered_proposal in filtered_proposals]
    filtered_proposals = torch.stack(filtered_proposals)
    return filtered_proposals
    
def filter_proposals_single(proposals: torch.Tensor, 
                     objectness: torch.Tensor,
                     num_proposer_pre_nms: int, 
                     num_proposer_post_nms: int,
                     nms_thresh: float,
                     min_count: int=config.box_batch_size_per_image):
    """ Filter proposals: select top K1 ->  remove small -> nms 

    Args:
        proposals (Tensor): shape [N,4]
        objectness (Tensor): shape [N,]
        num_proposer_pre_nms (int, optional): 
        nms_thresh (float, optional):
        min_count (int): minimum count of proposals to sample

    Returns:
        Tensor: shape [M,4]
    """

    device = proposals.device
    topk_objectness_pre_nms, topk_indexes_pre_nms = torch.topk(objectness, k=num_proposer_pre_nms)
    topk_proposals_pre_nms = proposals[topk_indexes_pre_nms]

    large_box_indexes = box.remove_small_boxes(topk_proposals_pre_nms)
    proposals_pre_nms = topk_proposals_pre_nms[large_box_indexes]
    objectness_pre_nms = topk_objectness_pre_nms[large_box_indexes]
    

    indexes_post_nms = nms(proposals_pre_nms.float(), objectness_pre_nms, iou_threshold=nms_thresh)

    # append min_count to the end in case of num_proposals < box_batch_size_per_image
    remaining_indexes = set(list(range(len(proposals_pre_nms)))) - set(list(indexes_post_nms.cpu()))
    added_indexes = torch.as_tensor(random.sample(list(remaining_indexes), min_count)).to(device)
    indexes_post_nms = torch.concatenate([indexes_post_nms, added_indexes])
        
    finale_proposals = proposals_pre_nms[indexes_post_nms[:num_proposer_post_nms]]
             
    return finale_proposals
    


    