from typing import Tuple, Dict
import torch
from torch import Tensor
from einops import repeat
from torchvision.ops import box_iou, nms
import math
import config

def bbox_xyxy2xywh(oriBboxes: Tensor):
    """
    Args:
        oriBboxes (Tensor): N,4  (x1,y1,x2,y2) format

    Returns:
        newBboxes (Tensor): N,4  (ctr_x, ctr_y, width, height) format
    """
    newBboxes = torch.zeros_like(oriBboxes)
    newBboxes[:,0] = (oriBboxes[:,0] + oriBboxes[:,2]) / 2  # ctr_x
    newBboxes[:,1] = (oriBboxes[:,1] + oriBboxes[:,3]) / 2  # ctr_y
    newBboxes[:,2] = oriBboxes[:,2] - oriBboxes[:,0]   # width
    newBboxes[:,3] = oriBboxes[:,3] - oriBboxes[:,1]   # height  
    return newBboxes

def bbox_xywh2xyxy(oriBboxes: Tensor):
    """
    Args:
        oriBboxes (Tensor): N,4  (ctr_x, ctr_y, width, height) format

    Returns:
        newBboxes (Tensor): N,4  (x1,y1,x2,y2) format
    """
    newBboxes = torch.zeros_like(oriBboxes)
    newBboxes[:,0] = oriBboxes[:,0] - oriBboxes[:,2] / 2  # x1
    newBboxes[:,1] = oriBboxes[:,1] - oriBboxes[:,3] / 2  # y1
    newBboxes[:,2] = oriBboxes[:,0] + oriBboxes[:,2] / 2  # x2
    newBboxes[:,3] = oriBboxes[:,1] + oriBboxes[:,3] / 2  # y2  
    return newBboxes

def remove_small_boxes(boxes: Tensor, 
                       min_size: float=1.0):
    """
    Remove boxes which contains at least one side smaller than min_size.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        min_size (float): minimum size

    Returns:
        keep (Tensor[M]): indexes of boxes that have both sides larger than min_size
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    keep = torch.where(keep)[0]
    return keep

def clip_boxes(boxes: Tensor, 
               img_size: Tuple[int, int]):
    """
    Clip boxes so that they lie inside an image of size `size`.
    
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        img_size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    device = boxes.device
    h, w = img_size
    sizes = torch.tensor([[w,h,w,h]]).to(device)
    sizes = repeat(sizes, 'h w->(repeat h) w', repeat=boxes.shape[0])
    zeros = torch.zeros_like(boxes)
    clipped_boxes = boxes.clamp(min=zeros, max=sizes)
    return clipped_boxes

def batched_nms(boxes: Tensor, 
                scores: Tensor, 
                class_idxes: Tensor, 
                iou_threshold: float):
    """
    Performs non-maximum suppression in a batched fashion.

    Arguments:
        boxes : Tensor[N, 4], boxes in xyxy format
        scores : Tensor[N], scores for each one of the boxes
        class_idxes : Tensor[N], indices of the categories for each one of the boxes.
        iou_threshold : float
            discards all overlapping boxes
            with IoU < iou_threshold

    Returns:
        Tuple[Tensor[N, 4],Tensor[N],Tensor[N]] : 
            ALl the boxes that have been kept by NMS, sorted
            in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    max_coordinate = boxes.max()
    offsets = class_idxes.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return boxes[keep], scores[keep], class_idxes[keep]

def box_area(boxes):
    """Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes in xyxy format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def encode_boxes(gt_boxes: Tensor, proposals: Tensor):
    """Encode a set of proposals with respect to gt_boxes

    Arguments:
        gt_boxes (Tensor[N, 4]): grounding truth boxes, xyxy format
        proposals (Tensor[N, 4]): boxes to be encoded(anchors), xywh format
    
    Returns:
        targets (Tensor[N, 4]): boxes regression target
    """
    proposals_x1 = proposals[:, 0::4]
    proposals_y1 = proposals[:, 1::4]
    proposals_x2 = proposals[:, 2::4]
    proposals_y2 = proposals[:, 3::4]
    gt_boxes_x1 = gt_boxes[:, 0::4]
    gt_boxes_y1 = gt_boxes[:, 1::4]
    gt_boxes_x2 = gt_boxes[:, 2::4]
    gt_boxes_y2 = gt_boxes[:, 3::4]

    proposals_widths = proposals_x2 - proposals_x1
    proposals_heights = proposals_y2 - proposals_y1
    proposals_ctr_x = proposals_x1 + 0.5 * proposals_widths
    proposals_ctr_y = proposals_y1 + 0.5 * proposals_heights
    gt_widths = gt_boxes_x2 - gt_boxes_x1
    gt_heights = gt_boxes_y2 - gt_boxes_y1
    gt_ctr_x = gt_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = gt_boxes_y1 + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - proposals_ctr_x) / proposals_widths
    targets_dy = (gt_ctr_y - proposals_ctr_y) / proposals_heights
    targets_dw = torch.log(gt_widths / proposals_widths)
    targets_dh = torch.log(gt_heights / proposals_heights)
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

def decode_boxes(proposals: Tensor, regression_codes: Tensor, bbox_xform_clip=math.log(1000. / 16)):
    """

    Args:        
        proposals(Tensor[N, 4]): anchors/proposals
        regression_codes(Tensor[N, 4]): bbox regression parameters

    Returns:
        pred_boxes(Tensor[N, 4]): decoded boxes from proposals with regression_codes
    """
    proposals_x1 = proposals[:, 0::4]
    proposals_y1 = proposals[:, 1::4]
    proposals_x2 = proposals[:, 2::4]
    proposals_y2 = proposals[:, 3::4]
    proposals_widths = proposals_x2 - proposals_x1
    proposals_heights = proposals_y2 - proposals_y1
    proposals_ctr_x = proposals_x1 + 0.5 * proposals_widths
    proposals_ctr_y = proposals_y1 + 0.5 * proposals_heights
    
    dx = regression_codes[:, 0::4] 
    dy = regression_codes[:, 1::4] 
    dw = regression_codes[:, 2::4] 
    dh = regression_codes[:, 3::4] 
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)
    
    pred_ctr_x = dx * proposals_widths + proposals_ctr_x
    pred_ctr_y = dy * proposals_heights + proposals_ctr_y
    pred_w = torch.exp(dw) * proposals_widths
    pred_h = torch.exp(dh) * proposals_heights
    
    pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
    pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

    pred_boxes = torch.concat((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1)
    return pred_boxes.int()

def match_proposals(proposals: Tensor, 
                    targets: Dict,
                    fg_iou_thresh: float=config.rpn_fg_iou_thresh,
                    bg_iou_thresh: float=config.rpn_bg_iou_thresh) -> Tuple[Tensor,Tensor]:
    """Match proposals to grounding truth boxes, and assign the labels

    Args:
        proposals (Tensor): shape [B,N,4]
        targets (Dict): 
        fg_iou_thresh (float)
        bg_iou_thresh (float)
    Returns:
        Tuple[matched_boxes,matched_labels]: 
    """
    matched_boxes = []
    matched_labels = []
    batch_size = len(proposals)
    for i in range(batch_size):
        proposal_per_image = proposals[i]
        target = targets[i]
    
        gt_boxes = target['boxes']
        labels = target['labels']
        ious = box_iou(proposal_per_image, gt_boxes)
        max_ious, matched_indexes = torch.max(ious, axis=1)
                
        obj_indexes = max_ious >= fg_iou_thresh        
        bg_indexes = max_ious <= bg_iou_thresh, 
        discard_indexes = torch.logical_and(max_ious < fg_iou_thresh, max_ious > bg_iou_thresh)     
        
        matched_boxes_per_image = gt_boxes[matched_indexes]
        matched_labels_per_image = labels[matched_indexes]    
        matched_labels_per_image[bg_indexes] = 0  # background
        matched_labels_per_image[discard_indexes] = -1    # discard

        if not torch.any(obj_indexes):
            # if all ious are less than threshold, add the idx with max iou to obj_indexes
            max_ious_by_gtbox, max_ious_by_gtbox_idxes = torch.max(ious, axis=0)
            for i, idx in enumerate(max_ious_by_gtbox_idxes):
                matched_labels_per_image[idx] = labels[i] 
        matched_boxes.append(matched_boxes_per_image)
        matched_labels.append(matched_labels_per_image)
    matched_boxes = torch.stack(matched_boxes)
    matched_labels = torch.stack(matched_labels)
    return matched_boxes, matched_labels
    

def resize_boxes(boxes: Tensor, 
                 orig_image_size: Tuple[int, int],
                 resized_image_size: Tuple[int, int]):
    """Map predicted boxes coordinates back to original image

    Args:
        boxes (Tensor): shape [N,4]
        orig_image_size (Tuple[int,int]): 
        resized_image_size (Tuple[int,int]): resized image size before padding
    Returns:
        Tensor: shape [N,4]
    """
    device = boxes.device
    h_ratio, w_ratio = float(resized_image_size[0]) / orig_image_size[0], float(resized_image_size[1]) / orig_image_size[1]
    ratios = torch.as_tensor([w_ratio, h_ratio, w_ratio, h_ratio]).expand_as(boxes).to(device)
    resized_boxes = boxes / ratios
    resized_boxes = clip_boxes(resized_boxes, orig_image_size)
    return resized_boxes.int()