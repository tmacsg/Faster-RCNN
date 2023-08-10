from typing import Tuple
import torch
from einops import repeat

def create_anchors(img_size: Tuple[int, int],
                   feature_size: Tuple[int, int],
                   anchor_size_ratios: Tuple[int, ...],
                   anchor_aspect_ratios: Tuple[float, ...]):
    """Create anchors mapped to image

    Args:
        img_size (Tuple[height, width]): image size 
        feature_size (Tuple[int, int]): feature map size
        anchor_size_ratios (Tuple[int, ...]): ahchor size rario with respect to max image size
        anchor_aspect_ratios (Tuple[float, ...]): width/height ratio of each anchor

    Returns:
        Tensor : anchors mapped to image, in xyxy format
    """
    base_anchors = _create_base_anchors(img_size, anchor_size_ratios, anchor_aspect_ratios)
    strides_x = torch.arange(0, img_size[1] // feature_size[1] * feature_size[1],  img_size[1] // feature_size[1])
    strides_y = torch.arange(0, img_size[0] // feature_size[0] * feature_size[0],  img_size[0] // feature_size[0])
    offsets_x, offsets_y = torch.meshgrid(strides_x, strides_y, indexing='xy')
    offsets_x, offsets_y = offsets_x.reshape(-1), offsets_y.reshape(-1)  
    base_offsets = torch.stack([offsets_x, offsets_y, offsets_x, offsets_y]).T 
    anchors = repeat(base_anchors, 'h w->(repeat h) w', repeat=base_offsets.shape[0])
    offsets = repeat(base_offsets, 'h w->(h repeat) w', repeat=base_anchors.shape[0])
    anchors = anchors + offsets
    return anchors

def _create_base_anchors(img_size: Tuple[int, int],
                         anchor_size_ratios: Tuple[int, ...], 
                         anchor_aspect_ratios: Tuple[float, ...]):
    """Create anchors with origin at (0,0)

    Args:
        img_size (Tuple[height, width]): image size 
        anchor_size_ratios (Tuple[int, ...]): ahchor size rario with respect to image size (area)
        anchor_aspect_ratios (Tuple[float, ...]): width/height ratio of each anchor

    Returns:
        Tensor : anchors with origin at (0,0), in xyxy format
    """
    ref_size = img_size[0] * img_size[1]
    anchor_sizes = torch.as_tensor([anchor_size_ratio * anchor_size_ratio * ref_size 
                                    for anchor_size_ratio in anchor_size_ratios])
    anchor_sizes = torch.sqrt(anchor_sizes)
    anchor_aspect_ratios = torch.as_tensor(anchor_aspect_ratios)
    h_ratios = torch.sqrt(anchor_aspect_ratios)
    w_ratios = 1.0 / h_ratios
    ws = (w_ratios[:, None] * anchor_sizes[None, :]).view(-1)
    hs = (h_ratios[:, None] * anchor_sizes[None, :]).view(-1)
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
    return base_anchors.round()

    
        