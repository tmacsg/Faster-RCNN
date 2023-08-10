from typing import List, Union
from numpy.typing import NDArray
from torch import Tensor
import os
import cv2
from copy import deepcopy
from pathlib import Path
import config
from utils.data_prepare import get_class_idx_dict


def display_boxes(image_source: Union[str, NDArray], 
                  bboxes: Union[Tensor,List[Tensor]], 
                  class_ids: Tensor=None, 
                  probs: List[float]=None, 
                  display: bool=True, 
                  saving: bool=False):
    """Display bounding boxes on image, optioanl to save the processed image

    Args:
        image_source (Union[str, NDArray]) : 
        bboxes (Tensor or List[Tensor]) : list of boudning boxes, shape [N,4]
        class_ids (Tensor) : class ids, shape [N,]. Defaults to None
        probs (Tensor) : probabilities, shape [N,]. Defaults to None
        display (Bool) : display image if True. Defaults to True
        saving (Bool) : save image if True. Defaults to False
    """

    bgr_colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 128),  # Gray
        (255, 255, 255)   # White
    ]


    _, idx2class = get_class_idx_dict()
    color_map = {}
    if class_ids is not None:
        class_names = [idx2class[class_id.cpu().item()] for class_id in class_ids]
        unique_names = set(class_names)
        for i, name in enumerate(unique_names):
            color_map[name] = bgr_colors[i]

    if isinstance(image_source, str):
        img = cv2.imread(image_source)
    else:
        img = deepcopy(image_source)
    # image_shape = str(img.shape[0]) + ', ' + str(img.shape[1])
    # cv2.putText(img, image_shape, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    for i in range(len(bboxes)):
        x1,y1,x2,y2 = bboxes[i].cpu().int().numpy()
        cv2.rectangle(img, (x1,y1), (x2,y2), bgr_colors[i % len(bgr_colors)], 2)
        if class_ids is not None:
            class_name =  idx2class[class_ids[i].cpu().item()]
            if probs is not None:
                prob = round(probs[i].cpu().item(),3)      
                text = class_name + ':' + str(prob) 
            cv2.putText(img, text, (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (x1,y1), (x2,y2), color_map[class_name], 2)
        else:
            cv2.rectangle(img, (x1,y1), (x2,y2), bgr_colors[0], 2)

    if saving and isinstance(image_source, str):
        path = Path(config.image_save_dir)
        path.mkdir(parents=True, exist_ok=True)
        image_name = image_source.split(os.sep)[-1]
        image_path = path / image_name
        cv2.imwrite(str(image_path), img)
    if display:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        