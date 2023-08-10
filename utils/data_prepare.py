from typing import Tuple, Dict
import os
import numpy as np
from torch import Tensor
import torch
import config
import xmltodict
from utils import transforms
import torch.nn.functional as F
import math
import cv2

def get_class_idx_dict():
    """ Get class and index dict

    Returns:
        class2idx (Dict) : {class name : id}
        idx2class (Dict) : {id : class name}
    """
    classes = ['background']
    for file in os.listdir(config.data_split_dir):
        if '_' in file and file.split('_')[0] not in classes:
            classes.append(file.split('_')[0])
    class2idx = {value: index for index, value in enumerate(classes)}
    idx2class = {index: value for index, value in enumerate(classes)}
    return class2idx, idx2class


def get_image_paths():
    """ Get train and test image paths

    Returns:
        train_image_paths (List[str]) 
        test_image_paths (List[str]) 
    """
    train_list = np.loadtxt(config.train_list_path, dtype='str')
    test_list = np.loadtxt(config.val_list_path, dtype='str')
    train_image_paths = [os.path.join(config.image_dir, train_list[i] + '.jpg') 
                         for i in range(len(train_list))]
    test_image_paths = [os.path.join(config.image_dir, test_list[i] + '.jpg') 
                        for i in range(len(test_list))]
    
    return train_image_paths, test_image_paths

def get_image_annot(image_path: str):
    """ Get annotation path for image_path

    Returns:
        (str) : image annotation path
    """
    return os.path.join(config.annotation_dir, 
                        image_path.split(os.path.sep)[-1].split('.')[0] + '.xml')
    
def parse_image_annot(image_annot_path: str):
    """Get class names and grounding bbox from image annotation

    Args:
        image_annot_path (str): image annotation path

    Returns:
        target (Dict)
    """   
    class2idx, _ = get_class_idx_dict()
    annot = xmltodict.parse(open(image_annot_path, 'rb'))
    objs = annot['annotation']['object']
    labels = []
    gtboxes = []
    iscrowd = []
   
    if not isinstance(objs, list):
        objs = [objs]
    for obj in objs:
        labels.append(class2idx[obj['name']])
        gtboxes.append(list(map(float, obj['bndbox'].values())))
        if "difficult" in obj:
            iscrowd.append(int(obj["difficult"]))
        else:
            iscrowd.append(0)

    gtboxes = torch.as_tensor(gtboxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
    area = (gtboxes[:, 3] - gtboxes[:, 1]) * (gtboxes[:, 2] - gtboxes[:, 0])

    target = {}
    target["boxes"] = gtboxes
    target["labels"] = labels
    target["area"] = area
    target["iscrowd"] = iscrowd
    return target

def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Create data transforms

    Returns:
        Tuple[train_transforms, test_transforms] 
    """
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(prob=0.5),        
        transforms.Contrast_Brightness(prob=0.5),
        transforms.Blur(prob=0.5),
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
    ])
    return train_transforms, test_transforms

def resize_image(image: Tensor, 
                 target: Dict=None,
                 min_size: int=config.IMAGE_MIN_SIZE, 
                 max_size: int=config.IMAGE_MAX_SIZE) -> Tuple[Tensor,Tuple[int,int],Dict]:
    """Resize image to range of (min_size, max_size), keep the image w/h ratio unchanged

    Args:
        image (Tensor): shape [C,H,W]
        target (Dict, optional): Defaults to None.
        min_size (int, optional): Defaults to config.IMAGE_MIN_SIZE.
        max_size (int, optional): Defaults to config.IMAGE_MAX_SIZE.

    Returns:
        Tuple[image,resized_image_size,target]: 
    """
    h, w  = image.shape[-2:]
    image_min_size = float(min(h, w))
    image_max_size = float(max(h, w))
    scale_ratio = min_size / image_min_size
    if image_max_size * scale_ratio > max_size:
        scale_ratio = max_size / image_max_size

    image = F.interpolate(image[None], 
                          scale_factor=scale_ratio, mode="bilinear", 
                          recompute_scale_factor=True,
                          align_corners=False)[0]
    
    resized_image_size = image.shape[-2:]
    if target is not None:
        w_ratio, h_ratio = resized_image_size[1] / float(w), resized_image_size[0] / float(h)
        bboxes = target['boxes']
        ratios = torch.as_tensor([w_ratio, h_ratio, w_ratio, h_ratio], dtype=torch.float32).expand_as(bboxes)
        bboxes = bboxes * ratios
        target['boxes'] = bboxes
    return image, resized_image_size, target


def image_batch_align(images: Tuple[Tensor,...], 
                      size_divisible: int=32):
    """Pad images to the same size to create a batch

    Args:
        images (Tuple[Tensor,...]): images with different sizes in a tuple
        size_divisible (int, optional): Defaults to 32.

    Returns:
        Tensor: batch of images, shape [B,C,H,W]
    """
    image_sizes = torch.as_tensor([list(img.shape[-2:]) for img in images])
    max_size, _ = torch.max(image_sizes, axis=0)
    stride = float(size_divisible)
    max_size[0] = int(math.ceil(float(max_size[0]) / stride) * stride)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)

    batch_aligned_imgs = torch.zeros((len(images), 3, max_size[0], max_size[1]))

    for img, aligned_img in zip(images, batch_aligned_imgs):
        aligned_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
    return batch_aligned_imgs

def collate_fn(batch: list[Tuple[Tensor, Dict]]) -> Tuple[Tensor,Tuple[Dict,...]]: 
    """Create a bacth from list of (image, image_sizes, target)

    Args:
        batch (list[Tuple[image, target]]): 

    Returns:
        Tuple[Tensor,Tensor,Tuple[target,...]]: 
            padded_images (Tensor): a batch of images, shape [B,C,H,W]
            resized_image_sizes (Tensor): resized image sizes 
            targets Tuple[Dict,...]: each Dict is a target for each image
    """
    resized_batch = [resize_image(image, target) for image, target in batch]
    images, resized_image_sizes, targets = tuple(zip(*resized_batch))
    padded_images = image_batch_align(images)
    resized_image_sizes = torch.as_tensor([list(resized_image_size) for resized_image_size in resized_image_sizes])
    return padded_images, resized_image_sizes, targets

def find_image_size(image_paths: list[str]) -> Tuple[Tensor,float,float]:
    """Get image max size and min/max h/w ratio

    Args:
        image_paths (List[str]):

    Returns:
        Tuple[max_sizes, min_ratio, max_ratio]: 
    """
    sizes = [cv2.imread(image_path).shape[:2] for image_path in image_paths]
    sizes = torch.as_tensor(sizes).float()
    max_sizes, _ = torch.max(sizes, axis=0)
    ratios = sizes[:,0] / sizes[:,1]
    min_ratio, max_ratio = torch.min(ratios), torch.max(ratios)
    return max_sizes, min_ratio, max_ratio
