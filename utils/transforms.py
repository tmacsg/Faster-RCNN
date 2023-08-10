import random
import torch
from torchvision.transforms import functional as F
import config
import cv2
import numpy as np

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class Normalize(object):
    def __call__(self, image, target):       
        image = F.normalize(image, 
                            [0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, width = image.shape[:2]
            image = cv2.flip(image, 1) 
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target
    
class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target):
        h, w = image.shape[:2]
        image = cv2.resize(image, dsize=(self.size, self.size))       
        w_ratio, h_ratio = self.size / w, self.size / h
        bboxes = target['boxes']
        ratios = torch.as_tensor([w_ratio, h_ratio, w_ratio, h_ratio], dtype=torch.float32).expand_as(bboxes)
        bboxes = bboxes * ratios
        target['boxes'] = bboxes
        return image, target

class Blur(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            ksize = random.choice([3,5,7])
            image = cv2.blur(image, (ksize, ksize))
        return image, target
    
class Contrast_Brightness(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            alpha = np.random.rand() * 2
            beta = np.random.rand() * 10 - 10
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            image = np.clip(image, 0, 255)
        return image, target
    