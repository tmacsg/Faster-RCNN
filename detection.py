from modules.faster_rcnn import FasterRCNN, backbone
from utils.utils import load_checkpoints, load_model
from utils import box, data_prepare
import math
import cv2
from torchvision import transforms
import torch
from PIL import Image
from utils import visualize
import random

def detect(image_paths: list[str]):
    original_images = [cv2.imread(image_path) for image_path in image_paths]
    orig_image_sizes = [original_image.shape[:2] for original_image in original_images]

    model, _, _, _ = load_checkpoints('checkpoints/checkpoint_epoch_30.pth', FasterRCNN())
    # model = load_model(FasterRCNN(), 'models/FasterRCNN_MobilenetV2_F7B3_best.pth')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformed_images = [transform(Image.fromarray(img)) for img in original_images]
    resized_images = [data_prepare.resize_image(image) for image in transformed_images]
    images, resized_image_sizes, _ = tuple(zip(*resized_images))
    resized_image_sizes = torch.as_tensor([list(resized_image_size) for resized_image_size in resized_image_sizes])
    max_size, _ = torch.max(resized_image_sizes, axis=0)
    stride = 32.0
    max_size[0] = int(math.ceil(float(max_size[0]) / stride) * stride)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    batch_aligned_imgs = torch.zeros((len(images), 3, max_size[0], max_size[1]))
    for img, aligned_img in zip(images, batch_aligned_imgs):
        aligned_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
       
    model.eval()
    with torch.inference_mode():
        detections, _ = model(batch_aligned_imgs, resized_image_sizes)

    filtered_detections = []
    for i, detection in enumerate(detections):
        if len(detection['boxes']) == 0:
            filtered_detections.append(detection)
        else:
            boxes, scores, labels = box.batched_nms(detection['boxes'], detection['scores'], detection['labels'], iou_threshold=0.3)
            boxes = box.resize_boxes(boxes, orig_image_sizes[i], resized_image_sizes[i])
            filtered_detections.append({'boxes': boxes, 'scores': scores, 'labels': labels})

    for image_path, result in zip(image_paths, filtered_detections):
        visualize.display_boxes(image_path, result['boxes'], result['labels'], result['scores'], display=True, saving=False)

if __name__ == '__main__':
    train_image_paths, test_image_paths = data_prepare.get_image_paths()
    random.shuffle(test_image_paths)
    detect(test_image_paths[0:15])
 
