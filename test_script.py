"""Test functions during development
"""

from utils import utils, data_prepare, visualize, box, anchor, transforms, data_sample
from modules import backbone, roi_head, rpn
from modules.faster_rcnn import FasterRCNN
import random
import numpy as np
import torch
import config
from torchvision.ops import box_iou
import cv2
import utils.data_setup as data_setup

def test_set_seeds():
    utils.set_seeds()
    x_array = np.random.rand(10)
    x_tensor = torch.rand(10)
    z1 = random.sample(x_array.tolist(), 5)

    utils.set_seeds()
    y_array = np.random.rand(10)
    y_tensor = torch.rand(10)
    z2 = random.sample(y_array.tolist(), 5)
   
    print(np.sum(x_array == y_array) == 10)
    print(torch.sum(x_tensor == y_tensor).item() == 10)
    print(z1 == z2)

def test_get_class_idx_dict():
    class2idx, idx2class = data_prepare.get_class_idx_dict()
    print(class2idx, idx2class)

def test_get_image_paths():
    train_image_paths, test_image_paths = data_prepare.get_image_paths()
    print(len(train_image_paths), len(test_image_paths))
    print(train_image_paths[0])

def test_parse_image_annot():
    train_image_paths, test_image_paths = data_prepare.get_image_paths()
    image_path = train_image_paths[105]
    image_annot_path = data_prepare.get_image_annot(image_path) 
    print(f'image path: {image_path}')
    print(f'image annotation path: {image_annot_path}')
    targets = data_prepare.parse_image_annot(image_annot_path)
    for i in range(len(targets['labels'])):
        print(f"\tlabel: {targets['labels'][i]},\
              bbox: {targets['boxes'][i]},\
              area: {targets['area'][i]},\
              iscrowd: {targets['iscrowd'][i]}")

def test_display_boxes():
    train_image_paths, _ = data_prepare.get_image_paths()
    image_path = train_image_paths[random.randint(0, len(train_image_paths))]
    image_annot_path = data_prepare.get_image_annot(image_path) 
    target = data_prepare.parse_image_annot(image_annot_path)
    visualize.display_boxes(image_path, target['boxes'].int(), target['labels'], saving=False)

def test_bbox_xyxy2xywh():
    x1y1 = torch.randint(low=0, high=300, size=(5,2))
    x2y2 = torch.randint(low=300, high=600, size=(5,2))
    box_xyxy = torch.concat((x1y1, x2y2), axis=1)
    box_xywh = box.bbox_xyxy2xywh(box_xyxy)
    print(f'before conversion: {box_xyxy}')
    print(f'after conversion: {box_xywh}')

def test_bbox_xywh2xyxy():
    box_xywh = torch.randint(low=0, high=600, size=(5,4))
    box_xyxy = box.bbox_xywh2xyxy(box_xywh)
    print(f'before conversion: {box_xywh}')
    print(f'after conversion: {box_xyxy}')

def test_remove_small_boxes():
    x1y1 = torch.randint(low=0, high=300, size=(5,2))
    x2y2 = torch.randint(low=300, high=600, size=(5,2))
    box_xyxy = torch.concat((x1y1, x2y2), axis=1)
    print(box_xyxy)
    filtered_box = box.remove_small_boxes(box_xyxy, min_size=200)
    print(filtered_box)

def test_clip_boxes_to_image():
    x1y1 = torch.randint(low=-100, high=100, size=(5,2))
    x2y2 = torch.randint(low=100, high=200, size=(5,2))
    box_xyxy = torch.concat((x1y1, x2y2), axis=1)
    print(box_xyxy)
    clipped_box = box.clip_boxes(box_xyxy, [100, 150])
    print(clipped_box)

def test_batched_nms():
    x1y1 = torch.randint(low=0, high=300, size=(5,2))
    x2y2 = torch.randint(low=300, high=600, size=(5,2))
    box_xyxy = torch.concat((x1y1, x2y2), axis=1)
    scores = torch.rand(5)
    class_idxes = torch.randint(0,2,(5,))
    print(box_xyxy, scores, class_idxes)
    print('ious: ', box_iou(box_xyxy, box_xyxy))
    filtered_boxes = box.batched_nms(box_xyxy.float(), scores, class_idxes, iou_threshold=0.2)
    print(filtered_boxes)

def test_encode_boxes():
    x1y1 = torch.randint(low=0, high=300, size=(5,4))
    x2y2 = torch.randint(low=300, high=600, size=(5,4))
    gt_xyxy = torch.concat((x1y1[:,0:2], x2y2[:,0:2]), axis=1)
    proposals = torch.concat((x1y1[:,2:4], x2y2[:,2:4]), axis=1)
    targets = box.encode_boxes(gt_xyxy, proposals)
    print(targets)
 
def test_decode_boxes():
    x1y1 = torch.randint(low=0, high=300, size=(5,2))
    x2y2 = torch.randint(low=300, high=600, size=(5,2))
    box_xyxy = torch.concat((x1y1, x2y2), axis=1)
    regression_codes = torch.rand(5,4)
    pre_boxes = box.decode_boxes(box_xyxy, regression_codes)
    print(pre_boxes)
    
def test_match_proposals():
    train_dataloader, test_dataloader = data_setup.get_dataloader()
    images, targets = next(iter(train_dataloader))
    batch_size = images.shape[0]
    proposals = [anchor.create_anchors((448,448),(14,14),config.anchor_size_ratios, config.anchor_aspect_ratios) 
                 for i in range(batch_size)]
    proposals = torch.stack(proposals)
    matched_boxes, matched_labels = box.match_proposals(proposals, targets)
    print(f'bg: {torch.sum(matched_labels == 0)}')
    print(f'discard: {torch.sum(matched_labels == -1)}')
    print(f'obj: {torch.sum(matched_labels >= 1)}')
    print(matched_boxes.shape)
    print(matched_labels.shape)
    
def test_sample_proposals():
    train_dataloader, test_dataloader = data_setup.get_dataloader()
    images, targets = next(iter(train_dataloader))
    batch_size = images.shape[0]
    proposals = [anchor.create_anchors((448,448),(14,14),config.anchor_size_ratios, config.anchor_aspect_ratios) 
                 for i in range(batch_size)]
    proposals = torch.stack(proposals)
    matched_boxes, matched_labels = box.match_proposals(proposals, targets)
    obj_boxes_indexes, bg_boxes_indexes = data_sample.sample_proposals(matched_labels[0])
    print(f'obj: {len(obj_boxes_indexes)}, bg: {len(bg_boxes_indexes)}')
    
    
def test__create_base_anchors():
    base_anchors = anchor._create_base_anchors((600,800),
                                anchor_size_ratios=config.anchor_size_ratios,
                                anchor_aspect_ratios=config.anchor_aspect_ratios)
    print(base_anchors)

def test_create_anchors():
    
    train_image_paths, _ = data_prepare.get_image_paths()
    # image_path = train_image_paths[random.randint(0, len(train_image_paths))]
    image_path = train_image_paths[0]
    image_annot_path = data_prepare.get_image_annot(image_path) 
    target = data_prepare.parse_image_annot(image_annot_path)
    image = cv2.imread(image_path)
    width, height = image.shape[:2]
    anchors = anchor.create_anchors((height, width), (height//32, width//32), config.anchor_size_ratios, config.anchor_aspect_ratios)
    print(anchors.shape)
    visualize.display_boxes(image_path, anchors[1494:1494+18])

def test_transforms():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(prob=1),        
        transforms.Contrast_Brightness(prob=1),
        transforms.Blur(prob=1),
        transforms.Resize(config.IMAGE_SIZE),
        # transforms.ToTensor(),
        # transforms.Normalize()
    ])
    
    train_image_paths, _ = data_prepare.get_image_paths()
    image_path = train_image_paths[random.randint(0, len(train_image_paths))]
    # image_path = train_image_paths[0]
    image_annot_path = data_prepare.get_image_annot(image_path) 
    target = data_prepare.parse_image_annot(image_annot_path)
    visualize.display_boxes(image_path, target['boxes'].int())
    image = cv2.imread(image_path)
    image, target = transform(image, target)
    print(image.shape, image.dtype)
    visualize.display_boxes(image, target['boxes'].int())

def test_dataset():
    train_image_paths, _ = data_prepare.get_image_paths()
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(prob=0.5),        
        transforms.Contrast_Brightness(prob=1),
        transforms.Blur(prob=1),
        # transforms.ToTensor(),
        # transforms.Normalize()
    ])

    dataset = data_setup.VOC07DataSet(train_image_paths, transform)
    print(len(dataset))
    image, target = dataset[0]
    print(target)
    visualize.display_boxes(image, target['boxes'].int())
    
def test_dataloader():
    train_image_paths, test_image_paths = data_prepare.get_image_paths()
    print(f'batch size: {config.BATCH_SIZE}')
    print(f'total train images: {len(train_image_paths)}')
    print(f'total test images: {len(test_image_paths)}')
    train_dataloader, test_dataloader = data_setup.get_dataloader()
    print(f'total train bacthes: {len(train_dataloader)}')
    print(f'total test bacthes: {len(test_dataloader)}')
    
    train_batch = next(iter(train_dataloader))
    images, image_sizes, targets = train_batch
    print(f'batched image shape: {images.shape}')
    print(f'image sizes before align: {image_sizes}')
    print(f'number of targets: {len(targets)}')
    print(targets[0])
 
def test_backbone_mobilenetv2():
    model = backbone.mobilenetv2_backbone()
    x = torch.rand(1,3,config.IMAGE_SIZE, config.IMAGE_SIZE) 
    y = model(x)
    print(f'out channels: {model.out_channels}')
    print(f'input: {x.shape}, outpur: {y.shape}')

def test_RPNHead():
    features = torch.rand(1, 256, 50, 50)
    rpn_head = rpn.RPNHead(256, 9)
    logits, bbox_regs = rpn_head(features)
    print(logits.shape, bbox_regs.shape)

def test_RPN():
    img_size = config.IMAGE_SIZE
    feature_size = img_size // 32
    backbone_ = backbone.mobilenetv2_backbone()
    num_anchors = len(config.anchor_size_ratios) * len(config.anchor_aspect_ratios)
    rpn_head = rpn.RPNHead(backbone_.out_channels, num_anchors)
    model = rpn.RPN(
        (img_size, img_size),
        (feature_size, feature_size),
        rpn_head
    )
    model.eval()
    x = torch.rand(8, backbone_.out_channels, feature_size, feature_size)
    logits, _ = model(x)
    print(logits.shape)
    print(model.anchors.shape)

def test_fasterrcnn():
    train_dataloader, test_dataloader = data_setup.get_dataloader()   
    model = FasterRCNN()
    images, targets = next(iter(train_dataloader))
    # model.eval()
    detections, losses = model(images, targets)    
    if model.training:
        print(losses)
    else:
        print(len(detections))
        print(detections[0]['boxes'].shape, 
            detections[0]['labels'].shape, 
            detections[0]['scores'].shape)
        
def test_find_image_size():
    train_image_paths, test_image_paths = data_prepare.get_image_paths()
    image_paths = train_image_paths + test_image_paths
    max_sizes, min_ratio, max_ratio = data_prepare.find_image_size(image_paths)
    print(max_sizes, min_ratio, max_ratio)
    
if __name__ == '__main__':
    # test_set_seeds()
    # test_get_class_idx_dict()
    # test_get_image_paths()
    # test_parse_image_annot()
    # test_display_boxes()
    # test_bbox_xyxy2xywh()
    # test_bbox_xywh2xyxy()
    # test_remove_small_boxes()
    # test_clip_boxes_to_image()
    # test_batched_nms()
    # test_encode_boxes()
    # test_decode_boxes()
    # test_match_proposals()
    # test_sample_proposals()
    # test__create_base_anchors()
    # test_create_anchors()
    # test_transforms()
    # test_dataset()
    # test_dataloader()
    # test_backbone_mobilenetv2()
    # test_RPNHead()
    # test_RPN()
    # test_fasterrcnn()
    test_find_image_size()
