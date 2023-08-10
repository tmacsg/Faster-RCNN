data_split_dir = '../data/VOCdevkit/VOC2007/ImageSets/Main'
train_list_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
val_list_path = '../data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
image_dir = '../data/VOCdevkit/VOC2007/JPEGImages'
annotation_dir = '../data/VOCdevkit/VOC2007/Annotations'

model_dir = 'models'
checkpoint_dir = 'checkpoints'
image_save_dir = 'images'

# rpn parameters
anchor_size_ratios = (0.2, 0.4, 0.6, 0.8)
anchor_aspect_ratios = (0.5, 1.0, 2)
rpn_fg_iou_thresh = 0.7
rpn_bg_iou_thresh = 0.3
rpn_batch_size_per_image = 128
rpn_positive_fraction = 0.5
rpn_pre_nms_top_n = 2000
rpn_post_nms_top_n = 1000
rpn_nms_thresh=0.7

# roi_head parameters
box_score_thresh=0.05
box_nms_thresh=0.5
box_detections_per_img=100
box_fg_iou_thresh=0.5
box_bg_iou_thresh=0.5
box_batch_size_per_image=256
box_positive_fraction=0.25

# EXPERIMENT_NAME = 'FasterRCNN_test'
# MODEL_NAME = 'FasterRCNN_test'
EXPERIMENT_NAME = 'FasterRCNN_MobilenetV2_F7B3'
MODEL_NAME = 'FasterRCNN_MobilenetV2_F7B3'
NUM_EPOCH = 100
IMAGE_MIN_SIZE = 500
IMAGE_MAX_SIZE = 800
ROI_SPATIAL_RATIO = 1.0 / 32    # feature map size / input size, depends on backbone
BATCH_SIZE = 8
NUM_CLASS = 21
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4