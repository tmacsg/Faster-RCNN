from torch.utils.data import Dataset, DataLoader
import torch
from utils import data_prepare 
import config
import cv2
import random

class VOC07DataSet(Dataset):
    def __init__(self, image_paths, transforms=None):
        """
        Args:
            image_paths (List[str]) : image paths
            transform (torchvision.transforms.transforms)
        """
        super().__init__()
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_annot_path = data_prepare.get_image_annot(image_path)
        image = cv2.imread(image_path)
        target = data_prepare.parse_image_annot(image_annot_path)
        target.update({'image_id': torch.tensor([idx])})
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def get_dataloader(num_train: int=None, num_test: int=None):
    """

    Args:
        num_train (int, optional): Defaults to None.
        num_test (int, optional): Defaults to None.

    Returns:
        Tuple[train_dataloader, test_dataloader]: 
    """
    train_image_paths, test_image_paths = data_prepare.get_image_paths()
    train_transforms, test_transforms = data_prepare.get_data_transforms()
  
    if num_train is not None and num_test is not None:  # create small dataset for testing purpose
        random.shuffle(train_image_paths)
        random.shuffle(test_image_paths)
        train_dataset = VOC07DataSet(train_image_paths[0:num_train], transforms=train_transforms)
        test_dataset = VOC07DataSet(test_image_paths[0:num_test], transforms=test_transforms)
    else:   # full VOC 
        train_dataset = VOC07DataSet(train_image_paths, transforms=train_transforms)
        test_dataset = VOC07DataSet(test_image_paths, transforms=test_transforms)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=data_prepare.collate_fn,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=data_prepare.collate_fn,
                                 drop_last=True)
    return train_dataloader, test_dataloader

def get_dataloader_by_ratio(train_val_ratio: float=0.98):
    """

    Args:
        train_val_ratio (int, optional): Defaults to 0.98

    Returns:
        Tuple[train_dataloader, test_dataloader]: 
    """
    train_image_paths, test_image_paths = data_prepare.get_image_paths()
    image_paths = train_image_paths + test_image_paths
    random.shuffle(image_paths)
    num_train = int(len(image_paths) * train_val_ratio)
    train_transforms, test_transforms = data_prepare.get_data_transforms()
    train_dataset = VOC07DataSet(image_paths[0:num_train], transforms=train_transforms)
    test_dataset = VOC07DataSet(image_paths[num_train:], transforms=test_transforms)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=data_prepare.collate_fn,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=data_prepare.collate_fn,
                                 drop_last=True)
    return train_dataloader, test_dataloader