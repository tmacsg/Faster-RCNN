from typing import Dict, List, Tuple
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import config

def set_seeds(seed=42):
    """Sets random sets 
    
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def create_writer(experiment_name: str, model_name: str) :
    """ Create a tensorboard summarywriter

    Args:
        experiment_name (str)
        model_name (str)

    Returns:
        SummaryWriter
    """
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)       
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

def save_model(model: torch.nn.Module, model_name: str):
    """ save model weights

    Args:
        model (torch.nn.Module)
        target_dir (str)
        model_name (str)
    """
    target_dir_path = Path(config.model_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Save model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module,
               weights_path: str) -> torch.nn.Module:
    """ load weights to model

    Args:
        model (torch.nn.Module)
        weights_path (str)

    Returns:
        model (torch.nn.Module)
    """
    assert weights_path.endswith(".pth") or weights_path.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    print(f"[INFO] Loading model weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path))
    return model
    
def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    epoch: int, 
                    loss: Dict[str, List[float]]):
    """ save training checkpoint

    Args:
        model (torch.nn.Module)
        optimizer (torch.optim.Optimizer)
        epoch (int)
        loss (Dict[str, List[float]]) : dict of train loss list and test loss list
        target_dir (str)
    """
    target_dir_path = Path(config.checkpoint_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_save_path = target_dir_path / f'checkpoint_epoch_{epoch}.pth'

    print(f"[INFO] Saving checkpoint to: {checkpoint_save_path}")
    torch.save({        
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, checkpoint_save_path)
    
def load_checkpoints(checkpoint_path: str, 
                     model: torch.nn.Module=None, 
                     optimizer: torch.optim.Optimizer=None,
                     device: torch.device=torch.device('cpu')):
    """ load checkpoint

    Args:
        checkpoint_path (str) 
        model (torch.nn.Module, Optional) : Defaults to None
        optimizer (torch.optim.Optimizer, Optional) : Defaults to None
    
    Returns:
        model (torch.nn.Module)
        optimizer (torch.optim.Optimizer)
        epoch (int)
        loss (Dict[str, List[float]]) : dict of train loss list and test loss list
    """
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if model is not None:
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


