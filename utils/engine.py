import torch
from tqdm.auto import tqdm, trange
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
from .utils import save_checkpoint, save_model
import config
from torchmetrics.detection import MeanAveragePrecision
from utils.utils import load_checkpoints

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> float:
    model.train()
    train_loss = 0
    for i, (images, resized_image_sizes, targets) in enumerate(tqdm(dataloader, desc='Train Batch')):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        _, loss = model(images, resized_image_sizes, targets)
        train_loss += loss.data.item()       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              device: torch.device,
              metric: MeanAveragePrecision=MeanAveragePrecision()) -> float:
    model.eval() 
    map = []
    map_50 = []
    map_75 = []
    with torch.inference_mode():
        for i, (images, resized_image_sizes, targets) in enumerate(tqdm(dataloader, desc='Test Batch')):
            images, resized_image_sizes = images.to(device), resized_image_sizes.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            detections, _ = model(images, resized_image_sizes, targets)    

            metric.update(detections, targets)
            result = metric.compute()
            map.append(result['map'].item())
            map_50.append(result['map_50'].item())
            map_75.append(result['map_75'].item())
    map = sum(map) / len(map)
    map_50 = sum(map_50) / len(map_50)
    map_75 = sum(map_75) / len(map_75)
    return map, map_50, map_75

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          writer: SummaryWriter = None,
          checkpoint_path: str = None) -> Dict[str, List]:
 
    print(f"Start training, using {device}")
    best_map = 0
    results = {"train_loss": [], "map": [], "map_50": [], "map_75": []}   

    start_epoch = 0
    if checkpoint_path is not None:
        model, optimizer, start_epoch, _ = load_checkpoints(checkpoint_path, model, optimizer, device)

    model.to(device)
    for epoch in trange(epochs - start_epoch, desc="Epoch"):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                device=device)
        map, map_50, map_75 = test_step(model=model,
                              dataloader=test_dataloader,
                              device=device)

        print(f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"map: {map:.4f} | "
          f"map_50: {map_50:.4f} | "
          f"map_75: {map_75:.4f} | ")
        
        if writer:
            writer.add_scalar('loss/train loss', train_loss, epoch + start_epoch + 1)
            writer.add_scalar('mAP/map', map, epoch + start_epoch + 1)   
            writer.add_scalar('mAP/map_50', map_50, epoch + start_epoch + 1)  
            writer.add_scalar('mAP/map_75', map_75, epoch + start_epoch + 1)  
            writer.close()                          
        else:
            pass

        results["train_loss"].append(train_loss)
        results["map"].append(map)
        results["map_50"].append(map_50)
        results["map_75"].append(map_75)

        if map > best_map:
            best_map = map
            save_model(model=model, model_name=f"{config.MODEL_NAME}_best.pth") 
        
        if (epoch + start_epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch + start_epoch + 1, results)

    print(f"Training done.")
    return results