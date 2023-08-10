import torch
import config
from modules.faster_rcnn import FasterRCNN, backbone
from utils import utils, data_setup, engine

device = "cuda" if torch.cuda.is_available() else "cpu"
model = FasterRCNN(backbone=backbone.mobilenetv2_backbone())
optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=config.LEARNING_RATE, 
                            momentum=config.MOMENTUM, 
                            weight_decay=config.WEIGHT_DECAY)
writer = utils.create_writer(config.EXPERIMENT_NAME, config.MODEL_NAME)
utils.set_seeds()

# train_dataloader, test_dataloader = data_setup.get_dataloader(num_train=120, num_test=32)
train_dataloader, test_dataloader = data_setup.get_dataloader_by_ratio()
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             optimizer=optimizer,
             epochs=config.NUM_EPOCH,
             device=device,
             writer=writer,
             checkpoint_path=None)

