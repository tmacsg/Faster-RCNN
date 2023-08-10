import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import vgg16, VGG16_Weights

class mobilenetv2_backbone(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        if pretrain:
            weights = MobileNet_V2_Weights.DEFAULT
            model = mobilenet_v2(weights=weights)
        else:
            model = mobilenet_v2(weights=None)
            
        self.model = model.features
        # for params in self.model.parameters():
        #     params.requires_grad = False
        self._out_channels = 1280
        
    def forward(self, x):
        return self.model(x)
    
    @property
    def out_channels(self):
        return self._out_channels
    
class vgg16_backbone(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        if pretrain:
            weights = VGG16_Weights.DEFAULT
            model = vgg16(weights=weights)
        else:
            model = vgg16(weights=None)
            
        self.model = model.features
        self._out_channels = 512
        
    def forward(self, x):
        return self.model(x)
    
    @property
    def out_channels(self):
        return self._out_channels
    