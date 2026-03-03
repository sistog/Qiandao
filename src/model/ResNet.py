import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetAudio(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        in_channels=1,
    ):
        super(ResNetAudio, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 
            num_classes
            )
    
    def forward(self, x):
        """
        x: [B, C, F, T]
        """
        x = x.repeat(1, 3, 1, 1)  # 将单通道复制为三通道 [B, 3, F, T]
        out = self.backbone(x)
        return out