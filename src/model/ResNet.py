import torch
import torch.nn as nn
from torchvision import models

class ResNetAudio(nn.Module):
    def __init__(
        self,
        model_name="resnet18",  # 支持 'resnet18', 'resnet34', 'resnet50' 等
        num_classes=2,
        pretrained=True,
    ):
        super(ResNetAudio, self).__init__()
        
        # 1. 动态获取模型构造函数和权重
        if pretrained:
            # 使用更通用的方式获取权重枚举
            weights = models.get_model_weights(model_name).DEFAULT
        else:
            weights = None
            
        # 2. 加载基础模型
        self.backbone = models.get_model(model_name, weights=weights)

        # 1. 保存原有全连接层的输入维度
        in_features = self.backbone.fc.in_features
        
        # 3. 替换全连接层 (所有 ResNet 的末端层都叫 .fc)
        # 2. 将原有的 fc 替换为一个包含 Dropout 的 Sequential 序列
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        # 适配音频单通道到 ImageNet 的三通道输入
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)

# 使用示例
if __name__ == "__main__":
    model = ResNetAudio(model_name="resnet50", num_classes=10)