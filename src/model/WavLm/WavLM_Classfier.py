import torch
import torch.nn as nn
from transformers import WavLMModel

class WavLMClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WavLMClassifier, self).__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.classifier = nn.Linear(self.wavlm.config.hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        outputs = self.wavlm(x)
        hidden_states = outputs.last_hidden_state  # 获取最后一层的隐藏状态
        pooled_output = hidden_states.mean(dim=1)  # 对时间维度进行平均池化
        logits = self.classifier(pooled_output)  # 分类器输出
        return logits