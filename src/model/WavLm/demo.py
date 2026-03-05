import torch
from transformers import WavLMConfig, WavLMModel

# config = WavLMConfig
# model = WavLMModel(config)
# config = model.config
# print(config)

model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
config = model.config
x = torch.randn(1,16000)  # 1秒音频

outputs = model(x)
print(outputs)  # 输出的特征维度