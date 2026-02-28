import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN2D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # -------- Block 1 --------
            nn.Conv2d(1, 4, kernel_size=5, stride=(1,1), padding=(2, 2)),
            nn.BatchNorm2d(4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # -------- Block 2 --------
            nn.Conv2d(4, 16, kernel_size=5, stride=(1,1), padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # -------- Block 3 --------
            nn.Conv2d(16, 32, kernel_size=3, stride=(1,2), padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1),

            # -------- Block 4 --------
            nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )

        # Linear 输入维度 = 64 * 最终序列长度
        # 可以先打印 features 输出 shape 来确认长度
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 1 * 32 * 32),  # 128 根据你的输入长度和卷积参数调整
            nn.ReLU(),
            nn.Linear(1 * 32 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        x: [B, C, T, F]
        """
        x = self.features(x)   # [B, 64, L_out]
        # print(x.shape)
        x = torch.flatten(x, 1)       # [B, 64*L_out]
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
    
if __name__ == "__main__":
    model = AudioCNN2D(num_classes=2)
    input_tensor = torch.randn(8, 1, 128, 512)  # Example input: batch size 8, 1 channel, length 2048
    output = model(input_tensor)
    print(output.shape)  # Expected output shape: [8, 10]
