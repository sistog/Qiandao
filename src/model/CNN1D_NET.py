import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN1D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # -------- Block 1 --------
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # -------- Block 2 --------
            nn.Conv1d(4, 16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # -------- Block 3 --------
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),

            # -------- Block 4 --------
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
        )

        # Linear 输入维度 = 64 * 最终序列长度
        # 可以先打印 features 输出 shape 来确认长度
        self.classifier = nn.Sequential(
            nn.Linear(64 * 128, 4 * 128),  # 128 根据你的输入长度和卷积参数调整
            nn.ReLU(),
            nn.Linear(4 * 128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        x: [B, 1, F]
        """
        x = self.features(x)   # [B, 64, L_out]
        x = torch.flatten(x, 1)       # [B, 64*L_out]
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    model = AudioCNN1D(num_classes=2)
    input_tensor = torch.randn(8, 1, 8192)  # Example input: batch size 8, 1 channel, length 2048
    output = model(input_tensor)
    print(output.shape)  # Expected output shape: [8, 10]
