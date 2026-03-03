import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioLSTM(nn.Module):
    def __init__(self, num_classes=2, input_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        """ 
        x: [B, C, F, T]
        """
        x = x.squeeze(1)  # [B, F, T]
        x = x.permute(0, 2, 1)  # [B, T, F]
        lstm_out, _ = self.lstm(x)  # lstm_out: [B, T, hidden_dim]
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [B, hidden_dim]
        out = self.classifier(last_output)  # [B, num_classes]
        return out
