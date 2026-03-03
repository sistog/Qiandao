import torch
import torch.nn as nn

class AcousticViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.patch_embed = nn.Conv2d(1, 512, 16, 16)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, 512))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
            )

        self.transformer=nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=6
        )

        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # print("输入形状:", x.shape)  # [B, 1, 128, 256]
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.transformer(x)

        cls_output = x[:, 0]
        out = self.head(cls_output)

        return out

if __name__ == "__main__":
    # 实例化模型
    model = AcousticViT(num_classes=10)

    # 构造假数据
    x = torch.randn(4, 1, 128, 512)   # batch=4

    # 前向传播
    out = model(x)

    print("输出形状:", out.shape)