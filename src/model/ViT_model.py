import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(
        self, 
        img_size=(128, 256),      # 频谱图尺寸 (H, W)
        patch_size=16,     # 每个 Patch 的大小
        in_channels=1,     # 单通道频谱图
        num_classes=4,     # 你的水声类别数
        embed_dim=768,     # 嵌入维度（建议从128开始，防止过拟合）
        depth=6,           # Transformer 层数
        num_heads=8,       # 多头注意力的头数
        mlp_ratio=4.0,     # FFN 放大倍数
        dropout=0.1
    ):
        super().__init__()
        
        H, W = img_size 
        patch_h = patch_w = patch_size # 假设 patch 还是正方形的

        # 分别计算垂直和水平方向的 patch 数量，然后相乘
        self.num_patches = (H // patch_h) * (W // patch_w)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

        # 2. Learnable Tokens & Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 更加稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Final MLP Head
        self.norm = nn.LayerNorm(embed_dim)
        # 将原本的单层 Linear 改为两层 MLP
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim//2, num_classes)
        )

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # Patching & Projection -> (B, N, D)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Position Embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer layers
        x = self.transformer_encoder(x)
        
        # 只取第一个 token (CLS token) 的输出做分类
        x = self.norm(x[:, 0])
        return self.mlp_head(x)

# 快速测试
if __name__ == "__main__":
    model = SimpleViT(img_size=(128, 256), num_classes=4)
    dummy_input = torch.randn(8, 1, 128, 256) # (Batch, Channel, H, W)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") # [8, 4]