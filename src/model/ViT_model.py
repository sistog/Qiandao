import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """ Stochastic Depth: 随机丢弃整个残差路径 """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class SliceEmbedding(nn.Module):
    def __init__(self, in_chans=1, embed_dim=384, slice_len=8, stride=4):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(128, 1),   # 👉 只在时间维切
            stride=(128, 1)
        )

    def forward(self, x):
        # x: (B, F, T)
        # x = x.unsqueeze(1)   # (B, 1, F, T)

        x = self.proj(x)     # (B, C, F, T')
        # print(x.shape)  # Debug: 查看卷积输出形状
        x = x.flatten(2).transpose(1, 2)

        return x             # (B, N, C)

class PatchEmbed(nn.Module):
    """ 
    原始 ViT 的 Patch Embedding：直接用一个大卷积实现切块和线性映射
    这里我们用 ConvStem 替代它，提取更丰富的底层特征
    """
    def __init__(self, img_size=(128, 256), patch_size=(16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 计算切块后的数量
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        # 用一个卷积层替代原始的切块 + 线性映射
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 输入 x 的 shape: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_h, W/patch_w]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class ConvStem(nn.Module):
    """ 
    卷积前置网络：代替原始简单的 Patch Embedding
    通过多层小卷积提取底层声学特征
    """
    def __init__(self, in_chans=1, embed_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)

class EnhancedAudioViT(nn.Module):
    def __init__(
        self, 
        img_size=(128, 256),    # 频谱图输入 (H, W)
        in_channels=1, 
        num_classes=4, 
        embed_dim=384,          # 中型维度，适合水声数据量
        depth=4, 
        num_heads=1, 
        mlp_ratio=2.0, 
        dropout=0.1,
        drop_path_rate=0.1      # 随机深度率
    ):
        super().__init__()
        
        # 1. Conv Stem 层
        # 经过 3 层 stride=2 的卷积后，H和W会缩小为原来的 1/8
        self.stem = SliceEmbedding(
            in_chans=in_channels,
            embed_dim=embed_dim,
            slice_len=64,   # 切片长度，时间维度上每8帧切一次
            stride=64     # 切片步长，时间维度上每4帧切一次，产生重叠切片
        )
        
        self.num_patches = 512

        # 2. Tokens & Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 绝对位置编码（如果输入长度固定）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Blocks (带 DropPath)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Classifier Head (Hybrid)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # [Step 1] 特征提取与 Patch 化
        # 输入 (B, 1, 128, 256) -> Stem -> (B, embed_dim, 16, 32)
        x = self.stem(x)
        # print(f"After Stem, x shape: {x.shape}")  # Debug: 查看 Stem 输出形状
        # x = x.flatten(2).transpose(1, 2) # (B, 512, embed_dim)
        
        # [Step 2] Add CLS & Position
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 513, embed_dim)
       
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # [Step 3] Transformer layers
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # [Step 4] 混合池化特征
        # cls_feature: 第一个 token
        # gap_feature: 对剩余所有 patch token 取平均
        cls_feature = x[:, 0]
        gap_feature = torch.mean(x[:, 1:], dim=1)
        
        combined = torch.cat([cls_feature, gap_feature], dim=1) # (B, embed_dim * 2)
        
        return self.head(combined)

# 测试代码
if __name__ == "__main__":
    # 模拟水声频谱图输入
    # 假设采样率和处理后得到 128x256 的 Fbank
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnhancedAudioViT(img_size=(128, 512), num_classes=4).to(device)
    
    dummy_input = torch.randn(8, 1, 128, 512).to(device)
    output = model(dummy_input)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")