import torch
import torch.nn as nn
import torch.nn.functional as F

class SliceEmbed(nn.Module):
    """
    1D-ViT 的核心：Slice 操作
    将 [Batch, Time, Freq] 转换为 [Batch, Time, Embed_Dim]
    """
    def __init__(self, freq_bins=128, embed_dim=512):
        super().__init__()
        self.proj = nn.Linear(freq_bins, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: [B, Time, Freq] 
        # 如果输入带 Channel 维 [B, 1, Time, Freq]，则需 squeeze(1)
        if x.dim() == 4:
            x = x.squeeze(1)
        
        x = self.proj(x) # [B, Time, Embed_Dim]
        x = self.norm(x)
        return x

class CrossAttention(nn.Module):
    """
    交叉注意力机制：用于融合 Mel 和 LOFAR 两个支路的特征
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_q, x_kv):
        B, N, C = x_q.shape
        # 生成 Query (来自支路 A), Key 和 Value (来自支路 B)
        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class CAF_ViT(nn.Module):
    def __init__(self, dim_a=128, dim_b=256, embed_dim=512, depth=6, num_heads=8, num_classes=4, max_time=512):
        super().__init__()
        
        # 1. Slice Embedding 层 (1D-ViT 特色)
        self.patch_embed_a = SliceEmbed(freq_bins=dim_a, embed_dim=embed_dim)
        self.patch_embed_b = SliceEmbed(freq_bins=dim_b, embed_dim=embed_dim)
        
        # 2. 位置编码 (1D 序列编码)
        self.pos_embed_a = nn.Parameter(torch.zeros(1, max_time, embed_dim))
        self.pos_embed_b = nn.Parameter(torch.zeros(1, max_time, embed_dim))
        
        # 3. Transformer 编码器块 (Self-Attention)
        self.blocks_a = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        self.blocks_b = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])
        
        # 4. Cross-Attention 融合层
        self.cross_attn = CrossAttention(dim=embed_dim, num_heads=num_heads)
        
        # 5. 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x_a, x_b):
        # Step 1: Slice Embedding & Positional Encoding
        # x_a: [B, Time, 128], x_b: [B, Time, 256]
        x_a = self.patch_embed_a(x_a) + self.pos_embed_a
        x_b = self.patch_embed_b(x_b) + self.pos_embed_b
        
        # Step 2: 独立支路 Self-Attention 提取时频特征
        for blk in self.blocks_a:
            x_a = blk(x_a)
        for blk in self.blocks_b:
            x_b = blk(x_b)
            
        # Step 3: Cross-Attention 融合 (支路 A 引导支路 B)
        # 也可以根据需要实现双向 Cross-Attention
        fused = self.cross_attn(x_a, x_b)
        
        # Step 4: 全局平均池化 & 分类
        # 沿时间维度平均
        out = fused.mean(dim=1) 
        out = self.norm(out)
        logits = self.head(out)
        
        return logits

# 测试模型维度
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模拟输入: [Batch, Time, Freq]
    test_mel = torch.randn(8, 512, 128).to(device)
    test_lofar = torch.randn(8, 512, 256).to(device)
    
    model = CAF_ViT(num_classes=4).to(device)
    output = model(test_mel, test_lofar)
    print(f"输入维度: {test_mel.shape}, {test_lofar.shape}")
    print(f"输出维度: {output.shape}") # 应为 [8, 4]