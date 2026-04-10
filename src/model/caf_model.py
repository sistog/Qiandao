import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    CAF-ViT 核心模块：实现模态间的跨模态注意力融合
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # 定义 Q, K, V 的线性映射
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x_q, x_kv):
        B, N, C = x_q.shape
        _, M, _ = x_kv.shape

        # 生成 Q (来自支路 A), K/V (来自支路 B)
        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_kv).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 矩阵乘法计算权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 融合特征
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CAF_ViT(nn.Module):
    """
    CAF-ViT 主模型：针对 DeepShip 四分类任务设计
    """
    def __init__(self, dim_a=512, dim_b=768, embed_dim=512, num_classes=4):
        super().__init__()
        
        # 1. 维度对齐层 (确保两个支路特征维度一致)
        self.align_a = nn.Linear(dim_a, embed_dim)
        self.align_b = nn.Linear(dim_b, embed_dim)
        
        # 2. 跨模态融合块 (A 融合 B 的信息)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
        self.cross_attn = CrossAttention(dim=embed_dim)
        
        self.dropout = nn.Dropout(0.05)

        self.linear = nn.Linear(embed_dim, num_classes)

        # 3. 分类头 (DeepShip 四分类)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, feat_a, feat_b):
        # feat_a: [B, N1, dim_a] (例如音频波形特征)
        # feat_b: [B, N2, dim_b] (例如时频谱特征)

        # 对齐并添加残差
        x_a = self.align_a(feat_a)
        x_b = self.align_b(feat_b)
        
        # 交叉融合：x_a 作为 Query 去查询 x_b 的关键信息
        fused = x_a + self.cross_attn(self.norm_a(x_a), self.norm_b(x_b))
        
        fused_drop = self.dropout(fused)
        # 全局池化并输出
        out = fused_drop.mean(dim=1)
        return self.mlp_head(out)

# --- 随机数据测试脚本 ---
if __name__ == "__main__":
    # 初始化模型，针对 DeepShip 4分类
    model = CAF_ViT(num_classes=4)
    
    # 模拟输入数据：Batch=8, 序列长度=128
    dummy_a = torch.randn(8, 128, 512) 
    dummy_b = torch.randn(8, 128, 768)
    
    logits = model(dummy_a, dummy_b)
    
    print(f"DeepShip 任务类别数: {logits.shape[1]}")
    print(f"模型输出 shape: {logits.shape}") # 应为 [8, 4]
    print("测试成功！模型可正常跑通。")