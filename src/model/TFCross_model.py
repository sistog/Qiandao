"""
Multi-Branch Cross-Attention Model for Audio Classification
Architecture: T-F Analysis -> Embedding -> Self-Attention Encoder
             -> Cross Attention (Feature Fusion) -> MLP Head -> Soft Voting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────
#  1. T-F Feature Extractors (per branch)
# ─────────────────────────────────────────────

class TFBranchEncoder(nn.Module):
    """
    Converts a single T-F image (spectrogram / CQT / STFT etc.)
    into a sequence of patch embeddings, then encodes with Transformer.

    Args:
        img_size    : (H, W) of the input T-F image
        patch_size  : size of each patch (square)
        in_channels : 1 for grayscale, 3 for RGB spectrograms
        embed_dim   : token / embedding dimension
        num_heads   : attention heads in the encoder
        depth       : number of Transformer encoder layers
        mlp_ratio   : expansion ratio inside the FFN
        dropout     : dropout probability
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, \
            "img_size must be divisible by patch_size"

        num_patches = (H // patch_size) * (W // patch_size)

        # Patch embedding: Conv2d acts as a learned linear projection
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,          # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) T-F image

        Returns:
            cls  : (B, embed_dim) — CLS token for branch classification
            tokens: (B, N, embed_dim) — patch tokens for cross-attention
        """
        B = x.size(0)

        # Patch embedding  →  (B, embed_dim, h, w)  →  (B, N, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)           # (B, N, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)      # (B, N+1, D)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Self-attention encoder
        x = self.encoder(x)
        x = self.norm(x)

        cls    = x[:, 0]       # (B, D)
        tokens = x[:, 1:]      # (B, N, D)
        return cls, tokens


# ─────────────────────────────────────────────
#  2. Cross-Attention Fusion Module
# ─────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuses two token sequences using bidirectional cross-attention
    followed by a residual + FFN block (as in the 'Fuse' box in the figure).

    One branch attends to the other (Q from branch A, K/V from branch B),
    then the outputs are concatenated and projected.

    Args:
        embed_dim : common embedding dimension for all branches
        num_heads : number of attention heads
        dropout   : dropout probability
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Branch A queries branch B, and vice versa
        self.cross_attn_a = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_b = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

        # Projection after concatenation
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens_a: torch.Tensor,   # (B, Na, D)
        tokens_b: torch.Tensor,   # (B, Nb, D)
    ) -> torch.Tensor:
        """
        Returns fused representation: (B, Na, D)
        """
        # A attends to B
        a_fused, _ = self.cross_attn_a(tokens_a, tokens_b, tokens_b)
        a_fused = self.norm_a(tokens_a + a_fused)

        # B attends to A
        b_fused, _ = self.cross_attn_b(tokens_b, tokens_a, tokens_a)
        b_fused = self.norm_b(tokens_b + b_fused)
        

        # Pool to fixed length (mean) then concat + project
        a_pool = a_fused.mean(dim=1)   # (B, D)
        b_pool = b_fused.mean(dim=1)   # (B, D)

        return a_pool, b_pool
        # fused = torch.cat([a_pool, b_pool], dim=-1)   # (B, 2D)
        # fused = self.proj(fused)
        # fused = self.norm_out(fused)
        # return fused                                   # (B, D)


# ─────────────────────────────────────────────
#  3. MLP Classification Head
# ─────────────────────────────────────────────

class MLPHead(nn.Module):
    """
    MLP head that outputs per-class probabilities.

    Args:
        in_dim    : input feature dimension
        hidden_dim: hidden layer width
        num_classes: number of target classes
        dropout   : dropout probability
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)
        Returns:
            logits: (B, num_classes)
        """
        return self.net(x)


# ─────────────────────────────────────────────
#  4. Soft Voting (Confidence-Weighted Ensemble)
# ─────────────────────────────────────────────

class SoftVoting(nn.Module):
    """
    Combines predictions from multiple branches via soft (probability-weighted) voting.
    
    Confidence β for each branch is estimated from the entropy of its softmax output:
        β = 1 - H(p) / log(C)   (normalised, so β ∈ [0, 1])
    where H(p) is the entropy of the probability distribution.

    The final prediction is:
        p_final = Σ_i β_i * p_i  /  Σ_i β_i
    """

    def forward(
        self,
        logits_list: List[torch.Tensor],   # list of (B, C) tensors
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            final_probs : (B, C) weighted-average probability distribution
            confidences : list of (B,) confidence scores per branch
        """
        probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]
        C = probs_list[0].size(-1)
        log_C = torch.log(torch.tensor(C, dtype=torch.float32))

        confidences = []
        for p in probs_list:
            # Entropy-based confidence  (lower entropy → higher confidence)
            entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)   # (B,)
            beta = 1.0 - entropy / log_C                        # (B,)
            confidences.append(beta)

        # Weighted average
        beta_stack = torch.stack(confidences, dim=1)            # (B, K)
        beta_stack = beta_stack / (beta_stack.sum(dim=1, keepdim=True) + 1e-8)

        probs_stack = torch.stack(probs_list, dim=1)            # (B, K, C)
        final_probs = (beta_stack.unsqueeze(-1) * probs_stack).sum(dim=1)  # (B, C)

        return final_probs, confidences


# ─────────────────────────────────────────────
#  5. Full Model
# ─────────────────────────────────────────────

class TFCrossAttnModel(nn.Module):
    """
    Full pipeline as shown in the figure:

        [T-F images] → TFBranchEncoder (×num_branches)
                     → CrossAttentionFusion (pairwise, produces fused branch)
                     → MLPHead (per branch + fused branch)
                     → SoftVoting
                     → Final Prediction  +  Average Loss

    Args:
        num_branches : how many T-F representations (e.g. 3: STFT, CQT, Mel)
        num_classes  : number of output classes
        img_size     : (H, W) of each T-F image
        patch_size   : patch size for the ViT-style encoder
        in_channels  : channels per T-F image
        embed_dim    : transformer embedding dimension
        num_heads    : attention heads
        depth        : encoder depth per branch
        mlp_ratio    : FFN expansion
        dropout      : dropout rate
    """

    def __init__(
        self,
        num_branches: int = 3,
        num_classes: int = 10,
        img_size: Tuple[int, int] = (128, 128),
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_branches = num_branches

        # ── Branch encoders (Self-Attention) ──────────────────────────
        self.branch_encoders = nn.ModuleList([
            TFBranchEncoder(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=depth,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_branches)
        ])

        # ── Cross-Attention Fusion (Fuse boxes) ───────────────────────
        # We fuse every pair of adjacent branches; produces num_branches-1 fused tokens
        # Then one final fusion across all branches
        self.cross_fusions = nn.ModuleList([
            CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_branches)
        ])

        # Global fusion from all branch CLS tokens
        self.global_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_branches, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # ── MLP Heads ─────────────────────────────────────────────────
        # One head per branch (single-feature branches) + one for fused
        total_heads = num_branches * 2          # branches + global fused
        self.mlp_heads = nn.ModuleList([
            MLPHead(embed_dim, embed_dim * 2, num_classes, dropout)
            for _ in range(total_heads)
        ])

        # ── Soft Voting ───────────────────────────────────────────────
        self.soft_voting = SoftVoting()

    def forward(
        self,
        images: List[torch.Tensor],        # list of (B, C, H, W), length = num_branches
    ) -> dict:
        """
        Args:
            images: list of T-F spectrograms, one per branch

        Returns:
            dict with keys:
                'final_probs'   : (B, num_classes)  soft-voted probabilities
                'branch_logits' : list of (B, num_classes) per-head logits
                'confidences'   : list of (B,) confidence scores
                'avg_loss_input': (B, num_classes) average of branch softmax probs
        """
        assert len(images) == self.num_branches, \
            f"Expected {self.num_branches} input images, got {len(images)}"

        # ── Step 1: Self-Attention Encoding per branch ─────────────────
        cls_list    = []   # (B, D) per branch
        token_list  = []   # (B, N, D) per branch

        for i, encoder in enumerate(self.branch_encoders):
            cls, tokens = encoder(images[i])
            cls_list.append(cls)
            token_list.append(tokens)

        # ── Step 2: Cross-Attention Fusion ─────────────────────────────
        # Pairwise fusion (branch i ↔ branch i+1)
        fused_tokens = []
        for i, cross_fuse in enumerate(self.cross_fusions):
            fused1, fused2 = cross_fuse(token_list[i-1], token_list[i])   # (B, D)
            fused_tokens.append(fused1)
            fused_tokens.append(fused2)

        # # Global fusion: concatenate all branch CLS tokens → project
        # global_feat = torch.cat(cls_list, dim=-1)          # (B, D * num_branches)
        # global_feat = self.global_fusion(global_feat)      # (B, D)

        # ── Step 3: MLP Heads ──────────────────────────────────────────
        # Single-branch heads
        all_logits = []
        for i in range(0, len(self.mlp_heads)):
            logits = self.mlp_heads[i](fused_tokens[i])        # (B, C)
            all_logits.append(logits)
        # print(len(all_logits))

        # # Fused / global head
        # fused_logits = self.mlp_heads[-1](global_feat)     # (B, C)
        # all_logits.append(fused_logits)

        # ── Step 4: Soft Voting ────────────────────────────────────────
        final_probs, confidences = self.soft_voting(all_logits)

        # Average loss target (average of softmax probs across all heads)
        avg_probs = torch.stack(
            [F.softmax(l, dim=-1) for l in all_logits], dim=0
        ).mean(dim=0)                                       # (B, C)

        return {
            'final_probs'   : final_probs,       # (B, C) — use for inference
            'branch_logits' : all_logits,        # list of (B, C) — use for loss
            'confidences'   : confidences,       # list of (B,) per head
            'avg_probs'     : avg_probs,         # (B, C) — average loss reference
        }


# ─────────────────────────────────────────────
#  6. Loss Function
# ─────────────────────────────────────────────

class MultiHeadLoss(nn.Module):
    """
    Computes:
        total_loss = mean over heads of CE(logits_i, labels)
                   + lambda_kl * KL(avg_probs || uniform)   [optional regulariser]

    Args:
        num_classes : number of target classes
        lambda_kl   : weight on the KL regulariser (set 0 to disable)
    """

    def __init__(self, num_classes: int, lambda_kl: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lambda_kl = lambda_kl
        self.num_classes = num_classes

    def forward(self, output: dict, labels: torch.Tensor) -> torch.Tensor:
        branch_logits = output['branch_logits']
        avg_probs     = output['avg_probs']

        # Average CE loss over all heads
        ce_loss = sum(self.ce(logits, labels) for logits in branch_logits)
        ce_loss /= len(branch_logits)

        total_loss = ce_loss

        # Optional: KL divergence of avg distribution from uniform (diversity reg.)
        if self.lambda_kl > 0:
            uniform = torch.full_like(avg_probs, 1.0 / self.num_classes)
            kl = F.kl_div(avg_probs.log(), uniform, reduction='batchmean')
            total_loss = total_loss + self.lambda_kl * kl

        return total_loss


# ─────────────────────────────────────────────
#  7. Quick Smoke Test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Hyper-parameters ───────────────────────────────────────────────
    BATCH       = 4
    NUM_BRANCHES = 3     # e.g. STFT / Mel / CQT
    NUM_CLASSES  = 4
    IMG_H, IMG_W = 128, 128
    PATCH_SIZE   = 16
    EMBED_DIM    = 256

    # ── Model & loss ───────────────────────────────────────────────────
    model = TFCrossAttnModel(
        num_branches=NUM_BRANCHES,
        num_classes=NUM_CLASSES,
        img_size=(IMG_H, IMG_W),
        patch_size=PATCH_SIZE,
        in_channels=1,
        embed_dim=EMBED_DIM,
        num_heads=8,
        depth=4,
    ).to(device)

    criterion = MultiHeadLoss(num_classes=NUM_CLASSES, lambda_kl=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ── Dummy forward pass ─────────────────────────────────────────────
    dummy_images = [
        torch.randn(BATCH, 1, IMG_H, IMG_W, device=device)
        for _ in range(NUM_BRANCHES)
    ]
    labels = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)

    output = model(dummy_images)
    loss   = criterion(output, labels)

    print(f"final_probs  shape : {output['final_probs'].shape}")    # (4, 10)
    print(f"# branch logits    : {len(output['branch_logits'])}")   # 4
    print(f"loss               : {loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Backward pass OK ✓")

    # Param count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")