"""
ViT-Base architecture with configurable positional encoding.

Supports four PE families:
- learned: standard learnable position embeddings
- sinusoidal: fixed sinusoidal embeddings (Vaswani et al.)
- rope: rotary position embeddings (Su et al.) applied in attention
- alibi: attention with linear biases (Press et al.)

Buffer naming follows the training-time convention from prior experiments
to ensure checkpoint compatibility (see model_loader for state_dict loading).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


PE_TYPES = ["learned", "sinusoidal", "rope", "alibi"]


# ============================================================================
# Positional Encoding Modules
# ============================================================================

class LearnedPE(nn.Module):
    """Standard learnable positional embeddings (ViT-style)."""

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional embeddings."""

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        position = torch.arange(num_patches + 1).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                              -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_patches + 1, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe


class RoPE(nn.Module):
    """Rotary position embeddings applied in attention (Q, K rotation)."""

    def __init__(self, head_dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin tables
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        seq_len = q.size(-2)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class ALiBiBias(nn.Module):
    """Attention with Linear Biases (Press et al.)."""

    def __init__(self, num_heads, max_seq_len=512):
        super().__init__()
        # Pre-broadcast shapes to match training-time convention:
        # slopes: (1, num_heads, 1, 1) for direct addition to attention scores
        # rel_dist: (1, 1, seq, seq)
        slopes = torch.tensor(self._get_slopes(num_heads))
        slopes = slopes.view(1, num_heads, 1, 1)
        self.register_buffer("slopes", slopes)

        pos = torch.arange(max_seq_len)
        rel_dist = (pos[None, :] - pos[:, None]).float()  # (seq, seq)
        rel_dist = rel_dist.view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("rel_dist", rel_dist)

    @staticmethod
    def _get_slopes(num_heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power)
            extra = get_slopes_power_of_2(2 * closest_power)[0::2][:num_heads - closest_power]
            return slopes + extra

    def forward(self, seq_len):
        # Returns bias to be added to attention scores: (1, num_heads, seq, seq)
        rel = self.rel_dist[:, :, :seq_len, :seq_len]
        return self.slopes * rel.abs() * -1.0


# ============================================================================
# Attention Module (variant-aware)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE or ALiBi."""

    def __init__(self, embed_dim, num_heads, pe_type, dropout=0.0, max_seq_len=512):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pe_type = pe_type

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        if pe_type == "rope":
            self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)
        elif pe_type == "alibi":
            self.alibi = ALiBiBias(num_heads, max_seq_len=max_seq_len)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.pe_type == "rope":
            q, k = self.rope(q, k)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N)

        if self.pe_type == "alibi":
            bias = self.alibi(N)  # (1, num_heads, seq, seq) -- broadcasts over batch
            attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# Transformer Block + Full Model
# ============================================================================

def make_mlp(embed_dim, mlp_ratio, dropout=0.0):
    """
    MLP block matching the training-time layout (nn.Sequential).
    State_dict keys are mlp.0.{weight,bias} (first Linear) and
    mlp.3.{weight,bias} (second Linear); indices 1/2/4 are activation
    and dropout layers, which have no parameters.
    """
    hidden = int(embed_dim * mlp_ratio)
    return nn.Sequential(
        nn.Linear(embed_dim, hidden),   # mlp.0
        nn.GELU(),                       # mlp.1
        nn.Dropout(dropout),             # mlp.2
        nn.Linear(hidden, embed_dim),    # mlp.3
        nn.Dropout(dropout),             # mlp.4
    )


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, pe_type, dropout=0.0, max_seq_len=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, pe_type, dropout, max_seq_len)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = make_mlp(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)             # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=100,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 pe_type="learned"):
        super().__init__()
        if pe_type not in PE_TYPES:
            raise ValueError(f"pe_type must be one of {PE_TYPES}, got {pe_type}")

        self.pe_type = pe_type
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # PE module (only for learned/sinusoidal — rope/alibi inject in attention)
        if pe_type == "learned":
            self.pos_encoding = LearnedPE(self.num_patches, embed_dim)
        elif pe_type == "sinusoidal":
            self.pos_encoding = SinusoidalPE(self.num_patches, embed_dim)
        else:
            self.pos_encoding = nn.Identity()

        self.pos_drop = nn.Dropout(dropout)

        max_seq_len = self.num_patches + 1  # +1 for CLS token
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, pe_type, dropout, max_seq_len)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)              # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)       # (B, num_patches+1, embed_dim)
        x = self.pos_encoding(x)             # additive PE (or identity)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                        # CLS token

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.head(features)
        return logits
