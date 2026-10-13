import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim=768, metadata_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(visual_dim, metadata_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=metadata_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(metadata_dim)

        self.ffn = nn.Sequential(
            nn.Linear(metadata_dim, metadata_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_dim * 2, metadata_dim),
        )

        self.norm2 = nn.LayerNorm(metadata_dim)

    def forward(self, visual_features, metadata_tokens):
        visual_proj = self.proj(visual_features)

        attn_out, attn_weights = self.attention(
            query=visual_proj,
            key=metadata_tokens,
            value=metadata_tokens,
        )

        x = self.norm(visual_proj + attn_out)
        x = self.norm2(x + self.ffn(x))
        fused = x.mean(dim=1)

        return fused
