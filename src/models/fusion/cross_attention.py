# src/models/fusion/cross_attention.py

import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of visual features and environmental metadata.

    Visual features:
        [B, 49, 768]

    Metadata tokens:
        [B, 1, 256]

    Output:
        fused features [B, 256]
    """

    def __init__(
        self,
        visual_dim=768,
        metadata_dim=256,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        # ==========================================
        # STEP A: DIMENSION ALIGNMENT
        # ==========================================

        # Swin visual features: 768
        # Metadata features: 256
        # Project visual features to 256
        self.proj = nn.Linear(visual_dim, metadata_dim)

        # ==========================================
        # STEP B: MULTI-HEAD CROSS-ATTENTION
        # ==========================================

        self.attention = nn.MultiheadAttention(
            embed_dim=metadata_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ==========================================
        # STEP C: FIRST LAYER NORMALIZATION
        # ==========================================

        self.norm = nn.LayerNorm(metadata_dim)

        # ==========================================
        # STEP D: FEED-FORWARD NETWORK
        # ==========================================

        self.ffn = nn.Sequential(
            nn.Linear(metadata_dim, metadata_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_dim * 2, metadata_dim),
        )

        # ==========================================
        # STEP E: SECOND LAYER NORMALIZATION
        # ==========================================

        self.norm2 = nn.LayerNorm(metadata_dim)

    # ==========================================
    # FORWARD
    # ==========================================

    def forward(
        self,
        visual_features,
        metadata_tokens,
    ):
        """
        Args:
            visual_features:
                [B, 49, 768]

            metadata_tokens:
                [B, 1, 256]

        Returns:
            fused:
                [B, 256]

            attn_weights:
                Attention weights returned by
                MultiheadAttention.
        """

        # ==========================================
        # STEP 1: PROJECT VISUAL FEATURES
        # ==========================================

        visual_proj = self.proj(visual_features)

        # Shape:
        # [B, 49, 768]
        #       ↓
        # [B, 49, 256]

        # ==========================================
        # STEP 2: CROSS-ATTENTION
        # ==========================================

        attn_out, attn_weights = self.attention(
            query=visual_proj,
            key=metadata_tokens,
            value=metadata_tokens,
        )

        # attn_out:
        # [B, 49, 256]

        # ==========================================
        # STEP 3: RESIDUAL + LAYER NORM
        # ==========================================

        x = self.norm(visual_proj + attn_out)

        # ==========================================
        # STEP 4: FEED-FORWARD + RESIDUAL + NORM
        # ==========================================

        x = self.norm2(x + self.ffn(x))

        # ==========================================
        # STEP 5: MEAN POOLING
        # ==========================================

        fused = x.mean(dim=1)

        # Shape:
        # [B, 49, 256]
        #       ↓
        # [B, 256]

        # ==========================================
        # RETURN
        # ==========================================

        return fused, attn_weights
