"""
SugarcaneClassifier — Full multimodal model.

Wires together:
  SwinBackbone  →  [B, 49, 768]  visual patch tokens
  MetadataEncoder → [B,  1, 256]  environmental token
  CrossAttentionFusion → [B, 256]  fused context-aware vector
  Linear head  → [B, num_classes] disease logits
"""

import torch
import torch.nn as nn

from src.models.backbone.swin_backbone import SwinBackbone
from src.models.encoders.metadata_encoder import MetadataEncoder
from src.models.fusion.cross_attention import CrossAttentionFusion


class SugarcaneClassifier(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        num_classes = config["data"]["num_classes"]
        visual_dim = config["model"]["embed_dim"]  # 768
        metadata_dim = config["model"]["metadata_out_dim"]  # 256
        num_heads = config["model"]["num_attention_heads"]  # 8
        dropout = config["model"]["fusion_dropout"]  # 0.1

        self.backbone = SwinBackbone(
            model_name=config["model"]["backbone"],
            pretrained=False,  # switch to True when training for real
        )
        self.meta_encoder = MetadataEncoder(
            input_dim=config["data"]["metadata_dim"],  # 4
            hidden_dim=config["model"]["metadata_hidden_dim"],  # 128
            output_dim=metadata_dim,
        )
        self.fusion = CrossAttentionFusion(
            visual_dim=visual_dim,
            metadata_dim=metadata_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.classifier = nn.Linear(metadata_dim, num_classes)

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Args:
            image    : [B, 3, 224, 224]
            metadata : [B, 4]  normalised to [0, 1]
            return_attention: if True, also return attention weights

        Returns:
            logits      : [B, num_classes]
            attn_weights: [B, num_heads, 49, 1]  (only if return_attention=True)
        """
        visual = self.backbone(image)  # [B, 49, 768]
        meta = self.meta_encoder(metadata)  # [B,  1, 256]
        fused, attn_weights = self.fusion(visual, meta)  # [B, 256]
        logits = self.classifier(fused)  # [B, num_classes]

        if return_attention:
            return logits, attn_weights
        return logits
