import timm
import torch
import torch.nn as nn

from src.models.encoders.metadata_encoder import MetadataEncoder

# MobileNetV3-Small-100 final feature dim (timm, num_classes=0)
_MOBILENET_V3_SMALL_DIM = 1024


class StudentClassifier(nn.Module):
    """
    Lightweight student model for knowledge distillation.

    Architecture (much cheaper than SugarcaneAI Teacher):
      - Backbone : MobileNetV3-Small (2.5M params, pretrained on ImageNet)
                   vs Swin-Tiny Teacher (28M params) -- ~91% smaller
      - Metadata : Same MetadataEncoder as Teacher (4 -> [B, 1, 256])
      - Fusion   : Simple concat + two-layer MLP  (no cross-attention)
      - Head     : Linear(1280, num_classes)  [1024 visual + 256 meta]
    """

    def __init__(self, num_classes: int = 4, backbone: str = "mobilenetv3_small_100"):
        super().__init__()

        # timm API: num_classes=0 removes the head and returns feature vectors
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        self.meta_enc = MetadataEncoder()  # [B, 4] -> [B, 1, 256]

        self.fusion_head = nn.Sequential(
            nn.Linear(_MOBILENET_V3_SMALL_DIM + 256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image    : [B, 3, 224, 224]
            metadata : [B, 4]

        Returns:
            logits   : [B, num_classes]
        """
        visual = self.backbone(image)  # [B, 576]
        meta = self.meta_enc(metadata).squeeze(1)  # [B, 256]
        fused = torch.cat([visual, meta], dim=1)  # [B, 832]
        return self.fusion_head(fused)  # [B, num_classes]
