import torch
import torch.nn as nn

from src.models.backbone.skin_backbone import SwinBackbone
from src.models.encoders.metadata_encoder import MetadataEncoder
from src.models.fusion.cross_attention import CrossAttentionFusion
class SugarcaneClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone =SwinBackbone()
        self.metadata_encoder=MetadataEncoder()
        self.fusion=CrossAttentionFusion()
        self.classifier=nn.Linear(256, num_classes)
    def forward(self,image,metadata):
        visual_features=self.backbone(image)
        meta_tokens=self.metadata_encoder(metadata)
        fused=self.fusion(visual_features, meta_tokens)
        logits=self.classifier(fused)
        return logits


