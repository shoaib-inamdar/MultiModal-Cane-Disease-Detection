import timm
import torch
import torch.nn as nn


class SwinBackbone(nn.Module):
    """Wraps a timm Swin Transformer as the visual encoder.

    Produces per-patch feature tokens (not pooled class logits) so that
    downstream Cross-Attention fusion can attend over individual image
    regions rather than a single global embedding.
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        self.embed_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalised RGB image tensor, shape [B, 3, 224, 224].

        Returns:
            Patch tokens, shape [B, 49, 768] for Swin-Tiny (7x7 patches).
        """
        tokens = self.model(x)

        # Some timm versions return Swin's feature map as a spatial grid
        # [B, H, W, C] instead of pre-flattened tokens [B, H*W, C]. Cross-
        # Attention (Phase 5) needs a flat token sequence, so normalise here.
        if tokens.dim() == 4:
            batch, height, width, channels = tokens.shape
            tokens = tokens.reshape(batch, height * width, channels)

        return tokens
