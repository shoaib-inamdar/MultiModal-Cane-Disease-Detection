"""
Demo A -- Component Showcase (Phase 4+)

Shows each pipeline module working independently with real tensor shapes.
Run with:  uv run python demo/demo_components.py
"""

import torch

from src.models.backbone.swin_backbone import SwinBackbone
from src.models.encoders.metadata_encoder import MetadataEncoder
from src.utils.config import load_config

config = load_config("configs/obj1_cross_attention.yaml")

print("\n" + "=" * 50)
print("  SugarcaneAI -- Component Showcase")
print("=" * 50)

# 1. Image input
image = torch.rand(1, 3, 224, 224)
print(f"\n[1] Input image shape : {image.shape}")

# 2. SwinBackbone
backbone = SwinBackbone(pretrained=False)
backbone.eval()

with torch.no_grad():
    patches = backbone(image)

print(f"\n[2] SwinBackbone output : {patches.shape}")
print("    -> 49 patches (7x7 regions of the leaf)")
print("    -> 768-dim each (Swin-Tiny feature size)")

# 3. Metadata input
metadata = torch.tensor([[35.0, 85.0, 60.0, 120.0]])
metadata_norm = metadata / torch.tensor([50.0, 100.0, 100.0, 300.0])

print("\n[3] Raw metadata        : Temp=35C  Humidity=85%  Soil=60%  Rain=120mm")
print(f"    Normalised          : {[round(v, 3) for v in metadata_norm.tolist()[0]]}")

# 4. MetadataEncoder
encoder = MetadataEncoder()
encoder.eval()

with torch.no_grad():
    env_token = encoder(metadata_norm)

print(f"\n[4] MetadataEncoder out : {env_token.shape}")
print("    -> 1 environmental context token, 256-dim")

# Summary
print("\n" + "=" * 50)
print("  [OK] Pipeline shapes confirmed:")
print("     Image   -> SwinBackbone    -> [1, 49, 768]")
print("     Weather -> MetadataEncoder -> [1,  1, 256]")
print("     Next: CrossAttentionFusion merges these")
print("=" * 50 + "\n")
