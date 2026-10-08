"""
Demo B -- Full Forward Pass (Phase 6+)

Shows the complete SugarcaneClassifier processing a leaf image + weather
data in a single forward pass and printing disease probabilities.

Run with:  uv run python demo/demo_full_model.py
"""

import torch

from src.models.sugarcane_classifier import SugarcaneClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Setup ─────────────────────────────────────────────
set_seed(42)
config = load_config("configs/obj1_cross_attention.yaml")
model = SugarcaneClassifier(config)
model.eval()

CLASS_NAMES = ["Healthy", "Red Rot", "Grassy Shoot", "Smut"]

# ── Simulate field input ──────────────────────────────
image = torch.rand(1, 3, 224, 224)

metadata = torch.tensor([[35.0, 85.0, 60.0, 120.0]])
metadata_norm = metadata / torch.tensor([50.0, 100.0, 100.0, 300.0])

# ── Forward pass ──────────────────────────────────────
with torch.no_grad():
    logits = model(image, metadata_norm)
    probs = torch.softmax(logits, dim=-1)

pred_idx = probs.argmax(dim=-1).item()
confidence = probs[0][pred_idx].item()

# ── Display ───────────────────────────────────────────
print("\n=== SugarcaneAI Disease Prediction ===")
print("Environmental context: Temp=35C, Humidity=85%, Soil=60%, Rain=120mm")
print()

for name, prob in zip(CLASS_NAMES, probs[0]):
    bar = "#" * int(prob.item() * 20)
    print(f"  {name:<15} {bar:<20} {prob.item() * 100:.1f}%")

print(f"\n-> Predicted: {CLASS_NAMES[pred_idx]} ({confidence * 100:.1f}% confidence)")
print("(Note: model uses random weights -- real predictions need training)\n")
