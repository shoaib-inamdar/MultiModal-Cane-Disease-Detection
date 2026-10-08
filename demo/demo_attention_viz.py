"""
Demo C -- Attention Heatmap Visualisation (Phase 5)

Shows WHICH regions of the leaf the model focuses on under different
environmental conditions. This is the publishable research visual.

Run with:  uv run python demo/demo_attention_viz.py

Output: demo/attention_heatmap.png
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # non-interactive backend (works without display)
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from src.models.sugarcane_classifier import SugarcaneClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed

# ── Setup ─────────────────────────────────────────────────────────────────────
set_seed(42)
config = load_config("configs/obj1_cross_attention.yaml")
model = SugarcaneClassifier(config)
model.eval()

CLASS_NAMES = ["Healthy", "Red Rot", "Grassy Shoot", "Smut"]

# ── Two contrasting weather scenarios ────────────────────────────────────────
SCENARIOS = [
    {
        "label": "Hot & Humid (Red Rot risk)",
        "raw": [38.0, 90.0, 70.0, 150.0],
        "color": "#e74c3c",
    },
    {
        "label": "Cool & Dry (Low disease risk)",
        "raw": [22.0, 35.0, 25.0, 20.0],
        "color": "#2ecc71",
    },
]
NORM_FACTORS = torch.tensor([50.0, 100.0, 100.0, 300.0])

# Same image for both scenarios -- isolates the weather effect
torch.manual_seed(42)
image = torch.rand(1, 3, 224, 224)


# ── Helper ───────────────────────────────────────────────────────────────────
def get_attention_map(scenario: dict):
    raw = torch.tensor([scenario["raw"]])
    metadata = raw / NORM_FACTORS

    with torch.no_grad():
        logits, attn_weights = model(image, metadata, return_attention=True)
        probs = torch.softmax(logits, dim=-1)[0]

    # attn_weights: [B, 49, 1]  (PyTorch averages over heads by default)
    # Squeeze key dimension -> [49], then reshape to 7x7 spatial grid
    heatmap = attn_weights[0].squeeze(-1)  # [49]
    heatmap = heatmap.reshape(7, 7).cpu().numpy()  # [7, 7]

    # Normalise to [0, 1]
    lo, hi = heatmap.min(), heatmap.max()
    heatmap = (heatmap - lo) / (hi - lo + 1e-8)

    return heatmap, probs.cpu().numpy()


# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
fig.suptitle(
    "SugarcaneAI -- Cross-Attention Heatmap\n"
    "Same leaf, different weather -> different attention focus",
    color="white",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

img_display = image[0].permute(1, 2, 0).numpy()
img_display = (img_display - img_display.min()) / (
    img_display.max() - img_display.min()
)

for col, scenario in enumerate(SCENARIOS):
    heatmap, probs = get_attention_map(scenario)
    pred_idx = probs.argmax()

    # Row 0: original leaf
    ax_img = fig.add_subplot(gs[0, col])
    ax_img.imshow(img_display)
    ax_img.set_title(scenario["label"], color="white", fontsize=9, pad=6)
    ax_img.axis("off")

    # Row 1: heatmap overlay
    ax_heat = fig.add_subplot(gs[1, col])
    ax_heat.imshow(img_display)
    im = ax_heat.imshow(
        heatmap,
        alpha=0.65,
        cmap="hot",
        extent=[0, 224, 224, 0],
        aspect="auto",
    )
    raw = scenario["raw"]
    ax_heat.set_title(
        f"Attention | Pred: {CLASS_NAMES[pred_idx]} ({probs[pred_idx] * 100:.0f}%)\n"
        f"T={raw[0]}C  RH={raw[1]}%  Soil={raw[2]}%  Rain={raw[3]}mm",
        color="white",
        fontsize=8,
        pad=6,
    )
    ax_heat.axis("off")
    cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

# Column 2: probability bar chart
ax_bar = fig.add_subplot(gs[:, 2])
ax_bar.set_facecolor("#16213e")
x = np.arange(len(CLASS_NAMES))
width = 0.35

for i, scenario in enumerate(SCENARIOS):
    _, probs = get_attention_map(scenario)
    bars = ax_bar.bar(
        x + i * width - width / 2,
        probs * 100,
        width,
        label=scenario["label"],
        color=scenario["color"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, p in zip(bars, probs):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{p * 100:.0f}%",
            ha="center",
            va="bottom",
            color="white",
            fontsize=7,
        )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(CLASS_NAMES, color="white", fontsize=9, rotation=15)
ax_bar.set_ylabel("Probability (%)", color="white")
ax_bar.set_title("Disease Probabilities\nper Scenario", color="white", fontsize=10)
ax_bar.tick_params(colors="white")
for spine in ax_bar.spines.values():
    spine.set_color("#444")
ax_bar.set_ylim(0, 60)
ax_bar.legend(
    fontsize=7,
    facecolor="#16213e",
    labelcolor="white",
    loc="upper right",
    framealpha=0.5,
)

# Footer
fig.text(
    0.5,
    0.005,
    "Note: model uses random weights (untrained) -- predictions are illustrative only.",
    ha="center",
    color="#aaa",
    fontsize=8,
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "demo/attention_heatmap.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[OK] Heatmap saved -> {out_path}")
print("     Open demo/attention_heatmap.png to view the result.\n")
