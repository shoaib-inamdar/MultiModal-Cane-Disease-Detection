"""
SugarcaneAI -- Multi-Model Comparison Demo
Compares all 5 models simultaneously on one uploaded leaf image.

Models:
  1. SugarcaneAI CrossAttn  (best_model.pt)        -- our main contribution
  2. Concat Fusion          (concat_fusion.pt)      -- ablation baseline
  3. Student MobileNetV3    (student_model.pt)      -- knowledge distillation
  4. ResNet50               (resnet50.pt)           -- image-only baseline
  5. EfficientNet-B3        (efficientnet_b3.pt)    -- image-only baseline

Run in Colab:
    !python demo/app.py   (share=True gives a public URL)

Run locally (after placing all 5 .pt files in checkpoints/):
    uv run python demo/app.py
"""

from __future__ import annotations

import time

import gradio as gr
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.evaluation.gradcam import GradCAMPlusPlus, apply_heatmap
from src.evaluation.mc_dropout import mc_predict
from src.models.backbone.swin_backbone import SwinBackbone
from src.models.encoders.metadata_encoder import MetadataEncoder
from src.models.student_classifier import StudentClassifier
from src.models.sugarcane_classifier import SugarcaneClassifier
from src.utils.config import load_config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = "configs/obj1_cross_attention.yaml"
CKPT_DIR = "checkpoints"  # local copies from Drive

CLASS_NAMES = ["Healthy", "Grassy Shoot", "Smut", "Pokkah Boeng"]

DISEASE_INFO = {
    "Healthy": "No infection. Normal green leaf coloration.",
    "Grassy Shoot": "Phytoplasma infection. Pale yellowish leaves, excessive tillers. Spreads via leafhoppers.",
    "Smut": "Sporisorium scitamineum. Black whip-like growth from the growing point. Very distinctive.",
    "Pokkah Boeng": "Fusarium moniliforme. Twisted/stunted top leaves with V-shaped chlorotic lesions.",
}

_META_RANGES = [(15.0, 45.0), (20.0, 100.0), (0.0, 100.0), (0.0, 300.0)]

device = "cuda" if torch.cuda.is_available() else "cpu"
config = load_config(CONFIG_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# Load all 4 models
# ──────────────────────────────────────────────────────────────────────────────


def _load_state(model: nn.Module, path: str) -> nn.Module:
    """Load a checkpoint into model, handling trainer.py's {'model': ...} wrapper."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt  # raw state_dict (e.g. image-only baselines saved directly)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _load_image_only(backbone: nn.Module, path: str) -> "_ImageOnlyWrapper":
    """Load a checkpoint saved from a bare timm model (no wrapper prefix).

    ResNet50 and EfficientNet-B3 were saved as raw state dicts with keys like
    'layer1.0.conv1.weight', not 'net.layer1.0.conv1.weight'.  Loading into the
    wrapper directly causes a key-mismatch error, so we load into the inner
    backbone first and then wrap it.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    backbone.load_state_dict(state)
    backbone.eval()
    wrapper = _ImageOnlyWrapper(backbone)
    return wrapper.to(device)


class _ImageOnlyWrapper(nn.Module):
    """Wrap ResNet/EfficientNet so they accept (image, metadata) like the other models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.net = backbone

    def forward(self, image: torch.Tensor, metadata: torch.Tensor | None = None):
        return self.net(image)


class _ConcatFusionClassifier(nn.Module):
    """Swin-Tiny + MetadataEncoder concatenated (no cross-attention).

    This is the ablation baseline trained in Phase 12 (Cell 5b).
    Architecture: visual mean-pool [B,768] || meta [B,256] → Linear → logits
    Checkpoint: concat_fusion.pt (saved during Phase 12 Colab cell).
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = SwinBackbone()  # → [B, 49, 768]
        self.meta_enc = MetadataEncoder()  # → [B, 1, 256]
        self.classifier = nn.Linear(768 + 256, num_classes)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor):
        visual = self.backbone(image).mean(dim=1)  # [B, 768]
        meta = self.meta_enc(metadata).squeeze(1)  # [B, 256]
        fused = torch.cat([visual, meta], dim=1)  # [B, 1024]
        return self.classifier(fused)


print("Loading models...")
models: dict[str, nn.Module] = {}
model_params: dict[str, int] = {}

# 1. CrossAttn (our main model)
_crossattn = SugarcaneClassifier(num_classes=4)
_crossattn = _load_state(_crossattn, f"{CKPT_DIR}/best_model.pt")
models["SugarcaneAI\n(CrossAttn)"] = _crossattn
model_params["SugarcaneAI\n(CrossAttn)"] = 28_286_014

# 2. Concat Fusion ablation baseline (Phase 12)
_concat = _ConcatFusionClassifier(num_classes=4)
_concat = _load_state(_concat, f"{CKPT_DIR}/concat_fusion.pt")
models["Concat Fusion\n(Swin-T + Meta)"] = _concat
model_params["Concat Fusion\n(Swin-T + Meta)"] = 28_070_148  # Swin-T + MetaEnc + Linear

# 3. Student (knowledge distillation)
_student = StudentClassifier(num_classes=4)
_student = _load_state(_student, f"{CKPT_DIR}/student_model.pt")
models["Student\n(MobileNetV3)"] = _student
model_params["Student\n(MobileNetV3)"] = 1_888_484

# 4. ResNet50 — saved as raw timm state dict (no wrapper prefix)
_resnet = _load_image_only(
    timm.create_model("resnet50", pretrained=False, num_classes=4),
    f"{CKPT_DIR}/resnet50.pt",
)
models["ResNet50\n(Image-only)"] = _resnet
model_params["ResNet50\n(Image-only)"] = 23_516_228

# 5. EfficientNet-B3 — saved as raw timm state dict (no wrapper prefix)
_effnet = _load_image_only(
    timm.create_model("efficientnet_b3", pretrained=False, num_classes=4),
    f"{CKPT_DIR}/efficientnet_b3.pt",
)
models["EfficientNet-B3\n(Image-only)"] = _effnet
model_params["EfficientNet-B3\n(Image-only)"] = 10_702_380

cam_engine = GradCAMPlusPlus(_crossattn)
print(f"All 5 models loaded on {device}.")

# ──────────────────────────────────────────────────────────────────────────────
# Transforms & helpers
# ──────────────────────────────────────────────────────────────────────────────

_IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _build_meta(temperature, humidity, soil_moisture, rainfall) -> torch.Tensor:
    raw = [temperature, humidity, soil_moisture, rainfall]
    norm = [(v - lo) / (hi - lo) for v, (lo, hi) in zip(raw, _META_RANGES)]
    return torch.tensor([norm], dtype=torch.float32)


def _model_card(
    name: str,
    disease: str,
    confidence: float,
    uncertainty: float,
    probs: list[float],
    params: int,
    elapsed_ms: float,
) -> str:
    color = "#22c55e" if disease == "Healthy" else "#ef4444"
    bar_items = "".join(
        f"""<div style='display:flex;align-items:center;gap:6px;margin:3px 0'>
              <span style='width:110px;font-size:12px;color:#94a3b8'>{CLASS_NAMES[i]}</span>
              <div style='flex:1;background:#1e293b;border-radius:4px;height:10px;overflow:hidden'>
                <div style='height:100%;width:{probs[i] * 100:.1f}%;background:{color};
                     border-radius:4px;transition:width 0.5s'></div>
              </div>
              <span style='width:42px;text-align:right;font-size:12px;color:#f1f5f9'>{probs[i] * 100:.1f}%</span>
            </div>"""
        for i in range(4)
    )
    model_short = name.split("\n")[0]
    badge_color = (
        "#6366f1"
        if "CrossAttn" in name
        else (
            "#a855f7"
            if "Concat" in name
            else (
                "#f59e0b"
                if "Student" in name
                else "#3b82f6"
                if "ResNet" in name
                else "#10b981"
            )
        )
    )
    return f"""
    <div style='background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                padding:16px;font-family:Inter,sans-serif;height:100%'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
        <span style='background:{badge_color};color:white;padding:3px 10px;
               border-radius:20px;font-size:12px;font-weight:600'>{model_short}</span>
        <span style='color:#475569;font-size:11px'>{params / 1e6:.1f}M params · {elapsed_ms:.0f}ms</span>
      </div>
      <div style='color:{color};font-size:22px;font-weight:700;margin-bottom:4px'>{disease}</div>
      <div style='color:#94a3b8;font-size:12px;margin-bottom:12px'>{DISEASE_INFO[disease]}</div>
      <div style='display:flex;gap:16px;margin-bottom:12px'>
        <div style='text-align:center'>
          <div style='color:{color};font-size:26px;font-weight:800'>{confidence * 100:.1f}%</div>
          <div style='color:#475569;font-size:11px'>Confidence</div>
        </div>
        <div style='text-align:center'>
          <div style='color:#f59e0b;font-size:26px;font-weight:800'>{uncertainty * 1000:.2f}<span style='font-size:14px'>×10⁻³</span></div>
          <div style='color:#475569;font-size:11px'>Uncertainty</div>
        </div>
      </div>
      <div style='border-top:1px solid #1e293b;padding-top:10px'>{bar_items}</div>
    </div>"""


# ──────────────────────────────────────────────────────────────────────────────
# Main inference function
# ──────────────────────────────────────────────────────────────────────────────


def diagnose_all(image, temperature, humidity, soil_moisture, rainfall):
    if image is None:
        empty = "<div style='color:#94a3b8;text-align:center;padding:40px'>Upload an image and click Diagnose</div>"
        return None, empty, empty, empty, empty, empty

    img_tensor = _IMG_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    meta_tensor = _build_meta(temperature, humidity, soil_moisture, rainfall)

    cards = {}
    for name, mdl in models.items():
        t0 = time.perf_counter()
        result = mc_predict(mdl, img_tensor, meta_tensor, device=device, num_passes=30)
        elapsed = (time.perf_counter() - t0) * 1000

        disease = CLASS_NAMES[result["pred_class"]]
        cards[name] = _model_card(
            name,
            disease,
            result["confidence"],
            result["uncertainty"],
            result["mean_probs"].tolist(),
            model_params[name],
            elapsed,
        )

    # Grad-CAM++ from CrossAttn only
    crossattn_result = mc_predict(
        _crossattn, img_tensor, meta_tensor, device=device, num_passes=1
    )
    heatmap, _ = cam_engine(
        img_tensor, meta_tensor, class_idx=crossattn_result["pred_class"], device=device
    )
    overlay_np = apply_heatmap(img_tensor, heatmap)
    overlay_pil = Image.fromarray(overlay_np)

    keys = list(cards.keys())
    return (
        overlay_pil,
        cards[keys[0]],
        cards[keys[1]],
        cards[keys[2]],
        cards[keys[3]],
        cards[keys[4]],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { background: #020617 !important; font-family: Inter, sans-serif; }
.header-title { font-size: 36px; font-weight: 800; background: linear-gradient(135deg, #6366f1, #22d3ee);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header-sub { color: #64748b; font-size: 14px; margin-top: 4px; }
button.lg { background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
  border: none !important; color: white !important; font-weight: 700 !important;
  border-radius: 8px !important; padding: 12px 28px !important;
  transition: transform 0.1s, box-shadow 0.2s !important; }
button.lg:hover { transform: translateY(-1px); box-shadow: 0 8px 20px #6366f155 !important; }
.model-grid .prose { background: transparent !important; }
"""

# Animated neural network SVG shown as "processing" placeholder
_SPINNER_SVG = """
<div id='spinner' style='display:none;text-align:center;padding:30px'>
  <svg width='200' height='120' viewBox='0 0 200 120'>
    <!-- Input node -->
    <circle cx='20' cy='60' r='8' fill='#6366f1' opacity='0.9'><animate attributeName='opacity' values='0.3;1;0.3' dur='1.2s' repeatCount='indefinite'/></circle>
    <!-- Hidden layer 1 -->
    <circle cx='80' cy='20' r='6' fill='#7c3aed'><animate attributeName='opacity' values='0.3;1;0.3' dur='1.2s' begin='0.2s' repeatCount='indefinite'/></circle>
    <circle cx='80' cy='60' r='6' fill='#7c3aed'><animate attributeName='opacity' values='0.3;1;0.3' dur='1.2s' begin='0.4s' repeatCount='indefinite'/></circle>
    <circle cx='80' cy='100' r='6' fill='#7c3aed'><animate attributeName='opacity' values='0.3;1;0.3' dur='1.2s' begin='0.6s' repeatCount='indefinite'/></circle>
    <!-- Hidden layer 2 -->
    <circle cx='140' cy='40' r='6' fill='#22d3ee'><animate attributeName='opacity' values='0.3;1;0.3' dur='1.2s' begin='0.8s' repeatCount='indefinite'/></circle>
    <circle cx='140' cy='80' r='6' fill='#22d3ee'><animate attributeName='opacity' values='0.3;1;0.3' dur='1.2s' begin='1.0s' repeatCount='indefinite'/></circle>
    <!-- Output -->
    <circle cx='190' cy='60' r='8' fill='#22c55e'><animate attributeName='opacity' values='0.3;1;0.3' dur='1.2s' begin='1.2s' repeatCount='indefinite'/></circle>
    <!-- Edges (simplified) -->
    <line x1='28' y1='60' x2='74' y2='20' stroke='#6366f133' stroke-width='1.5'/>
    <line x1='28' y1='60' x2='74' y2='60' stroke='#6366f133' stroke-width='1.5'/>
    <line x1='28' y1='60' x2='74' y2='100' stroke='#6366f133' stroke-width='1.5'/>
    <line x1='86' y1='20' x2='134' y2='40' stroke='#7c3aed33' stroke-width='1'/>
    <line x1='86' y1='60' x2='134' y2='40' stroke='#7c3aed33' stroke-width='1'/>
    <line x1='86' y1='60' x2='134' y2='80' stroke='#7c3aed33' stroke-width='1'/>
    <line x1='86' y1='100' x2='134' y2='80' stroke='#7c3aed33' stroke-width='1'/>
    <line x1='146' y1='40' x2='182' y2='60' stroke='#22d3ee33' stroke-width='1.5'/>
    <line x1='146' y1='80' x2='182' y2='60' stroke='#22d3ee33' stroke-width='1.5'/>
  </svg>
  <p style='color:#64748b;font-size:13px;margin-top:8px'>Neural networks analyzing...</p>
</div>
<script>
function showSpinner() { document.getElementById('spinner').style.display='block'; }
</script>
"""

HEADER_HTML = """
<div style='text-align:center;padding:24px 0 16px'>
  <div class='header-title'>🌾 SugarcaneAI — Disease Comparison Lab</div>
  <div class='header-sub'>Upload one leaf image → all 5 models diagnose simultaneously → compare results</div>
  <div style='display:flex;justify-content:center;gap:12px;margin-top:12px;flex-wrap:wrap'>
    <span style='background:#6366f122;border:1px solid #6366f155;color:#818cf8;padding:4px 12px;border-radius:20px;font-size:12px'>🔬 CrossAttn SugarcaneAI</span>
    <span style='background:#a855f722;border:1px solid #a855f755;color:#d8b4fe;padding:4px 12px;border-radius:20px;font-size:12px'>🔗 Concat Fusion (ablation)</span>
    <span style='background:#f59e0b22;border:1px solid #f59e0b55;color:#fcd34d;padding:4px 12px;border-radius:20px;font-size:12px'>⚡ Student MobileNetV3</span>
    <span style='background:#3b82f622;border:1px solid #3b82f655;color:#93c5fd;padding:4px 12px;border-radius:20px;font-size:12px'>🏛️ ResNet50</span>
    <span style='background:#10b98122;border:1px solid #10b98155;color:#6ee7b7;padding:4px 12px;border-radius:20px;font-size:12px'>🔋 EfficientNet-B3</span>
  </div>
</div>
"""

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="SugarcaneAI") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── Left: Input panel ─────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### 📷 Input")
            img_input = gr.Image(type="pil", label="Leaf Image", height=220)
            gr.Markdown("### 🌡️ Environmental Conditions")
            temp_sl = gr.Slider(15, 45, value=28, step=0.5, label="Temperature (°C)")
            hum_sl = gr.Slider(20, 100, value=65, step=1, label="Humidity (%)")
            soil_sl = gr.Slider(0, 100, value=50, step=1, label="Soil Moisture (%)")
            rain_sl = gr.Slider(0, 300, value=80, step=1, label="Rainfall (mm/day)")
            btn = gr.Button(
                "🔬 Diagnose with All 5 Models", variant="primary", size="lg"
            )
            cam_out = gr.Image(label="Grad-CAM++ (CrossAttn model)", height=200)

        # ── Right: Model comparison grid ──────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Model Comparison")
            with gr.Row(elem_classes=["model-grid"]):
                out1 = gr.HTML()  # CrossAttn
                out2 = gr.HTML()  # Concat Fusion
            with gr.Row(elem_classes=["model-grid"]):
                out3 = gr.HTML()  # Student
                out4 = gr.HTML()  # ResNet50
            with gr.Row(elem_classes=["model-grid"]):
                out5 = gr.HTML()  # EfficientNet-B3

    gr.HTML("""
    <div style='background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                padding:14px 20px;margin-top:16px;font-size:13px;color:#64748b'>
      <b style='color:#94a3b8'>How to read this:</b>
      Confidence = probability of the top prediction (higher = more certain).
      Uncertainty = MC-Dropout variance (lower = more reliable).
      Grad-CAM shows which leaf regions CrossAttn focused on (red = high attention).
    </div>
    """)

    btn.click(
        diagnose_all,
        inputs=[img_input, temp_sl, hum_sl, soil_sl, rain_sl],
        outputs=[cam_out, out1, out2, out3, out4, out5],
    )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
