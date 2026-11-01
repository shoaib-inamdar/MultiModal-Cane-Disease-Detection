"""
SugarcaneAI -- Gradio Demo
Run: uv run python demo/app.py
"""

from __future__ import annotations

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from src.evaluation.gradcam import GradCAMPlusPlus, apply_heatmap
from src.evaluation.mc_dropout import mc_predict
from src.models.sugarcane_classifier import SugarcaneClassifier
from src.utils.config import load_config

CONFIG_PATH = "configs/obj1_cross_attention.yaml"
CHECKPOINT_PATH = "checkpoints/best_model.pt"

CLASS_NAMES = ["Healthy", "Grassy Shoot", "Smut", "Pokkah Boeng"]
DISEASE_INFO = {
    "Healthy": "No infection detected. The leaf shows normal green coloration.",
    "Grassy Shoot": "Phytoplasma infection. Pale yellowish leaves, excessive thin tillers.",
    "Smut": "Sporisorium scitamineum. Black whip-like growth from the growing point.",
    "Pokkah Boeng": "Fusarium moniliforme. Twisted/stunted top leaves with V-shaped lesions.",
}


_META_RANGES = [(15.0, 45.0), (20.0, 100.0), (0.0, 100.0), (0.0, 300.0)]

device = "cuda" if torch.cuda.is_available() else "cpu"

config = load_config(CONFIG_PATH)
num_classes = config["data"]["num_classes"]

model = SugarcaneClassifier(num_classes=num_classes)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

cam_engine = GradCAMPlusPlus(model)

IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




def build_meta_tensor(
    temperature: float, humidity: float, soil_moisture: float, rainfall: float
) -> torch.Tensor:
    """Normalise raw metadata values to [0, 1] matching the training distribution."""
    raw = [temperature, humidity, soil_moisture, rainfall]
    normalised = [(v - lo) / (hi - lo) for v, (lo, hi) in zip(raw, _META_RANGES)]
    return torch.tensor([normalised], dtype=torch.float32)


def run_inference(
    image: Image.Image,
    temperature: float,
    humidity: float,
    soil_moisture: float,
    rainfall: float,
) -> dict:
    """Core inference: MC-Dropout prediction + Grad-CAM++ heatmap."""
    img_tensor = IMG_TRANSFORM(image.convert("RGB")).unsqueeze(0)  # [1,3,224,224]
    meta_tensor = build_meta_tensor(temperature, humidity, soil_moisture, rainfall)

    result = mc_predict(model, img_tensor, meta_tensor, device=device, num_passes=30)

    heatmap, _ = cam_engine(
        img_tensor, meta_tensor, class_idx=result["pred_class"], device=device
    )
    overlay_np = apply_heatmap(img_tensor, heatmap)  # [224,224,3] uint8
    result["overlay"] = Image.fromarray(overlay_np)
    return result




def diagnose(image, temperature, humidity, soil_moisture, rainfall):
    if image is None:
        return None, "<p>Please upload a leaf image.</p>", {}

    result = run_inference(image, temperature, humidity, soil_moisture, rainfall)

    pred_cls = result["pred_class"]
    confidence = result["confidence"]
    uncertainty = result["uncertainty"]
    mean_probs = result["mean_probs"]
    disease = CLASS_NAMES[pred_cls]
    color = "#22c55e" if disease == "Healthy" else "#ef4444"

    label_html = f"""
    <div style='padding:14px;border-radius:10px;background:#1e293b;color:#f1f5f9;font-family:sans-serif'>
      <h2 style='margin:0 0 6px 0;color:{color}'>{disease}</h2>
      <p style='margin:0 0 10px 0;font-size:14px;color:#94a3b8'>{DISEASE_INFO[disease]}</p>
      <div style='display:flex;gap:24px'>
        <div><b style='font-size:22px;color:{color}'>{confidence * 100:.1f}%</b><br>
             <span style='font-size:12px;color:#64748b'>Confidence</span></div>
        <div><b style='font-size:22px;color:#f59e0b'>{uncertainty * 1000:.3f}x10^-3</b><br>
             <span style='font-size:12px;color:#64748b'>Uncertainty</span></div>
      </div>
      <p style='margin:8px 0 0 0;font-size:12px;color:#475569'>
        Temp {temperature}C | Humidity {humidity}% | Soil {soil_moisture}% | Rain {rainfall}mm
      </p>
    </div>"""

    class_probs = {CLASS_NAMES[i]: float(mean_probs[i]) for i in range(num_classes)}
    return result["overlay"], label_html, class_probs


with gr.Blocks(theme=gr.themes.Soft(), title="SugarcaneAI") as demo:
    gr.Markdown("# SugarcaneAI -- Disease Diagnosis")
    gr.Markdown("Upload a sugarcane leaf photo and set the environmental conditions.")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Leaf Image")
            temp_sl = gr.Slider(15, 45, value=28, step=0.5, label="Temperature (C)")
            hum_sl = gr.Slider(20, 100, value=65, step=1, label="Humidity (%)")
            soil_sl = gr.Slider(0, 100, value=50, step=1, label="Soil Moisture (%)")
            rain_sl = gr.Slider(0, 300, value=80, step=1, label="Rainfall (mm)")
            btn = gr.Button("Diagnose", variant="primary")

        with gr.Column(scale=1):
            cam_out = gr.Image(label="Grad-CAM++ (Red = High Attention)")
            html_out = gr.HTML()
            lbl_out = gr.Label(num_top_classes=4, label="Class Probabilities")

    btn.click(
        diagnose,
        inputs=[img_input, temp_sl, hum_sl, soil_sl, rain_sl],
        outputs=[cam_out, html_out, lbl_out],
    )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
