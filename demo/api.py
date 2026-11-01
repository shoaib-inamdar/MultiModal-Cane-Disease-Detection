"""
SugarcaneAI -- FastAPI REST endpoint (optional)
Run: uv run uvicorn demo.api:app --reload
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image as PILImage

from demo.app import CLASS_NAMES, num_classes, run_inference

app = FastAPI(title="SugarcaneAI API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/diagnose")
async def diagnose_api(
    file: UploadFile = File(...),
    temperature: float = Form(28.0),
    humidity: float = Form(65.0),
    soil_moisture: float = Form(50.0),
    rainfall: float = Form(80.0),
) -> JSONResponse:
    """
    Diagnose a sugarcane leaf disease.

    Form fields:
        file          : image file (JPEG/PNG)
        temperature   : float, degrees C
        humidity      : float, %
        soil_moisture : float, %
        rainfall      : float, mm
    """
    img_bytes = await file.read()
    pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

    result = run_inference(pil_img, temperature, humidity, soil_moisture, rainfall)

    return JSONResponse(
        {
            "predicted_class": CLASS_NAMES[result["pred_class"]],
            "confidence": round(result["confidence"], 4),
            "uncertainty": round(result["uncertainty"], 6),
            "class_probabilities": {
                CLASS_NAMES[i]: round(float(result["mean_probs"][i]), 4)
                for i in range(num_classes)
            },
        }
    )
