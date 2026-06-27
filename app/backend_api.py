from __future__ import annotations

import base64
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .inference import Detector, list_models

app = FastAPI(
    title="Insulator Defect Detector API",
    description="Детекция дефектов ЛЭП (изоляторы, гасители вибрации, гнёзда, таблички).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "insulator-defect-detector",
        "endpoints": ["/health", "/models", "/predict", "/predict/image", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    return {"models": list_models()}


def _run(model_id: str, data: bytes, conf: float, iou: float) -> dict:
    try:
        detector = Detector.get(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return detector.predict(data, conf=conf, iou=iou)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form("accurate"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    result = _run(model_id, data, conf=conf, iou=iou)
    payload = {k: v for k, v in result.items() if k != "annotated_jpeg"}
    payload["annotated_jpeg_b64"] = base64.b64encode(result["annotated_jpeg"]).decode("ascii")
    return JSONResponse(payload)


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    model_id: str = Form("accurate"),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
) -> Response:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    result = _run(model_id, data, conf=conf, iou=iou)
    return Response(content=result["annotated_jpeg"], media_type="image/jpeg")
