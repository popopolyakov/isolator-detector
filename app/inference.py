from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES: list[dict[str, Any]] = [
    {"id": 0, "key": "vibration_damper",    "ru": "Гаситель вибрации",            "is_violation": False, "color": (225, 96, 55)},
    {"id": 1, "key": "festoon_insulators",  "ru": "Гирлянда изоляторов (стекло)",  "is_violation": False, "color": (102, 255, 102)},
    {"id": 2, "key": "polymer_insulators",  "ru": "Гирлянда полимерных изоляторов","is_violation": False, "color": (204, 51, 102)},
    {"id": 3, "key": "traverse",            "ru": "Траверса опоры",                "is_violation": False, "color": (158, 217, 7)},
    {"id": 4, "key": "nest",                "ru": "Гнездо на траверсе",            "is_violation": True,  "color": (131, 224, 112)},
    {"id": 5, "key": "bad_insulator",       "ru": "Изолятор отсутствует",          "is_violation": True,  "color": (93, 87, 117)},
    {"id": 6, "key": "damaged_insulator",   "ru": "Поврежденный изолятор",         "is_violation": True,  "color": (49, 147, 245)},
    {"id": 7, "key": "safety_sign+",        "ru": "Табличка / знак",               "is_violation": False, "color": (204, 51, 102)},
]
CLASS_BY_ID = {c["id"]: c for c in CLASSES}

MODELS: dict[str, dict[str, Any]] = {
    "fast": {
        "id": "fast",
        "label": "YOLOv8n-OBB — быстрая",
        "description": "Nano OBB-модель. Низкие задержки, подходит для real-time и слабых машин. Точность ниже.",
        "size_mb_approx": 7,
        "weights": Path("app/models/fast_yolov8n-obb.pt"),
        "fallback": "yolov8n-obb.pt",
    },
    "accurate": {
        "id": "accurate",
        "label": "YOLO11l-OBB — точная",
        "description": "Large OBB-модель (YOLO11). Самая высокая точность в нашей линейке. Медленнее в 4–5×.",
        "size_mb_approx": 53,
        "weights": Path("app/models/accurate_yolo11l-obb.pt"),
        "fallback": "yolo11l-obb.pt",
    },
}


@dataclass
class Detection:
    class_id: int
    class_key: str
    class_ru: str
    is_violation: bool
    confidence: float
    bbox_xyxy: list[float]    # AABB-обёртка OBB в пикселях (4 числа) — для UI-таблицы
    bbox_points: list[float]  # 4 вершины OBB в пикселях (8 чисел: x1 y1 … x4 y4)
    rotation_deg: float       # угол OBB в градусах (для отображения)


def _model_source(entry: dict[str, Any]) -> str:
    p: Path = entry["weights"]
    return str(p) if p.exists() else entry["fallback"]


class Detector:
    """Lazy-loaded YOLO wrapper. One instance per process per model id."""

    _cache: dict[str, "Detector"] = {}

    def __init__(self, model_id: str):
        if model_id not in MODELS:
            raise KeyError(f"Unknown model id: {model_id}. Available: {list(MODELS)}")
        self.entry = MODELS[model_id]
        from ultralytics import YOLO
        source = _model_source(self.entry)
        self.using_local_weights = self.entry["weights"].exists()
        self.model = YOLO(source)
        self.model_id = model_id

    @classmethod
    def get(cls, model_id: str) -> "Detector":
        if model_id not in cls._cache:
            cls._cache[model_id] = cls(model_id)
        return cls._cache[model_id]

    def predict(self, image_bytes: bytes, conf: float = 0.25, iou: float = 0.45,
                imgsz: int = 640) -> dict[str, Any]:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(pil_img)
        t0 = time.perf_counter()
        results = self.model.predict(source=arr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        annotated_arr = arr.copy()
        if results:
            r = results[0]
            annotated_arr = r.plot()  # OBB model — рисует повёрнутые рамки; BGR out
            annotated_arr = annotated_arr[..., ::-1]  # BGR → RGB для PIL
            names = r.names
            obb = getattr(r, "obb", None)
            if obb is not None and len(obb) > 0:
                # OBB output: xyxyxyxy — Nx8 (4 угловые точки в пикселях)
                pts = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else obb.xyxyxyxy
                confs = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else obb.conf
                clss = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else obb.cls
                for i in range(len(pts)):
                    cls_idx = int(clss[i])
                    cls_meta = CLASS_BY_ID.get(cls_idx, {"key": names.get(cls_idx, "?"), "ru": "?", "is_violation": False})
                    # pts[i] имеет форму (4, 2) — 4 угла × 2 координаты. flatten → 8 чисел.
                    poly = pts[i].reshape(-1).tolist()
                    xs = poly[0::2]
                    ys = poly[1::2]
                    xyxy = [min(xs), min(ys), max(xs), max(ys)]
                    # Восстанавливаем угол из xywhr: r.obb.xywhr → (cx, cy, w, h, theta_rad)
                    xywhr = obb.xywhr.cpu().numpy() if hasattr(obb.xywhr, "cpu") else obb.xywhr
                    rotation_deg = float(xywhr[i, 4]) * 180.0 / 3.141592653589793
                    detections.append(Detection(
                        class_id=cls_idx,
                        class_key=cls_meta["key"] if isinstance(cls_meta, dict) else cls_meta.key,
                        class_ru=cls_meta["ru"] if isinstance(cls_meta, dict) else cls_meta.ru,
                        is_violation=cls_meta["is_violation"] if isinstance(cls_meta, dict) else cls_meta.is_violation,
                        confidence=float(confs[i]),
                        bbox_xyxy=[round(v, 1) for v in xyxy],
                        bbox_points=[round(v, 1) for v in poly],
                        rotation_deg=round(rotation_deg, 1),
                    ))

        annotated_pil = Image.fromarray(annotated_arr)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG", quality=88)
        annotated_jpeg = buf.getvalue()

        return {
            "model_id": self.model_id,
            "model_label": self.entry["label"],
            "using_local_weights": self.using_local_weights,
            "inference_ms": round(elapsed_ms, 1),
            "image_size": [pil_img.width, pil_img.height],
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_key": d.class_key,
                    "class_ru": d.class_ru,
                    "is_violation": d.is_violation,
                    "confidence": round(d.confidence, 4),
                    "bbox_xyxy": d.bbox_xyxy,
                    "bbox_points": d.bbox_points,
                    "rotation_deg": d.rotation_deg,
                } for d in detections
            ],
            "annotated_jpeg": annotated_jpeg,
        }


def list_models() -> list[dict[str, Any]]:
    out = []
    for mid, entry in MODELS.items():
        out.append({
            "id": mid,
            "label": entry["label"],
            "description": entry["description"],
            "size_mb_approx": entry["size_mb_approx"],
            "weights_available": entry["weights"].exists(),
        })
    return out
