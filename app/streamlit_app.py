from __future__ import annotations

import base64
import io
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from app.inference import CLASSES, MODELS, list_models

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
# На Streamlit Cloud bind на 0.0.0.0 снаружи бывает заблокирован, поэтому
# хост по умолчанию 127.0.0.1 — Streamlit UI всё равно ходит на localhost.
BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
_BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))


def _backend_already_up(host: str, port: int) -> bool:
    """Проверка, не запущен ли FastAPI извне (отдельным процессом)."""
    check_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        try:
            sock.connect((check_host, port))
        except OSError:
            return False
        return True


def _wait_for_backend(url: str, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"Backend at {url} did not become ready within {timeout_s}s")


def _spawn_backend_once() -> None:
    """Поднять FastAPI в фоновом потоке, если порт свободен.
    Делается один раз на процесс (через флаг модуля).
    Идемпотентно: повторные вызовы не плодят uvicorn-серверы.
    """
    if getattr(_spawn_backend_once, "_done", False):
        print(f"[streamlit_app] backend already spawned, skipping", flush=True)
        return

    print(f"[streamlit_app] checking if backend up at {BACKEND_HOST}:{_BACKEND_PORT}",
          flush=True)
    if _backend_already_up(BACKEND_HOST, _BACKEND_PORT):
        print("[streamlit_app] backend already up, not starting another", flush=True)
        _spawn_backend_once._done = True  # type: ignore[attr-defined]
        return

    print(f"[streamlit_app] starting uvicorn on {BACKEND_HOST}:{_BACKEND_PORT}",
          flush=True)
    try:
        import uvicorn
        from app.backend_api import app as fastapi_app

        config = uvicorn.Config(
            fastapi_app,
            host=BACKEND_HOST,
            port=_BACKEND_PORT,
            log_level=os.environ.get("BACKEND_LOG_LEVEL", "info"),
            reload=False,
            workers=1,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
        thread.start()
        _wait_for_backend(BACKEND_URL)
        print(f"[streamlit_app] backend ready at {BACKEND_URL}", flush=True)
    except Exception as exc:
        print(f"[streamlit_app] backend start FAILED: {type(exc).__name__}: {exc}",
              flush=True)
        raise
    finally:
        _spawn_backend_once._done = True  # type: ignore[attr-defined]


# Поднимаем бэкенд сразу при импорте — Streamlit Cloud делает
# `streamlit run app/streamlit_app.py`, поэтому до любого rerun это безопасно.
try:
    _spawn_backend_once()
except Exception as exc:
    print(f"[streamlit_app] _spawn_backend_once raised: {exc}", flush=True)

st.set_page_config(
    page_title="Детектор дефектов ЛЭП",
    page_icon="⚡",
    layout="wide",
)

# ----------------------------- Header / Sidebar --------------------------------

st.title("⚡ Детектор дефектов ЛЭП")
st.markdown(
    "Веб-приложение для автоматического обнаружения элементов и повреждений "
    "на воздушных линиях электропередачи: **изоляторы**, **гасители вибрации**, "
    "**траверсы**, **гнёзда птиц**, **диспетчерские таблички**."
)

with st.sidebar:
    st.header("⚙️ Параметры")

    # ---- model selector (this is the "не менее 2 моделей" feature) ----
    available = list_models()
    by_id = {m["id"]: m for m in available}

    choice = st.radio(
        "Модель",
        options=list(by_id.keys()),
        format_func=lambda mid: f"{by_id[mid]['label']}\n"
                               f"   ~{by_id[mid]['size_mb_approx']} МБ • "
                               f"{'локальные веса' if by_id[mid]['weights_available'] else 'fallback на претрен.'}",
        help=(
            "**Быстрая** — YOLOv8n-OBB: запускается на слабых машинах, ниже точность.\n\n"
            "**Точная** — YOLO11l-OBB: основной выбор, выше mAP, но требует больше ресурсов.\n\n"
            "Обе модели — OBB (Oriented Bounding Box): предсказывают повёрнутые "
            "рамки, что критично для диагональных гирлянд изоляторов."
        ),
    )
    st.caption(by_id[choice]["description"])
    if not by_id[choice]["weights_available"]:
        st.warning(
            "Локально обученные веса не найдены — приложение работает на "
            "претренированных YOLOv8 из ultralytics. Чтобы получить "
            "детекцию наших классов, сначала запустите `notebooks/train.ipynb`.",
            icon="ℹ️",
        )

    st.divider()

    conf = st.slider("Порог уверенности (conf)", 0.05, 0.95, 0.25, 0.05,
                     help="Отсекает детекции ниже этой вероятности.")
    iou = st.slider("IoU (NMS)", 0.05, 0.95, 0.45, 0.05,
                    help="Порог для non-maximum suppression.")
    imgsz = st.select_slider("Размер входа", options=[320, 480, 640, 800, 960],
                             value=960,
                             help="Больше = точнее на мелких объектах, но медленнее. "
                                  "Модели обучены на 960.")

    st.divider()
    st.markdown("**Классы детекции**")
    cls_df = pd.DataFrame([{
        "ID": c["id"],
        "Класс": c["ru"],
        "Нарушение": "⚠️" if c["is_violation"] else "",
    } for c in CLASSES])
    st.dataframe(cls_df, hide_index=True, use_container_width=True)


# ----------------------------- Main area --------------------------------------

uploaded = st.file_uploader(
    "Загрузите изображение с ЛЭП",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Поддерживаются jpg/png. Для дрон-съёмки типично 4K — приложение само уменьшит до imgsz.",
)

# Demo gallery: pick a sample from the local val set if the user hasn't uploaded
SAMPLES_DIR = Path("data/samples")
sample_files: list[Path] = []
if SAMPLES_DIR.exists():
    sample_files = sorted(SAMPLES_DIR.iterdir())[:8]
if sample_files and not uploaded:
    st.markdown("##### или выберите пример из валидационного набора")
    cols = st.columns(4)
    picked: Path | None = None
    for i, p in enumerate(sample_files):
        with cols[i % 4]:
            try:
                thumb = Image.open(p)
                thumb.thumbnail((220, 220))
                if st.button(f"📷 {p.stem[:24]}", key=f"sample_{i}", use_container_width=True):
                    picked = p
                st.image(thumb, use_container_width=True)
            except Exception:
                pass
    if picked is not None:
        # BufferedReader из Path.open("rb") read-only по .name, поэтому заворачиваем
        # в BytesIO — у него есть и .read(), и .seek(), и атрибут .name можно ставить.
        uploaded = io.BytesIO(picked.read_bytes())


if uploaded is not None:
    file_bytes = uploaded.read() if hasattr(uploaded, "read") else uploaded
    # Reset pointer for downstream reads
    if hasattr(uploaded, "seek"):
        try:
            uploaded.seek(0)
        except Exception:
            pass

    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    st.markdown("### Результаты")

    col1, col2 = st.columns(2)

    with st.spinner(f"Запускаю модель «{by_id[choice]['label']}»…"):
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{BACKEND_URL}/predict",
                files={"file": ("image.jpg", file_bytes, "image/jpeg")},  # type: ignore[arg-type]
                data={"model_id": choice, "conf": conf, "iou": iou, "imgsz": imgsz},
                timeout=60,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            body: dict[str, Any] = resp.json()
            annotated = base64.b64decode(body["annotated_jpeg_b64"])
            result = {
                "detections": body["detections"],
                "inference_ms": body["inference_ms"],
                "image_size": body["image_size"],
                "annotated_jpeg": annotated,
            }
            http_meta = {
                "url": f"{BACKEND_URL}/predict",
                "status": resp.status_code,
                "elapsed_ms": elapsed_ms,
                "model_label": body.get("model_label", ""),
                "using_local_weights": body.get("using_local_weights", False),
            }
        except requests.exceptions.ConnectionError as exc:
            st.error(
                f"Не удалось подключиться к бэкенду по адресу {BACKEND_URL}/predict. "
                f"Запустите `python -m app.serve` или `uvicorn app.backend_api:app`.\n\n"
                f"Ошибка: {exc}"
            )
            st.stop()
        except Exception as exc:  # pragma: no cover
            st.error(f"Ошибка инференса: {exc}")
            st.stop()

    # Индикатор «запрос ушёл через HTTP» — рядом с результатами.
    weights_tag = "локальные веса" if http_meta["using_local_weights"] else "fallback на претрен."
    st.caption(
        f"POST /predict → `{http_meta['url']}` · "
        f"✓ {http_meta['status']} · {http_meta['elapsed_ms']:.0f} мс · "
        f"{http_meta['model_label']} · {weights_tag}"
    )

    with col1:
        st.markdown("**Исходное изображение**")
        st.image(pil_img, use_container_width=True)

    with col2:
        st.markdown("**Детекции**")
        st.image(result["annotated_jpeg"], use_container_width=True)

    # ---- summary metrics ----
    n = len(result["detections"])
    violations = sum(1 for d in result["detections"] if d["is_violation"])
    by_class: dict[str, int] = {}
    for d in result["detections"]:
        by_class[d["class_ru"]] = by_class.get(d["class_ru"], 0) + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Всего объектов", n)
    m2.metric("Нарушений", violations,
              help="Классы с флагом is_violation: гнездо, отсутствующий/повреждённый изолятор.")
    m3.metric("Время инференса, мс", f"{result['inference_ms']:.0f}")
    m4.metric("Размер изображения", f"{pil_img.width}×{pil_img.height}")

    if n == 0:
        st.info("На изображении не найдено объектов с заданным порогом. "
                "Попробуйте снизить conf.", icon="🔍")
    else:
        st.markdown("#### Таблица детекций")
        df = pd.DataFrame([{
            "Класс (рус.)": d["class_ru"],
            "Класс (англ.)": d["class_key"],
            "Уверенность": d["confidence"],
            "Нарушение": "⚠️ да" if d["is_violation"] else "нет",
            "Угол, °": d.get("rotation_deg", 0.0),
            "Центр (x, y)": f"({(d['bbox_xyxy'][0] + d['bbox_xyxy'][2]) / 2:.0f}, "
                            f"{(d['bbox_xyxy'][1] + d['bbox_xyxy'][3]) / 2:.0f})",
        } for d in result["detections"]])
        df = df.sort_values("Уверенность", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("📐 4 вершины OBB (детально)"):
            for d in result["detections"]:
                p = d["bbox_points"]
                st.markdown(
                    f"**{d['class_ru']}** ({d['confidence']:.2f}, "
                    f"угол {d.get('rotation_deg', 0.0):.1f}°):  "
                    f"`({p[0]:.0f},{p[1]:.0f}) ({p[2]:.0f},{p[3]:.0f}) "
                    f"({p[4]:.0f},{p[5]:.0f}) ({p[6]:.0f},{p[7]:.0f})`"
                )

        st.markdown("#### Распределение по классам")
        st.bar_chart(pd.DataFrame(
            {"Количество": list(by_class.values())},
            index=list(by_class.keys()),
        ))

        # ---- download annotated image ----
        st.download_button(
            "⬇️ Скачать размеченное изображение (JPEG)",
            data=result["annotated_jpeg"],
            file_name="annotated.jpg",
            mime="image/jpeg",
        )

    # ---- raw JSON (for debugging / API users) ----
    with st.expander("🔧 Сырой JSON ответа (как у FastAPI /predict)"):
        debug = {k: v for k, v in result.items() if k != "annotated_jpeg"}
        debug["annotated_jpeg_b64_len"] = len(result["annotated_jpeg"])
        st.code(json.dumps(debug, ensure_ascii=False, indent=2), language="json")

else:
    st.info("⬆️ Загрузите изображение, чтобы начать детекцию.", icon="📤")

    with st.expander("ℹ️ О моделях и обучении"):
        st.markdown(
            f"""
**Используемые фреймворки:** `ultralytics` (YOLOv8), `streamlit`, `fastapi`.

**Датасет:** «Дефекты линий электропередач v3» — {len(CLASSES)} классов, ~8000 изображений,
~38.6k размеченных объектов.

**Обучение:** смотри `notebooks/train.ipynb`. Там обучаются обе модели:
- **{MODELS['fast']['label']}** — для real-time / слабых CPU
- **{MODELS['accurate']['label']}** — для финальной разметки

После обучения лучшие веса автоматически сохраняются в `app/models/` и подхватываются
этим приложением. До этого момента приложение работает на претренированных весах ultralytics
(детекция «общих» классов COCO — для демонстрации пайплайна).

**Запуск бэкенда отдельно (FastAPI):**
```bash
uvicorn app.backend_api:app --reload
# Документация: http://127.0.0.1:8000/docs
```
"""
        )
