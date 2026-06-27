# Детектор дефектов воздушных линий электропередачи

Проект для автоматического обнаружения элементов и повреждений на снимках
ВЛЭП, полученных дронами. Детектируются **8 классов**: гирлянды стеклянных и
полимерных изоляторов, гасители вибрации, траверсы, гнёзда птиц,
отсутствующие и повреждённые изоляторы, диспетчерские таблички. Классы с
флагом `is_violation` (`nest`, `bad_insulator`, `damaged_insulator`)
трактуются как нарушения и считаются отдельно.

Пайплайн — две модели **YOLO OBB** (Oriented Bounding Box) из библиотеки
`ultralytics`, обёрнутые в **Streamlit**-фронтенд и **FastAPI**-бэкенд.

---

## Содержание

1. [Состав проекта](#состав-проекта)
2. [Датасет](#датасет)
3. [Подготовка данных](#подготовка-данных)
4. [Обучение моделей](#обучение-моделей)
5. [Результаты обучения](#результаты-обучения)
6. [Метрики и валидация](#метрики-и-валидация)
7. [Реестр моделей и инференс](#реестр-моделей-и-инференс)
8. [Веб-приложение](#веб-приложение)
9. [REST API](#rest-api)
10. [Воспроизводимость и окружение](#воспроизводимость-и-окружение)

---

## Состав проекта

```
isolator-detector/
├── app/
│   ├── streamlit_app.py            # Streamlit-фронтенд
│   ├── backend_api.py              # FastAPI-бэкенд
│   ├── inference.py                # Общая логика инференса, реестр моделей
│   └── models/                     # Обученные веса + metrics.json
│       ├── fast_yolov8n-obb.pt     # 6 МБ
│       ├── accurate_yolo11l-obb.pt # 50 МБ
│       └── metrics.json
├── notebooks/
│   └── train.ipynb                 # Полный пайплайн обучения
├── scripts/
│   └── prepare_dataset.py          # COCO → YOLO-OBB, split, oversample
├── data/                           # Сгенерированный датасет (в .gitignore)
├── runs/                           # Артефакты обучения (Ultralytics)
├── .streamlit/config.toml
├── packages.txt
├── requirements.txt
└── README.md
```

---

## Датасет

Источник — «Дефекты линий электропередач v3» (единый COCO-json
`annotation_data.json`).

| Параметр                 | Значение |
|--------------------------|----------|
| Изображений              |    7988  |
| Аннотаций                |   38642  |
| Классов                  |       8  |
| Дисбаланс festoon/nest   |   ~52x   |

### Классы и маппинг COCO id → YOLO index

| idx | Класс               | COCO id | is_violation |
|----:|---------------------|---------|:------------:|
|   0 | vibration_damper    | 2140001 |      no      |
|   1 | festoon_insulators  | 2150001 |      no      |
|   2 | polymer_insulators  | 2280011 |      no      |
|   3 | traverse            | 2160001 |      no      |
|   4 | nest                | 2220001 |     yes      |
|   5 | bad_insulator       | 2280000 |     yes      |
|   6 | damaged_insulator   | 2280001 |     yes      |
|   7 | safety_sign+        | 2270001 |      no      |

---

## Подготовка данных

Скрипт `scripts/prepare_dataset.py` преобразует исходный COCO в YOLO-OBB
(DOTA-формат — `cls x1 y1 x2 y2 x3 y3 x4 y4` — 4 вершины повёрнутого
прямоугольника в нормализованных координатах). Обрабатывает поле `rotation`
для классов `festoon_insulators` и `polymer_insulators` (≈46% всех боксов);
для классов без поворота четыре вершины совпадают с углами AABB.

Особенности сплита:
- **Стратифицированный по классам** — каждый класс гарантированно попадает
  в валидационную выборку.
- **Oversample** train-изображений с редкими классами: `nest` ×3,
  `safety_sign+` ×2 — компенсация 52-кратного дисбаланса.
- Доля валидации: 15% (по умолчанию).
- Параметр `--use-symlinks` — экономит ≈7 ГБ диска за счёт ссылок на
  оригиналы вместо копий.

Выходная структура:

```
data/insulators_yolo/
├── data.yaml
├── images/{train,val}/*.jpg
└── labels/{train,val}/*.txt
```

---

## Обучение моделей

Полный пайплайн собран в `notebooks/train.ipynb` и состоит из пяти шагов.

### Шаг 1. Окружение

Авто-выбор устройства в порядке приоритета: `cuda` → `mps` → `cpu`. Версия
`torch` подтягивается под видеокарту (например, `2.12.0+cu132` под CUDA
12.x). Зависимости ноутбука — `ultralytics>=8.2`, `opencv-python`, `pyyaml`,
`torchvision`.

### Шаг 2. Подготовка датасета

Запуск `prepare_dataset.py` с параметрами `MAX_IMAGES=0` (полный датасет),
`VAL_FRAC=0.15`, `USE_SYMLINKS=True`. Для smoke-теста — `MAX_IMAGES=600`.

### Шаг 3. Аугментации (общие для обеих моделей)

Подобраны под дрон-съёмку:
повороты камеры, съёмка с разных ракурсов, переменная освещённость.

| Группа      | Параметры                                                                                              |
|-------------|--------------------------------------------------------------------------------------------------------|
| Color       | `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`                                                                |
| Geometry    | `degrees=10.0`, `translate=0.15`, `scale=0.5`, `shear=2.0`, `perspective=0.0005`, `fliplr=0.5`, `flipud=0.1` |
| Composition | `mosaic=1.0`, `mixup=0.15`, `copy_paste=0.2`                                                            |
| Occlusion   | `erasing=0.2`                                                                                          |

### Шаг 4. Обучение YOLOv8n-OBB (быстрая модель)

| Параметр    | Значение                                       |
|-------------|------------------------------------------------|
| Архитектура | YOLOv8n-OBB, 82 слоя, 3.08M параметров, 8.3 GFLOPs |
| Эпохи       | 60 (на полном датасете) / 15 (smoke-тест)      |
| imgsz       | 960                                            |
| batch       | 12                                             |
| patience    | 15                                             |
| workers     | 4                                              |

### Шаг 5. Обучение YOLO11l-OBB (точная модель)

| Параметр    | Значение                                        |
|-------------|-------------------------------------------------|
| Архитектура | YOLO11l-OBB, 200 слоёв, 26.13M параметров, 90.3 GFLOPs |
| Эпохи       | 60 (на полном датасете) / 20 (smoke-тест)       |
| imgsz       | 960                                             |
| batch       | 4                                               |
| patience    | 25                                              |
| workers     | 2                                               |

### Шаг 6. Сохранение артефактов

Лучшие веса из `runs/{model}_insulators/weights/best.pt` копируются в
`app/models/{fast_yolov8n-obb,accurate_yolo11l-obb}.pt`. Метрики
записываются в `app/models/metrics.json`. Дополнительно прогоняется
sanity-check: обе модели загружаются и делают предсказание на случайной
val-картинке.

---

## Результаты обучения

Замеры — на валидационной выборке после обучения на полном датасете
(7988 изображений, сплит 15% val → 1198 val-изображений, 7004
ground-truth инстанса). Окружение — `ultralytics 8.4.67`, `python 3.14.6`,
`torch 2.12.0+cu132`, GPU NVIDIA GeForce RTX 4070 SUPER (12 ГБ).

| Модель                  | Размер | Время инференса (GPU) | mAP@0.5  | mAP@0.5:0.95 | Precision | Recall |
|-------------------------|-------:|----------------------:|---------:|-------------:|----------:|-------:|
| YOLOv8n-OBB (fast)      |   6 МБ |          3.2 мс/image | **0.8717** |       0.6865 |     0.874 |  0.827 |
| YOLO11l-OBB (accurate)  |  50 МБ |         25.7 мс/image | **0.9130** |       0.7527 |     0.906 |  0.873 |

Содержимое `app/models/metrics.json`:

```json
{
  "fast":     {"model": "yolov8n-obb", "mAP@0.5": 0.8717, "mAP@0.5:0.95": 0.6865, "weights": "fast_yolov8n-obb.pt"},
  "accurate": {"model": "yolo11l-obb", "mAP@0.5": 0.9130, "mAP@0.5:0.95": 0.7527, "weights": "accurate_yolo11l-obb.pt"}
}
```

Обе модели перекрывают целевой порог **mAP@0.5 ≥ 0.6** с запасом. Точная
модель даёт +4.1 п.п. по mAP@0.5 и +6.6 п.п. по mAP@0.5:0.95 относительно
быстрой ценой ≈8× большего размера весов и ≈8× большего времени инференса.

---

## Метрики и валидация

Основная метрика — **mAP@0.5** (критерий задания ≥ 0.6). Дополнительно
считается **mAP@0.5:0.95** — она чувствительна к качеству локализации и
именно на ней максимально проявляется выигрыш OBB над AABB.

После обучения Ultralytics пишет в `runs/{model}_insulators/val*/`:
- `Box(P)`, `R`, `mAP50`, `mAP50-95` по каждому классу;
- per-class PR-кривые;
- матрицы ошибок (`confusion_matrix_*.png`);
- `results.csv` с метриками по эпохам.

В UI дополнительно отображаются:
- `mAP@0.5` для выбранной модели;
- время инференса (мс) на загруженном снимке;
- таблица детекций с `class_ru`, `confidence`, `bbox`;
- bar-chart распределения детекций по классам;
- счётчик нарушений (классы с `is_violation=true`: `nest`,
  `bad_insulator`, `damaged_insulator`).

---

## Реестр моделей и инференс

Единый реестр в `app/inference.py::MODELS`:

| id        | Модель      | Размер | Назначение             | Fallback                       |
|-----------|-------------|-------:|------------------------|--------------------------------|
| `fast`    | YOLOv8n-OBB |  ~7 МБ | real-time, слабые CPU  | `yolov8n-obb.pt` (pretrained)  |
| `accurate`| YOLO11l-OBB | ~53 МБ | финальная разметка     | `yolo11l-obb.pt` (pretrained)  |

Каждая запись содержит `id`, `label`, `description`, `size_mb_approx`,
`weights: Path` и `fallback: str`. Если файл из `weights` существует —
загружается он; иначе — претренированная модель из `ultralytics` по
`fallback` (это поведение позволяет открыть приложение сразу после
клонирования, до обучения).

`Detector` — ленивый wrapper над `ultralytics.YOLO`, кешируется по
`model_id` (один экземпляр на процесс на модель). Метод `predict()`:
- принимает байты изображения, `conf`, `iou`, `imgsz`;
- возвращает словарь с `inference_ms`, `image_size`, списком `detections`
  (поля `class_id`, `class_key`, `class_ru`, `is_violation`,
  `confidence`, `bbox_xyxy`, `bbox_points` для OBB-вершин,
  `rotation_deg`) и `annotated_jpeg_b64`.

Аннотирование — через `PIL.ImageDraw`: рисуется повёрнутый
четырёхугольник по 4 вершинам OBB, подпись — `class_ru @ conf`,
цвет — из записи класса в `CLASSES`.

---

## Веб-приложение

Streamlit-фронтенд (`app/streamlit_app.py`) — единая точка входа для
демо и ручной валидации. Главный файл для деплоя на Streamlit Cloud:
`app/streamlit_app.py`.

---

## REST API

FastAPI-бэкенд (`app/backend_api.py`) — программный доступ к инференсу.
Запуск — `uvicorn app.backend_api:app --reload`, Swagger UI —
`/docs`.

| Метод | URL              | Описание                                                            |
|-------|------------------|---------------------------------------------------------------------|
| GET   | `/`              | Информация о сервисе                                               |
| GET   | `/health`        | `{"status":"ok"}`                                                  |
| GET   | `/models`        | Список моделей из реестра                                          |
| POST  | `/predict`       | `multipart`: `file`, `model_id`, `conf`, `iou` -> JSON + base64    |
| POST  | `/predict/image` | То же, что `/predict`, но возвращает JPEG напрямую                 |

Структура ответа `/predict`:

```json
{
  "model_id": "accurate",
  "model_label": "YOLOv11l-OBB — точная",
  "using_local_weights": true,
  "inference_ms": 142.3,
  "image_size": [4032, 3024],
  "detections": [
    {
      "class_id": 6,
      "class_key": "damaged_insulator",
      "class_ru": "Поврежденный изолятор",
      "is_violation": true,
      "confidence": 0.87,
      "bbox_xyxy": [1023.4, 540.2, 1180.1, 720.9]
    }
  ],
  "annotated_jpeg_b64": "..."
}
```

---

### Артефакты

- `app/models/fast_yolov8n-obb.pt` — 6 МБ (≈ 7 МБ у `ultralytics`,
  разница — квантование заголовка).
- `app/models/accurate_yolo11l-obb.pt` — 50 МБ (≈ 53 МБ у `ultralytics`).
- `app/models/metrics.json` — заполняется на финальном шаге ноутбука.
- `runs/{model}_insulators/` — полные логи Ultralytics: `results.csv`,
  `train_batch*.jpg`, `val_batch*.jpg`, PR-кривые, матрицы ошибок.

