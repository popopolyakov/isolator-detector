from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CATEGORY_ID_TO_INDEX = {
    2140001: 0,  # vibration_damper
    2150001: 1,  # festoon_insulators
    2280011: 2,  # polymer_insulators
    2160001: 3,  # traverse
    2220001: 4,  # nest
    2280000: 5,  # bad_insulator
    2280001: 6,  # damaged_insulator
    2270001: 7,  # safety_sign+
}

CLASS_NAMES_RU = {
    0: "vibration_damper",
    1: "festoon_insulators",
    2: "polymer_insulators",
    3: "traverse",
    4: "nest",
    5: "bad_insulator",
    6: "damaged_insulator",
    7: "safety_sign+",
}

# Классы, которые мы oversample'им в train для борьбы с 52-кратным дисбалансом
# между festoon (13846 инстансов) и nest (264 инстанса). Числа — это
# коэффициенты дублирования: train-изображение, содержащее `nest`, попадает
# в train-сплит 3 дополнительными копиями, утраивая число примеров nest,
# которые модель видит за эпоху.
OVERSAMPLE_RULES: dict[int, int] = {
    4: 3,  # nest         — 264 инстанса (самый редкий)
    7: 2,  # safety_sign+ — 414 инстансов
}

# Минимальное число val-изображений, в которых должен присутствовать каждый
# класс. Защита от «тихого нуля»: стратифицированный сплит может случайно
# выкинуть редкий класс из валидации, и тогда mAP по нему теряет смысл.
MIN_VAL_IMAGES_PER_CLASS = 5


def coco_ob_to_obb_points(
    bbox: list[float], rotation_deg: float
) -> list[tuple[float, float]]:
    """Преобразует (x, y, w, h) + угол поворота (градусы) в 4 угловые точки
    OBB в пиксельных координатах, упорядоченные по периметру вокруг центра
    (cx, cy): верхний-левый → верхний-правый → нижний-правый → нижний-левый.

    `bbox` в нашем COCO — это неповёрнутый (axis-aligned) прямоугольник;
    `rotation_deg` — угол длинной оси OBB относительно горизонтали,
    измеренный вокруг (cx, cy). Для аннотаций без поля `rotation` передайте 0.
    """
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    hw, hh = w / 2.0, h / 2.0
    rad = math.radians(rotation_deg or 0.0)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    # Углы в локальной системе OBB, затем поворот.
    corners: list[tuple[float, float]] = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        rx = cx + dx * cos_r - dy * sin_r
        ry = cy + dx * sin_r + dy * cos_r
        corners.append((rx, ry))
    return corners


def normalize_and_clip(
    points: list[tuple[float, float]], img_w: int, img_h: int
) -> list[tuple[float, float]]:
    """Переводит пиксельные координаты в [0, 1] и обрезает в единичный квадрат."""
    out: list[tuple[float, float]] = []
    for px, py in points:
        nx = min(max(px / img_w, 0.0), 1.0)
        ny = min(max(py / img_h, 0.0), 1.0)
        out.append((nx, ny))
    return out


def build_index(images_root: Path, file_names: list[str]) -> dict[str, Path]:
    """Строит индекс basename -> путь один раз. Значения `file_name` в COCO-json
    имеют вид 'подпапка/хеш.jpg'."""
    by_basename: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            by_basename.setdefault(path.name, path)
    index: dict[str, Path] = {}
    for fn in file_names:
        rel = Path(fn)
        if rel.name in by_basename:
            index[fn] = by_basename[rel.name]
    return index


def write_data_yaml(out_dir: Path) -> None:
    yaml_lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(CLASS_NAMES_RU)}",
        "names:",
    ]
    for idx in sorted(CLASS_NAMES_RU):
        yaml_lines.append(f"  {idx}: {CLASS_NAMES_RU[idx]}")
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")


def stratified_split(
    image_ids: list[int],
    class_to_image_ids: dict[int, set[int]],
    val_frac: float,
    rng: random.Random,
) -> tuple[set[int], set[int]]:
    """Стратифицированный train/val-сплит, гарантирующий минимум
    MIN_VAL_IMAGES_PER_CLASS изображений каждого класса в val.

    Алгоритм:
      1. Для каждого класса берём обязательный минимум в val (5% от размера
         класса, но не меньше MIN_VAL_IMAGES_PER_CLASS) — только из тех
         изображений, которые ещё не заняты другим классом.
      2. Добиваем val до целевого `val_frac` от общего числа, случайно
         выбирая из оставшихся изображений.
      3. Всё остальное → train.
    """
    n_total = len(image_ids)
    target_val = max(1, int(n_total * val_frac))
    val_ids: set[int] = set()
    used: set[int] = set()

    # обязательные per-class val-примеры.
    for cls_idx, ids in class_to_image_ids.items():
        n_for_class = max(
            MIN_VAL_IMAGES_PER_CLASS,
            int(len(ids) * 0.05),
        )
        candidates = sorted(ids - used)
        rng.shuffle(candidates)
        picked = candidates[:n_for_class]
        val_ids.update(picked)
        used.update(picked)

    # добиваем до target_val из оставшихся изображений.
    if len(val_ids) < target_val:
        pool = [iid for iid in image_ids if iid not in used]
        rng.shuffle(pool)
        for iid in pool:
            if len(val_ids) >= target_val:
                break
            val_ids.add(iid)
            used.add(iid)
    elif len(val_ids) > target_val:
        # Если минимумы уже превысили target (редко на маленьких датасетах), случайно убираем лишние, чтобы попасть в target.
        excess = list(val_ids)
        rng.shuffle(excess)
        val_ids = set(val_ids) - set(excess[: len(val_ids) - target_val])

    train_ids = set(image_ids) - val_ids
    return train_ids, val_ids


def main() -> None:
    p = argparse.ArgumentParser(
        description="Подготовка датасета изоляторов: COCO -> YOLO-OBB, "
                    "стратифицированный сплит, oversample редких классов."
    )
    p.add_argument("--source", type=Path,
                   default=Path("Дефекты линий электропередач_v3/insulators"),
                   help="Корневая директория исходного датасета (содержит annotation_data.json).")
    p.add_argument("--out", type=Path, default=Path("data/insulators_yolo"),
                   help="Куда записать YOLO-OBB датасет.")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Доля валидационной выборки (по умолчанию 0.15).")
    p.add_argument("--seed", type=int, default=42,
                   help="Сид для воспроизводимости сплита.")
    p.add_argument("--max-images", type=int, default=0,
                   help="Если >0, ограничить размер датасета для быстрых прогонов "
                        "(0 = использовать все изображения).")
    p.add_argument("--use-symlinks", action="store_true",
                   help="Создавать симлинки на исходные изображения вместо копирования "
                        "(экономит диск).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    src_dir: Path = args.source.resolve()
    images_root = src_dir / "images"
    out_dir: Path = args.out.resolve()
    # Перед каждой записью полностью очищаем выходную директорию
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    with (src_dir / "annotation_data.json").open(encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    if args.max_images > 0:
        rng.shuffle(images)
        images = images[: args.max_images]
    keep_ids = {im["id"] for im in images}

    index = build_index(images_root, [im["file_name"] for im in images])
    print(f"Найдено {len(index)}/{len(images)} изображений на диске")

    # Группируем аннотации по изображениям и строим индекс класс -> изображения, который нужен для стратификации.
    anns_by_image: dict[int, list] = defaultdict(list)
    class_to_image_ids: dict[int, set[int]] = defaultdict(set)
    skipped_unknown = 0
    for ann in annotations:
        if ann["image_id"] not in keep_ids:
            continue
        if ann["category_id"] not in CATEGORY_ID_TO_INDEX:
            skipped_unknown += 1
            continue
        anns_by_image[ann["image_id"]].append(ann)
        cls_idx = CATEGORY_ID_TO_INDEX[ann["category_id"]]
        class_to_image_ids[cls_idx].add(ann["image_id"])
    if skipped_unknown:
        print(f"Пропущено {skipped_unknown} аннотаций с неизвестным category_id")

    # Стратифицированный сплит.
    image_ids = [im["id"] for im in images]
    rng.shuffle(image_ids)
    train_ids, val_ids = stratified_split(
        image_ids, class_to_image_ids, args.val_frac, rng,
    )
    print(
        f"Сплит: train={len(train_ids)} (после oversample станет больше), "
        f"val={len(val_ids)}"
    )

    # Sanity-check покрытия классов в val.
    val_class_counts: Counter = Counter()
    for ann in annotations:
        if ann["image_id"] in val_ids and ann["category_id"] in CATEGORY_ID_TO_INDEX:
            val_class_counts[CATEGORY_ID_TO_INDEX[ann["category_id"]]] += 1
    val_img_per_class: dict[int, int] = {
        cls: sum(1 for iid in ids if iid in val_ids)
        for cls, ids in class_to_image_ids.items()
    }
    print("Val: изображений на класс:")
    for cls in sorted(CLASS_NAMES_RU):
        print(
            f"  {cls} {CLASS_NAMES_RU[cls]:<22} "
            f"images={val_img_per_class.get(cls, 0):>4}  "
            f"instances={val_class_counts.get(cls, 0):>5}"
        )

    # Пишем лейблы и ссылки на изображения.
    n_written: Counter = Counter()
    n_boxes: Counter = Counter()
    n_oversample_dup: Counter = Counter()
    n_missing = 0
    n_empty = 0
    for im in images:
        src_path = index.get(im["file_name"])
        if src_path is None:
            n_missing += 1
            continue
        split = "val" if im["id"] in val_ids else "train"

        # Собираем OBB-лейбл (по одной строке на аннотацию, 9 колонок).
        image_anns = anns_by_image.get(im["id"], [])
        lines: list[str] = []
        for ann in image_anns:
            rotation = float(ann.get("rotation") or 0.0)
            corners = coco_ob_to_obb_points(ann["bbox"], rotation)
            corners = normalize_and_clip(corners, im["width"], im["height"])
            cls = CATEGORY_ID_TO_INDEX[ann["category_id"]]
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
            lines.append(f"{cls} {coords}")

        # Считаем коэффициент дублирования: 1 для обычных, N для изображений с редкими классами (берём максимум из всех подходящих правил).
        dup_factors = [1]
        for ann in image_anns:
            cls = CATEGORY_ID_TO_INDEX[ann["category_id"]]
            if cls in OVERSAMPLE_RULES:
                dup_factors.append(OVERSAMPLE_RULES[cls])
        n_copies = max(dup_factors) if split == "train" else 1

        # Пишем копии (или симлинки) изображения и соответствующие лейблы.
        for k in range(n_copies):
            suffix = "" if k == 0 else f"_dup{k}"
            dst_img = out_dir / "images" / split / f"{src_path.stem}{suffix}{src_path.suffix}"
            dst_lbl = out_dir / "labels" / split / f"{src_path.stem}{suffix}.txt"
            if args.use_symlinks:
                if not dst_img.exists():
                    os.symlink(src_path, dst_img)
            else:
                if not dst_img.exists():
                    shutil.copy2(src_path, dst_img)
            dst_lbl.write_text("\n".join(lines))
            n_written[split] += 1
            n_boxes[split] += len(lines)
            if k > 0:
                n_oversample_dup[split] += 1
        if not image_anns:
            n_empty += 1

    write_data_yaml(out_dir)

    print()
    print("=== Итог по датасету ===")
    print(f"Выходная директория: {out_dir}")
    print(f"  train images: {n_written['train']}, boxes: {n_boxes['train']}")
    print(f"  val   images: {n_written['val']},   boxes: {n_boxes['val']}")
    if n_oversample_dup["train"]:
        print(
            f"  oversample dup изображений в train: {n_oversample_dup['train']} "
            f"(правила: {OVERSAMPLE_RULES})"
        )
    print(f"Изображений без боксов: {n_empty}")
    if n_missing:
        print(f"Изображений не найдено на диске: {n_missing}")
    print(f"data.yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
