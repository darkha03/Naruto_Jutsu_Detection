from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BACKGROUND_LABEL = "background"


def safe_label_name(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", label.strip())
    return cleaned or "unknown"


def default_source_dir() -> Path:
    preferred = ROOT_DIR / "Jutsu.v6i"
    fallback = ROOT_DIR / "Jutsu.v6i.yolov8"
    if preferred.exists():
        return preferred
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create background crops that do not overlap YOLO detections, "
            "with class-balanced sampling."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source_dir(),
        help="Folder containing images to process.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT_DIR / "models" / "best.engine",
        help="YOLO model path (.pt, .engine, etc.).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "images" / "cropped_by_label",
        help="Output root containing per-label folders.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO inference.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for YOLO NMS.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Final number of background crops to match each class. If omitted, infer from existing class folders.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("min", "mean", "median", "max"),
        default="mean",
        help="How to infer target-per-class from existing class counts.",
    )
    parser.add_argument(
        "--max-candidates-per-image",
        type=int,
        default=4,
        help="Maximum accepted background crops from one image.",
    )
    parser.add_argument(
        "--tries-per-candidate",
        type=int,
        default=40,
        help="Random placement tries for each candidate patch.",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=1.1,
        help="Scale applied to detected box sizes when generating background crop sizes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible crop placement.",
    )
    parser.add_argument(
        "--reset-background",
        action="store_true",
        help="Delete existing files in output/background before generating new crops.",
    )
    return parser.parse_args()


def collect_images(source_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_names(names_obj) -> dict[int, str]:
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, list):
        return {idx: str(name) for idx, name in enumerate(names_obj)}
    return {}


def count_files_in_dir(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file())


def clear_files_in_dir(path: Path) -> int:
    if not path.exists():
        return 0
    deleted = 0
    for file_path in path.iterdir():
        if file_path.is_file():
            file_path.unlink()
            deleted += 1
    return deleted


def compute_target(counts: list[int], mode: str) -> int:
    if not counts:
        return 0
    sorted_counts = sorted(counts)
    if mode == "min":
        return sorted_counts[0]
    if mode == "max":
        return sorted_counts[-1]
    if mode == "median":
        mid = len(sorted_counts) // 2
        if len(sorted_counts) % 2 == 1:
            return sorted_counts[mid]
        return int(round((sorted_counts[mid - 1] + sorted_counts[mid]) / 2.0))
    return int(round(sum(sorted_counts) / float(len(sorted_counts))))


def box_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def build_candidate_sizes(
    image_w: int,
    image_h: int,
    det_boxes: list[tuple[int, int, int, int]],
    size_scale: float,
) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for x1, y1, x2, y2 in det_boxes:
        w = max(16, int(round((x2 - x1) * size_scale)))
        h = max(16, int(round((y2 - y1) * size_scale)))
        w = min(w, image_w)
        h = min(h, image_h)
        sizes.append((w, h))

    if not sizes:
        sizes = [
            (max(32, image_w // 4), max(32, image_h // 4)),
            (max(32, image_w // 3), max(32, image_h // 3)),
        ]

    unique: list[tuple[int, int]] = []
    seen = set()
    for w, h in sizes:
        key = (w, h)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def try_make_nonoverlap_crop(
    image,
    crop_w: int,
    crop_h: int,
    forbidden_boxes: list[tuple[int, int, int, int]],
    tries: int,
    rng: random.Random,
):
    h, w = image.shape[:2]
    if crop_w > w or crop_h > h:
        return None

    max_x = w - crop_w
    max_y = h - crop_h

    for _ in range(max(1, tries)):
        x1 = rng.randint(0, max_x) if max_x > 0 else 0
        y1 = rng.randint(0, max_y) if max_y > 0 else 0
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        candidate = (x1, y1, x2, y2)

        overlaps = any(box_iou_xyxy(candidate, box) > 0.0 for box in forbidden_boxes)
        if overlaps:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        return crop

    return None


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    model_path = args.model.resolve()
    output_dir = args.output.resolve()
    background_dir = output_dir / BACKGROUND_LABEL

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if args.max_candidates_per_image <= 0:
        raise ValueError("--max-candidates-per-image must be > 0")
    if args.tries_per_candidate <= 0:
        raise ValueError("--tries-per-candidate must be > 0")
    if args.size_scale <= 0:
        raise ValueError("--size-scale must be > 0")

    images = collect_images(source_dir)
    if not images:
        print(f"No images found in: {source_dir}")
        return

    model = YOLO(str(model_path))
    names = normalize_names(getattr(model, "names", {}))
    class_ids = sorted(names.keys())
    class_labels = {cid: safe_label_name(names[cid]) for cid in class_ids}

    if not class_ids:
        raise RuntimeError("Could not read class names from model.")

    output_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)

    removed_background = 0
    if args.reset_background:
        removed_background = clear_files_in_dir(background_dir)

    foreground_counts = [
        count_files_in_dir(output_dir / class_labels[cid]) for cid in class_ids
    ]
    inferred_target = compute_target(foreground_counts, args.target_mode)
    target_total = args.target_per_class if args.target_per_class is not None else inferred_target
    target_total = max(0, int(target_total))

    existing_background = count_files_in_dir(background_dir)
    to_create = max(0, target_total - existing_background)

    if to_create == 0:
        print(f"Source: {source_dir}")
        print(f"Output: {output_dir}")
        print(f"Target background count: {target_total}")
        print(f"Existing background count: {existing_background}")
        print("Nothing to do. Background already meets/exceeds target.")
        return

    per_class_quota = {cid: to_create // len(class_ids) for cid in class_ids}
    for cid in class_ids[: to_create % len(class_ids)]:
        per_class_quota[cid] += 1

    per_class_saved = {cid: 0 for cid in class_ids}
    rng = random.Random(args.seed)

    images_scanned = 0
    crops_saved = 0

    for image_path in images:
        if crops_saved >= to_create:
            break

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        images_scanned += 1
        rel_stem = image_path.relative_to(source_dir).with_suffix("")
        stem_token = "__".join(rel_stem.parts)

        results = model(str(image_path), conf=args.conf, iou=args.iou, verbose=False)
        if not results:
            continue

        result = results[0]
        boxes = result.boxes

        det_boxes: list[tuple[int, int, int, int]] = []
        image_classes: set[int] = set()
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                det_boxes.append(
                    (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
                )
                image_classes.add(int(box.cls.item()))

        h, w = image.shape[:2]
        candidate_sizes = build_candidate_sizes(w, h, det_boxes, args.size_scale)
        rng.shuffle(candidate_sizes)

        local_saved = 0
        possible_classes = sorted(c for c in image_classes if c in per_class_quota)
        if not possible_classes:
            possible_classes = class_ids

        for crop_w, crop_h in candidate_sizes:
            if crops_saved >= to_create or local_saved >= args.max_candidates_per_image:
                break

            available = [
                c
                for c in possible_classes
                if per_class_saved[c] < per_class_quota[c]
            ]
            if not available:
                available = [c for c in class_ids if per_class_saved[c] < per_class_quota[c]]
            if not available:
                break

            assigned_cls = min(available, key=lambda c: (per_class_saved[c], c))
            crop = try_make_nonoverlap_crop(
                image=image,
                crop_w=crop_w,
                crop_h=crop_h,
                forbidden_boxes=det_boxes,
                tries=args.tries_per_candidate,
                rng=rng,
            )
            if crop is None:
                continue

            out_name = f"{stem_token}_bg_{class_labels[assigned_cls]}_{per_class_saved[assigned_cls]:04d}.jpg"
            out_path = background_dir / out_name
            if cv2.imwrite(str(out_path), crop):
                per_class_saved[assigned_cls] += 1
                crops_saved += 1
                local_saved += 1

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Class count source mode: {args.target_mode}")
    print(f"Inferred target-per-class: {inferred_target}")
    print(f"Requested target-per-class: {target_total}")
    if args.reset_background:
        print(f"Removed existing background before run: {removed_background}")
    print(f"Existing background before run: {existing_background}")
    print(f"Background created this run: {crops_saved}")
    print(f"Background total after run: {existing_background + crops_saved}")
    print(f"Images scanned: {images_scanned}")

    print("Per-class background contribution:")
    for cid in class_ids:
        print(
            f"  {class_labels[cid]}: "
            f"{per_class_saved[cid]}/{per_class_quota[cid]}"
        )


if __name__ == "__main__":
    main()
