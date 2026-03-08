from __future__ import annotations

import argparse
from pathlib import Path
import re

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
            "Run YOLO on a folder, crop each detection with padding, and save by label."
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
        help="Output folder for per-label cropped images.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Padding ratio added on each side of a detection box (default: 0.10).",
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
    return parser.parse_args()


def collect_images(source_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def crop_with_padding(image, box_xyxy, padding_ratio: float):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio

    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(w, int(round(x2 + pad_x)))
    bottom = min(h, int(round(y2 + pad_y)))

    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def save_background_image(output_dir: Path, stem_token: str, image) -> bool:
    background_dir = output_dir / BACKGROUND_LABEL
    background_dir.mkdir(parents=True, exist_ok=True)
    out_path = background_dir / f"{stem_token}_bg.jpg"
    return bool(cv2.imwrite(str(out_path), image))


def main() -> None:
    args = parse_args()
    source_dir = args.source.resolve()
    model_path = args.model.resolve()
    output_dir = args.output.resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if args.padding < 0:
        raise ValueError("Padding must be >= 0")

    images = collect_images(source_dir)
    if not images:
        print(f"No images found in: {source_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    total_detections = 0
    total_saved = 0
    total_background_saved = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[skip] unreadable image: {image_path}")
            continue

        relative_stem = image_path.relative_to(source_dir).with_suffix("")
        stem_token = "__".join(relative_stem.parts)

        results = model(str(image_path), conf=args.conf, iou=args.iou, verbose=False)
        if not results:
            if save_background_image(output_dir, stem_token, image):
                total_background_saved += 1
            continue

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if save_background_image(output_dir, stem_token, image):
                total_background_saved += 1
            continue

        names = result.names

        for idx, box in enumerate(boxes):
            total_detections += 1

            cls_id = int(box.cls.item())
            label = safe_label_name(str(names.get(cls_id, f"class_{cls_id}")))
            crop = crop_with_padding(image, box.xyxy[0].tolist(), args.padding)
            if crop is None or crop.size == 0:
                continue

            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            conf = float(box.conf.item()) if box.conf is not None else 0.0
            out_name = f"{stem_token}_det{idx:02d}_c{conf:.3f}.jpg"
            out_path = label_dir / out_name
            if cv2.imwrite(str(out_path), crop):
                total_saved += 1

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Images scanned: {len(images)}")
    print(f"Detections found: {total_detections}")
    print(f"Crops saved: {total_saved}")
    print(f"Background images saved: {total_background_saved}")


if __name__ == "__main__":
    main()
