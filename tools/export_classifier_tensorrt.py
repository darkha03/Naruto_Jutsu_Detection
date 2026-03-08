from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from torchvision import models


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MobileNetV3-Small classifier checkpoint to ONNX and TensorRT engine."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_MODEL_DIR / "classifier.pt",
        help="Path to classifier checkpoint (.pt) with model_state_dict.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_MODEL_DIR / "classifier_labels.json",
        help="Path to labels JSON containing 'classes'.",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=DEFAULT_MODEL_DIR / "classifier.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--engine-out",
        type=Path,
        default=DEFAULT_MODEL_DIR / "classifier.engine",
        help="Output TensorRT engine file path.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Model input size (square).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Static batch size used for export/build.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device used for torch -> ONNX export.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Build TensorRT engine with FP16 (requires GPU support).",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=2048,
        help="TensorRT workspace in MB passed to trtexec.",
    )
    parser.add_argument(
        "--trtexec",
        default="trtexec",
        help="Path to trtexec executable.",
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Only export ONNX, do not build TensorRT engine.",
    )
    return parser.parse_args()


def load_classes(labels_path: Path) -> list[str]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    if not isinstance(classes, list) or not classes:
        raise RuntimeError(f"Invalid or empty 'classes' in labels file: {labels_path}")
    return [str(c) for c in classes]


def build_classifier(num_classes: int) -> torch.nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def export_onnx(
    checkpoint_path: Path,
    classes: list[str],
    onnx_out: Path,
    img_size: int,
    batch_size: int,
    opset: int,
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU for ONNX export.")
        device = "cpu"

    torch_device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=torch_device)
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint must contain key: 'model_state_dict'")

    model = build_classifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(torch_device)
    model.eval()

    dummy = torch.randn(batch_size, 3, img_size, img_size, device=torch_device)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=None,
    )


def build_engine(
    trtexec: str,
    onnx_out: Path,
    engine_out: Path,
    img_size: int,
    batch_size: int,
    workspace_mb: int,
    fp16: bool,
) -> None:
    resolved_trtexec = resolve_trtexec(trtexec)
    engine_out.parent.mkdir(parents=True, exist_ok=True)
    shape = f"images:{batch_size}x3x{img_size}x{img_size}"

    cmd = [
        resolved_trtexec,
        f"--onnx={onnx_out}",
        f"--saveEngine={engine_out}",
        f"--workspace={workspace_mb}",
        f"--shapes={shape}",
    ]
    if fp16:
        cmd.append("--fp16")

    print("Running:", " ".join(str(x) for x in cmd))
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Failed to execute trtexec. Provide a valid path via --trtexec or add trtexec to PATH. "
            f"Resolved value: {resolved_trtexec}"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"trtexec failed with exit code {result.returncode}")


def resolve_trtexec(trtexec_arg: str) -> str:
    raw = (trtexec_arg or "").strip().strip('"').strip("'")
    if not raw:
        raw = "trtexec"

    # Handle accidentally double-escaped backslashes from shell quoting.
    raw = raw.replace("\\\\", "\\")

    p = Path(raw)
    if p.suffix.lower() == ".exe" or p.is_absolute() or "\\" in raw or "/" in raw:
        if p.exists():
            return str(p)

    on_path = shutil.which(raw)
    if on_path:
        return on_path

    if os.name == "nt":
        candidates = [
            Path("C:/TensorRT/bin/trtexec.exe"),
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/bin/trtexec.exe"),
            Path("C:/Program Files/TensorRT/bin/trtexec.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "Could not find trtexec. Install TensorRT and ensure trtexec.exe is in PATH, "
        "or pass --trtexec with the full executable path."
    )


def main() -> None:
    args = parse_args()

    checkpoint = args.checkpoint.resolve()
    labels = args.labels.resolve()
    onnx_out = args.onnx_out.resolve()
    engine_out = args.engine_out.resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not labels.exists():
        raise FileNotFoundError(f"Labels file not found: {labels}")
    if args.img_size <= 0:
        raise ValueError("--img-size must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.workspace_mb <= 0:
        raise ValueError("--workspace-mb must be > 0")

    classes = load_classes(labels)
    export_onnx(
        checkpoint_path=checkpoint,
        classes=classes,
        onnx_out=onnx_out,
        img_size=args.img_size,
        batch_size=args.batch_size,
        opset=args.opset,
        device=args.device,
    )
    print(f"ONNX exported: {onnx_out}")

    if not args.skip_engine:
        build_engine(
            trtexec=args.trtexec,
            onnx_out=onnx_out,
            engine_out=engine_out,
            img_size=args.img_size,
            batch_size=args.batch_size,
            workspace_mb=args.workspace_mb,
            fp16=args.fp16,
        )
        print(f"TensorRT engine exported: {engine_out}")
    else:
        print("Skipped TensorRT engine build (--skip-engine).")


if __name__ == "__main__":
    main()
