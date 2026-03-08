from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models, transforms


class Classifier:
    def __init__(
        self,
        model_path,
        label_path,
        img_size=224,
        use_gpu=True,
        fallback_model_path=None,
    ):
        self.model_path = str(model_path)
        self.label_path = str(label_path)
        self.fallback_model_path = str(fallback_model_path) if fallback_model_path else None
        self.img_size = img_size

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.active_model_path = None

        self.labels = self._load_labels(self.label_path)
        self.class_names = self.labels.get("classes", [])
        if not self.class_names:
            raise RuntimeError(f"No classes found in labels file: {self.label_path}")

        self.model = self._load_model_with_fallback()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.img_size, self.img_size), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _candidate_model_paths(self):
        candidates = [self.model_path]
        if self.fallback_model_path:
            candidates.append(self.fallback_model_path)

        deduped = []
        for path in candidates:
            if path and path not in deduped:
                deduped.append(path)
        return deduped

    def _load_labels(self, label_path):
        path = Path(label_path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier labels file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_model(self, num_classes: int):
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    def _load_model_with_fallback(self):
        errors = []
        for candidate_path in self._candidate_model_paths():
            try:
                candidate = Path(candidate_path)
                if not candidate.exists():
                    raise FileNotFoundError(f"File not found: {candidate}")

                checkpoint = torch.load(candidate, map_location=self.device)
                model = self._build_model(len(self.class_names))
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.to(self.device)
                model.eval()

                self.active_model_path = str(candidate)
                if str(candidate) != self.model_path:
                    print(f"Primary classifier failed, using fallback model: {candidate}")
                return model
            except Exception as exc:
                errors.append(f"{candidate_path}: {exc}")

        joined_errors = " | ".join(errors) if errors else "No candidate classifier models were provided."
        raise RuntimeError(f"Failed to load any classifier candidate. Details: {joined_errors}")

    def warmup(self):
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        _ = self.classify_crop(dummy)

    def classify_crop(self, crop_bgr):
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return {
                "class_name": None,
                "confidence": 0.0,
                "class_id": -1,
            }

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        class_id = int(idx.item())
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
        return {
            "class_name": class_name,
            "confidence": float(conf.item()),
            "class_id": class_id,
        }