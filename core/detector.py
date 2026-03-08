import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path


class Detector:
    def __init__(
        self,
        model_path,
        img_size=320,
        confidence=0.5,
        iou=0.5,
        max_detections=1,
        use_gpu=True,
        warmup_height=480,
        warmup_width=640,
        fallback_model_path=None,
    ):
        self.model_path = model_path
        self.fallback_model_path = fallback_model_path
        self.img_size = img_size
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 0 if self.use_gpu else "cpu"
        self.use_half = self.use_gpu
        self.active_model_path = None

        self.model = self._load_model_with_fallback(
            warmup_height=warmup_height,
            warmup_width=warmup_width,
        )

    def _candidate_model_paths(self):
        candidates = [self.model_path]

        if self.fallback_model_path:
            candidates.append(self.fallback_model_path)
        else:
            primary = Path(self.model_path)
            if primary.suffix.lower() == ".engine":
                auto_pt = str(primary.with_suffix(".pt"))
                if auto_pt != self.model_path:
                    candidates.append(auto_pt)

        deduped = []
        for path in candidates:
            if path not in deduped:
                deduped.append(path)
        return deduped

    def _load_model_with_fallback(self, warmup_height, warmup_width):
        errors = []

        for candidate_path in self._candidate_model_paths():
            try:
                candidate_model = YOLO(candidate_path)
                self.model = candidate_model
                self._warmup(warmup_height=warmup_height, warmup_width=warmup_width)
                self.active_model_path = candidate_path

                if candidate_path != self.model_path:
                    print(f"Primary model failed, using fallback model: {candidate_path}")
                return candidate_model
            except Exception as exc:
                errors.append(f"{candidate_path}: {exc}")

        joined_errors = " | ".join(errors) if errors else "No candidate models were provided."
        raise RuntimeError(f"Failed to load any model candidate. Details: {joined_errors}")

    def _warmup(self, warmup_height, warmup_width):
        warmup_frame = np.zeros((warmup_height, warmup_width, 3), dtype=np.uint8)
        self._predict_raw(warmup_frame)

    def _predict_raw(self, frame):
        return self.model.predict(
            frame,
            stream=False,
            conf=self.confidence,
            iou=self.iou,
            agnostic_nms=True,
            imgsz=self.img_size,
            max_det=self.max_detections,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )

    @staticmethod
    def _extract_speed_ms(result):
        if result is None:
            return 0.0, 0.0, 0.0
        speed = result.speed if hasattr(result, "speed") and result.speed is not None else {}
        pre_ms = float(speed.get("preprocess", 0.0))
        inf_ms = float(speed.get("inference", 0.0))
        post_ms = float(speed.get("postprocess", 0.0))
        return pre_ms, inf_ms, post_ms

    def detect_best_box(self, frame):
        results = self._predict_raw(frame)
        result = results[0] if results else None

        best_confidence = 0.0
        best_box_xyxy = None

        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box_xyxy = box.xyxy[0].tolist()

        pre_ms, inf_ms, post_ms = self._extract_speed_ms(result)
        return {
            "has_detection": best_box_xyxy is not None,
            "confidence": best_confidence,
            "box_xyxy": best_box_xyxy,
            "raw_result": result,
            "pre_ms": pre_ms,
            "inf_ms": inf_ms,
            "post_ms": post_ms,
            "detect_ms": pre_ms + inf_ms + post_ms,
        }

    def predict(self, frame):
        results = self._predict_raw(frame)

        detected_class = None
        best_confidence = -1.0
        best_box_xyxy = None

        result = results[0] if results else None
        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            names = result.names if hasattr(result, "names") else self.model.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                else:
                    class_name = names[class_id] if class_id < len(names) else str(class_id)

                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                if confidence > best_confidence:
                    best_confidence = confidence
                    detected_class = class_name
                    best_box_xyxy = box.xyxy[0].tolist()

        return {
            "has_detection": detected_class is not None,
            "class_name": detected_class,
            "confidence": best_confidence if detected_class is not None else 0.0,
            "box_xyxy": best_box_xyxy,
            "raw_result": result,
        }