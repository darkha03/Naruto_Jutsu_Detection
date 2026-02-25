import numpy as np
import torch
from ultralytics import YOLO


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
    ):
        self.model_path = model_path
        self.img_size = img_size
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 0 if self.use_gpu else "cpu"
        self.use_half = self.use_gpu

        self.model = YOLO(self.model_path)
        self._warmup(warmup_height=warmup_height, warmup_width=warmup_width)

    def _warmup(self, warmup_height, warmup_width):
        warmup_frame = np.zeros((warmup_height, warmup_width, 3), dtype=np.uint8)
        self.model.predict(
            warmup_frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.img_size,
            max_det=self.max_detections,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )

    def predict(self, frame):
        results = self.model.predict(
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