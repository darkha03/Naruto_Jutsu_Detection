from __future__ import annotations

import queue
import threading
import time


def put_latest(q: "queue.Queue", item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            _ = q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def clamp_box(box_xyxy, frame_shape):
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 = max(0, min(x1, frame_shape[1] - 1))
    y1 = max(0, min(y1, frame_shape[0] - 1))
    x2 = max(0, min(x2, frame_shape[1]))
    y2 = max(0, min(y2, frame_shape[0]))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


class DetectorWorker(threading.Thread):
    def __init__(self, detector, in_queue, out_queue, stop_event, lock, latest_state):
        super().__init__(name="detector_worker", daemon=True)
        self.detector = detector
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.lock = lock
        self.latest_state = latest_state
        self.detect_fps_local = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                packet = self.in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            frame = packet["frame"]
            frame_id = packet["frame_id"]
            frame_time = packet["frame_time"]

            detection = self.detector.detect_best_box(frame)
            inf_ms = float(detection.get("inf_ms", 0.0))
            if inf_ms > 0:
                inst = 1000.0 / inf_ms
                self.detect_fps_local = (0.9 * self.detect_fps_local) + (0.1 * inst) if self.detect_fps_local > 0 else inst

            state = {
                "frame_id": frame_id,
                "frame_time": frame_time,
                "box_xyxy": detection.get("box_xyxy"),
                "detection_confidence": float(detection.get("confidence", 0.0)),
                "detect_ms": float(detection.get("detect_ms", 0.0)),
                "detect_fps": self.detect_fps_local,
            }
            with self.lock:
                self.latest_state.update(state)

            put_latest(
                self.out_queue,
                {
                    "frame": frame,
                    "frame_id": frame_id,
                    "frame_time": frame_time,
                    "box_xyxy": detection.get("box_xyxy"),
                    "detection_confidence": float(detection.get("confidence", 0.0)),
                },
            )


class ClassifierWorker(threading.Thread):
    def __init__(self, classifier, in_queue, stop_event, lock, latest_state):
        super().__init__(name="classifier_worker", daemon=True)
        self.classifier = classifier
        self.in_queue = in_queue
        self.stop_event = stop_event
        self.lock = lock
        self.latest_state = latest_state
        self.classify_fps_local = 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                packet = self.in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            frame = packet["frame"]
            frame_id = packet["frame_id"]
            frame_time = packet["frame_time"]
            raw_box = packet["box_xyxy"]

            class_name = None
            class_confidence = 0.0
            classify_ms = 0.0
            clamped_box = None

            if raw_box is not None:
                clamped_box = clamp_box(raw_box, frame.shape)
                if clamped_box is not None:
                    x1, y1, x2, y2 = clamped_box
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        t0 = time.perf_counter()
                        cls = self.classifier.classify_crop(crop)
                        classify_ms = (time.perf_counter() - t0) * 1000.0
                        class_name = cls.get("class_name")
                        class_confidence = float(cls.get("confidence", 0.0))
                        if classify_ms > 0:
                            inst = 1000.0 / classify_ms
                            self.classify_fps_local = (0.9 * self.classify_fps_local) + (0.1 * inst) if self.classify_fps_local > 0 else inst

            state = {
                "frame_id": frame_id,
                "class_name": class_name,
                "class_confidence": class_confidence,
                "classify_ms": classify_ms,
                "classify_fps": self.classify_fps_local,
                "box_xyxy": clamped_box,
                "frame_time": frame_time,
            }
            with self.lock:
                self.latest_state.update(state)
