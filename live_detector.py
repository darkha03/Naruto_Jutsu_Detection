import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from stabilizer import Stabilizer
from logger import Logger
from frame_grabber import LatestFrameGrabber

# 1. INITIALIZE THE MODEL, STABILIZER, AND LOGGER
MODEL_PATH = 'bests.engine'  # Path to the exported YOLOv8 engine file
CLASS = "Tiger"

# Performance knobs (trade speed vs accuracy)
IMG_SIZE = 320
CONFIDENCE = 0.5
IOU = 0.5
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS_TARGET = 60
MAX_DETECTIONS = 1
FLUSH_EVERY_N_LOGS = 10

USE_GPU = torch.cuda.is_available()
DEVICE = 0 if USE_GPU else "cpu"
USE_HALF = USE_GPU

model = YOLO(MODEL_PATH)
stabilizer = Stabilizer(enter_point=2.50, confirm_threshold=6.0, exit_point=1.5)
current_stable_class = None
logger = Logger(model_path=MODEL_PATH, logs_directory="logs", max_records=500)

# Warm-up to reduce first-frame latency spikes
_warmup_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
model.predict(
    _warmup_frame,
    conf=CONFIDENCE,
    iou=IOU,
    imgsz=IMG_SIZE,
    max_det=MAX_DETECTIONS,
    device=DEVICE,
    half=USE_HALF,
    verbose=False,
)

# 2. OPEN THE WEBCAM
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS_TARGET)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Sharingan Activated! Looking for Jutsus... Press 'q' to quit.")
print(f"Logging predictions to: {logger.log_file_path}")
print(f"Device: {'GPU' if USE_GPU else 'CPU'} | imgsz={IMG_SIZE} | cam={CAM_WIDTH}x{CAM_HEIGHT}")
actual_cam_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera target FPS: {CAM_FPS_TARGET} | Camera reported FPS: {actual_cam_fps:.2f}")

prev_time = time.perf_counter()
fps = 0.0
frame_idx = 0
pending_logs = 0
capture_fps = 0.0
last_fps_print = time.perf_counter()

grabber = LatestFrameGrabber(cap)
grabber.start()

while cap.isOpened():
    frame, _frame_time, _threaded_capture_fps = grabber.read_latest()
    if frame is None:
        time.sleep(0.001)
        continue

    capture_fps = _threaded_capture_fps

    frame_idx += 1
    
    frame = cv2.flip(frame, 1) 
    
    # 3. RUN THE FRAME THROUGH YOLO
    results = model.predict(
        frame,
        stream=False,
        conf=CONFIDENCE,
        iou=IOU,
        agnostic_nms=True,
        imgsz=IMG_SIZE,
        max_det=MAX_DETECTIONS,
        device=DEVICE,
        half=USE_HALF,
        verbose=False,
    )

    # 4. DRAW THE PREDICTIONS
    annotated_frame = frame
    detected_class = None
    best_confidence = -1.0
    best_box_xyxy = None

    result = results[0] if results else None

    if result is not None and result.boxes is not None and len(result.boxes) > 0:
        names = result.names if hasattr(result, "names") else model.names
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

    if best_box_xyxy is not None and detected_class is not None:
        x1, y1, x2, y2 = map(int, best_box_xyxy)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            annotated_frame,
            f"{detected_class} {best_confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    if detected_class is not None:
        did_log = logger.log_prediction(detected_class, fps, confidence=best_confidence)
        if did_log:
            pending_logs += 1

    detected_confidence = best_confidence if detected_class is not None else 0.0
    stable_class = stabilizer.update(detected_class, detected_confidence)
#    if stable_class is not None:
    current_stable_class = stable_class
    print(f"Stabilized class: {current_stable_class}")

    if pending_logs >= FLUSH_EVERY_N_LOGS:
        logger.flush()
        pending_logs = 0

    # 5. CALCULATE AND DRAW FPS
    current_time = time.perf_counter()
    delta = current_time - prev_time
    prev_time = current_time

    if delta > 0:
        instant_fps = 1.0 / delta
        fps = (0.9 * fps) + (0.1 * instant_fps) if fps > 0 else instant_fps

    cv2.putText(
        annotated_frame,
        f"Loop FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated_frame,
        f"Capture FPS: {capture_fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )

    stable_text = current_stable_class if current_stable_class is not None else "None"
    cv2.putText(
        annotated_frame,
        f"Stable: {stable_text}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    now_print = time.perf_counter()
    if now_print - last_fps_print >= 1.0:
        print(f"Loop FPS={fps:.1f} | Capture FPS={capture_fps:.1f}")
        last_fps_print = now_print

    # 6. SHOW THE VIDEO ON SCREEN
    cv2.imshow('Jutsu Detector - YOLOv8', annotated_frame)

    # Press 'q' to close safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

grabber.stop()
cap.release()
cv2.destroyAllWindows()
logger.flush()
logger.close()

precision, correct_predictions, total_predictions = logger.calculate_precision(CLASS)
summary_file_path, decision = logger.save_run_decision(CLASS, precision)
print(f"Default class: {CLASS}")
print(f"Correct predictions: {correct_predictions}/{total_predictions}")
print(f"Precision: {precision:.4f}")
print(f"Decision: {decision}")
print(f"Saved summary to: {summary_file_path}")
print(f"Current stable class at end of run: {current_stable_class}")