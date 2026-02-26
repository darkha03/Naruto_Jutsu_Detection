import cv2
import time
import numpy as np
import torch
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.stabilizer import Stabilizer
from core.logger import Logger
from core.frame_grabber import LatestFrameGrabber

MODEL_PATH = str(ROOT_DIR / "models" / "bests.engine")
CLASS = "Tiger"

IMG_SIZE = 320
CONFIDENCE = 0.5
IOU = 0.5
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS_TARGET = 60
MAX_DETECTIONS = 1

# Test toggles
USE_MJPG = False
HEADLESS = False
DRAW_EVERY_N_FRAMES = 1
DISPLAY_EVERY_N_FRAMES = 1
PRINT_EVERY_SEC = 1.0


def main():
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"
    use_half = use_gpu

    model = YOLO(MODEL_PATH)
    stabilizer = Stabilizer()
    current_stable_class = None
    logger = Logger(model_path=MODEL_PATH, logs_directory=str(ROOT_DIR / "logs"), max_records=500)

    warmup = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    model.predict(
        warmup,
        conf=CONFIDENCE,
        iou=IOU,
        imgsz=IMG_SIZE,
        max_det=MAX_DETECTIONS,
        device=device,
        half=use_half,
        verbose=False,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Device={'GPU' if use_gpu else 'CPU'} | Model={MODEL_PATH} | IMG_SIZE={IMG_SIZE}")
    print(f"Requested cam={CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS_TARGET} | Reported FPS={cap.get(cv2.CAP_PROP_FPS):.1f} | MJPG={USE_MJPG}")

    grabber = LatestFrameGrabber(cap)
    grabber.start()

    prev_time = time.perf_counter()
    last_print = prev_time
    loop_fps = 0.0
    infer_fps = 0.0
    capture_fps = 0.0
    frame_idx = 0
    last_logged_frame_time = None

    try:
        while cap.isOpened():
            frame, frame_time, threaded_capture_fps = grabber.read_latest()
            if frame is None:
                time.sleep(0.001)
                continue

            frame_idx += 1
            capture_fps = threaded_capture_fps
            frame_age_ms = max(0.0, (time.perf_counter() - frame_time) * 1000.0) if frame_time > 0 else 0.0

            frame = cv2.flip(frame, 1)

            results = model.predict(
                frame,
                stream=False,
                conf=CONFIDENCE,
                iou=IOU,
                agnostic_nms=True,
                imgsz=IMG_SIZE,
                max_det=MAX_DETECTIONS,
                device=device,
                half=use_half,
                verbose=False,
            )

            annotated = frame
            detected_class = None
            best_confidence = -1.0
            best_box_xyxy = None

            result = results[0] if results else None
            pre_ms, inf_ms, post_ms = 0.0, 0.0, 0.0
            if result is not None:
                speed = result.speed if hasattr(result, "speed") and result.speed is not None else {}
                pre_ms = float(speed.get("preprocess", 0.0))
                inf_ms = float(speed.get("inference", 0.0))
                post_ms = float(speed.get("postprocess", 0.0))
                if inf_ms > 0:
                    infer_inst = 1000.0 / inf_ms
                    infer_fps = (0.9 * infer_fps) + (0.1 * infer_inst) if infer_fps > 0 else infer_inst

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

            if frame_idx % DRAW_EVERY_N_FRAMES == 0 and best_box_xyxy is not None and detected_class is not None:
                x1, y1, x2, y2 = map(int, best_box_xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    annotated,
                    f"{detected_class} {best_confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            detected_confidence = best_confidence if detected_class is not None else 0.0
            stable_class= stabilizer.update(detected_class, detected_confidence)
            current_stable_class = stable_class
                        
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                loop_inst = 1.0 / dt
                loop_fps = (0.9 * loop_fps) + (0.1 * loop_inst) if loop_fps > 0 else loop_inst

            model_total_ms = pre_ms + inf_ms + post_ms

            cv2.putText(annotated, f"Loop FPS: {loop_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Capture FPS: {capture_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Infer FPS: {infer_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 70), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Model ms: {model_total_ms:.1f} | FrameAge ms: {frame_age_ms:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Stable: {current_stable_class if current_stable_class else 'None'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if detected_class is not None:
                is_new_frame = (last_logged_frame_time is None) or (frame_time != last_logged_frame_time)
                if is_new_frame:
                    last_logged_frame_time = frame_time

                logger.log_prediction(
                    detected_class,
                    loop_fps,
                    confidence=best_confidence,
                    frame_time=frame_time,
                    is_new_frame=is_new_frame,
                )

            if not HEADLESS:
                if frame_idx % DISPLAY_EVERY_N_FRAMES == 0:
                    cv2.imshow("Jutsu Detector - Testbed", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if now - last_print >= PRINT_EVERY_SEC:
                    print(
                        f"Loop FPS={loop_fps:.1f} | Capture FPS={capture_fps:.1f} | Infer FPS={infer_fps:.1f} | "
                        f"Model ms={model_total_ms:.1f} | FrameAge ms={frame_age_ms:.1f}"
                    )
                    last_print = now
    finally:
        grabber.stop()
        cap.release()
        if not HEADLESS:
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


if __name__ == "__main__":
    main()
