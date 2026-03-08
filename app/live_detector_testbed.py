import cv2
import time
import torch
import sys
import queue
from pathlib import Path
import threading

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.classifier import Classifier
from core.detector import Detector
from core.stabilizer import Stabilizer
from core.logger import Logger
from core.frame_grabber import LatestFrameGrabber
from core.pipeline_workers import DetectorWorker, ClassifierWorker, put_latest, clamp_box

MODEL_DIR = ROOT_DIR / "models"
DETECTOR_MODEL_PATH = MODEL_DIR / "bestn.engine"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "classifier.pt"
CLASSIFIER_LABELS_PATH = MODEL_DIR / "classifier_labels.json"

CLASS = "Tiger"

IMG_SIZE = 640
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
    detector = Detector(
        model_path=str(DETECTOR_MODEL_PATH),
        img_size=IMG_SIZE,
        confidence=CONFIDENCE,
        iou=IOU,
        max_detections=MAX_DETECTIONS,
        use_gpu=True,
        warmup_height=CAM_HEIGHT,
        warmup_width=CAM_WIDTH,
    )
    classifier = Classifier(
        model_path=str(CLASSIFIER_MODEL_PATH),
        label_path=str(CLASSIFIER_LABELS_PATH),
        img_size=224,
        use_gpu=True,
    )
    stabilizer = Stabilizer()
    current_stable_class = None
    logger = Logger(model_path=str(classifier.active_model_path), logs_directory=str(ROOT_DIR / "logs"), max_records=500)

    classifier.warmup()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(
        f"Device={'GPU' if use_gpu else 'CPU'} | Detector={DETECTOR_MODEL_PATH} | "
        f"Classifier={classifier.active_model_path} | IMG_SIZE={IMG_SIZE}"
    )
    print(f"Requested cam={CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS_TARGET} | Reported FPS={cap.get(cv2.CAP_PROP_FPS):.1f} | MJPG={USE_MJPG}")

    grabber = LatestFrameGrabber(cap)
    grabber.start()

    detector_in_q: queue.Queue = queue.Queue(maxsize=1)
    classifier_in_q: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    detector_lock = threading.Lock()
    classifier_lock = threading.Lock()
    latest_detector = {
        "frame_id": -1,
        "frame_time": 0.0,
        "box_xyxy": None,
        "detection_confidence": 0.0,
        "detect_ms": 0.0,
        "detect_fps": 0.0,
    }
    latest_classifier = {
        "frame_id": -1,
        "class_name": None,
        "class_confidence": 0.0,
        "classify_ms": 0.0,
        "classify_fps": 0.0,
        "box_xyxy": None,
        "frame_time": 0.0,
    }

    detector_thread = DetectorWorker(
        detector=detector,
        in_queue=detector_in_q,
        out_queue=classifier_in_q,
        stop_event=stop_event,
        lock=detector_lock,
        latest_state=latest_detector,
    )
    classifier_thread = ClassifierWorker(
        classifier=classifier,
        in_queue=classifier_in_q,
        stop_event=stop_event,
        lock=classifier_lock,
        latest_state=latest_classifier,
    )
    detector_thread.start()
    classifier_thread.start()

    prev_time = time.perf_counter()
    last_print = prev_time
    loop_fps = 0.0
    capture_fps = 0.0
    frame_idx = 0
    last_logged_frame_id = -1
    last_stabilized_frame_id = -1

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
            put_latest(
                detector_in_q,
                {
                    "frame": frame.copy(),
                    "frame_id": frame_idx,
                    "frame_time": frame_time,
                },
            )

            annotated = frame

            with detector_lock:
                detector_snapshot = dict(latest_detector)
            with classifier_lock:
                classifier_snapshot = dict(latest_classifier)

            best_box_xyxy = detector_snapshot.get("box_xyxy")
            best_detection_confidence = float(detector_snapshot.get("detection_confidence", 0.0))
            detect_ms = float(detector_snapshot.get("detect_ms", 0.0))
            detect_fps = float(detector_snapshot.get("detect_fps", 0.0))

            detected_class = classifier_snapshot.get("class_name")
            best_confidence = float(classifier_snapshot.get("class_confidence", 0.0))
            classify_ms = float(classifier_snapshot.get("classify_ms", 0.0))
            classify_fps = float(classifier_snapshot.get("classify_fps", 0.0))

            draw_box = classifier_snapshot.get("box_xyxy")
            if draw_box is None and best_box_xyxy is not None:
                draw_box = clamp_box(best_box_xyxy, frame.shape)

            if frame_idx % DRAW_EVERY_N_FRAMES == 0 and draw_box is not None and detected_class is not None:
                x1, y1, x2, y2 = draw_box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(
                    annotated,
                    f"{detected_class} cls:{best_confidence:.2f} det:{max(0.0, best_detection_confidence):.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            detected_confidence = best_confidence if detected_class is not None else 0.0
            cls_frame_id = int(classifier_snapshot.get("frame_id", -1))
            if cls_frame_id >= 0 and cls_frame_id != last_stabilized_frame_id:
                stable_class = stabilizer.update(detected_class, detected_confidence)
                current_stable_class = stable_class
                last_stabilized_frame_id = cls_frame_id
                        
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                loop_inst = 1.0 / dt
                loop_fps = (0.9 * loop_fps) + (0.1 * loop_inst) if loop_fps > 0 else loop_inst

            cv2.putText(annotated, f"Loop FPS: {loop_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Capture FPS: {capture_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Detect FPS: {detect_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 70), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Classify FPS: {classify_fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 120), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Detect ms: {detect_ms:.1f} | Classify ms: {classify_ms:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"FrameAge ms: {frame_age_ms:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Stable: {current_stable_class if current_stable_class else 'None'}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if detected_class is not None:
                is_new_frame = cls_frame_id != last_logged_frame_id
                if is_new_frame:
                    last_logged_frame_id = cls_frame_id

                logger.log_prediction(
                    detected_class,
                    loop_fps,
                    confidence=best_confidence,
                    frame_time=classifier_snapshot.get("frame_time", frame_time),
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
                        f"Loop FPS={loop_fps:.1f} | Capture FPS={capture_fps:.1f} | Detect FPS={detect_fps:.1f} | "
                        f"Classify FPS={classify_fps:.1f} | Detect ms={detect_ms:.1f} | Classify ms={classify_ms:.1f} "
                        f"| FrameAge ms={frame_age_ms:.1f}"
                    )
                    last_print = now
    finally:
        stop_event.set()
        detector_thread.join(timeout=1.0)
        classifier_thread.join(timeout=1.0)
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
