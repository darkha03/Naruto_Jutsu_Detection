import cv2
import sys
import time
import queue
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.animator import Animator
from core.chainer import Chainer
from core.classifier import Classifier
from core.detector import Detector
from core.frame_annotator import FrameAnnotator
from core.stabilizer import Stabilizer
from core.logger import Logger
from core.frame_grabber import LatestFrameGrabber
from core.pipeline_workers import DetectorWorker, ClassifierWorker, put_latest, clamp_box


MODEL_DIR = ROOT_DIR / "models"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "classifier.pt"
CLASSIFIER_LABELS_PATH = MODEL_DIR / "classifier_labels.json"

class LiveDetector:
    def __init__(
        self,
        model_path=None,
        classifier_model_path=None,
        classifier_labels_path=None,
        default_class="Tiger",
        img_size=640,
        confidence=0.5,
        iou=0.5,
        cam_width=640,
        cam_height=480,
        cam_fps_target=60,
        loop_fps_target=60,
        max_detections=1,
        flush_every_n_logs=10,
        infer_on_new_frame_only=True,
    ):
        default_model_path = MODEL_DIR / "bestn.engine"
        fallback_model_path = MODEL_DIR / "bestn.pt"

        self.model_path = str(Path(model_path)) if model_path else str(default_model_path)
        self.classifier_model_path = (
            str(Path(classifier_model_path)) if classifier_model_path else str(CLASSIFIER_MODEL_PATH)
        )
        self.classifier_labels_path = (
            str(Path(classifier_labels_path)) if classifier_labels_path else str(CLASSIFIER_LABELS_PATH)
        )
        self.default_class = default_class
        self.img_size = img_size
        self.confidence = confidence
        self.iou = iou
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_fps_target = cam_fps_target
        self.loop_fps_target = loop_fps_target
        self.loop_frame_budget = (1.0 / loop_fps_target) if loop_fps_target and loop_fps_target > 0 else 0.0
        self.max_detections = max_detections
        self.flush_every_n_logs = flush_every_n_logs
        self.infer_on_new_frame_only = infer_on_new_frame_only

        self.detector = Detector(
            model_path=self.model_path,
            img_size=self.img_size,
            confidence=self.confidence,
            iou=self.iou,
            max_detections=self.max_detections,
            use_gpu=True,
            warmup_height=self.cam_height,
            warmup_width=self.cam_width,
            fallback_model_path=str(fallback_model_path),
        )
        self.classifier = Classifier(
            model_path=self.classifier_model_path,
            label_path=self.classifier_labels_path,
            img_size=224,
            use_gpu=True,
        )
        self.classifier.warmup()

        self.use_gpu = self.detector.use_gpu
        self.annotator = FrameAnnotator()
        self.animator = Animator(
            width=self.cam_width,
            height=self.cam_height,
            animations_dir=str(ROOT_DIR / "animations"),
        )
        self.stabilizer = Stabilizer(enter_point=3.0, confirm_threshold=8.0, exit_point=2.0, queue_size=16)
        self.chainer = Chainer()
        self.logger = Logger(
            model_path=str(self.classifier.active_model_path),
            logs_directory=str(ROOT_DIR / "logs"),
            max_records=500,
        )

        self.current_stable_class = None
        self.current_sequence = []
        self.fps = 0.0
        self.pending_logs = 0
        self.capture_fps = 0.0
        self.prev_time = time.perf_counter()
        self.last_fps_print = time.perf_counter()
        self.last_seen_frame_time = None
        self.frame_id_seq = 0
        self.last_logged_frame_id = -1
        self.last_stabilized_frame_id = -1

        self.detector_in_q: queue.Queue = queue.Queue(maxsize=1)
        self.classifier_in_q: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        self.detector_lock = threading.Lock()
        self.classifier_lock = threading.Lock()
        self.latest_detector = {
            "frame_id": -1,
            "frame_time": 0.0,
            "box_xyxy": None,
            "detection_confidence": 0.0,
            "detect_ms": 0.0,
            "detect_fps": 0.0,
        }
        self.latest_classifier = {
            "frame_id": -1,
            "class_name": None,
            "class_confidence": 0.0,
            "classify_ms": 0.0,
            "classify_fps": 0.0,
            "box_xyxy": None,
            "frame_time": 0.0,
        }

        self.detector_thread = DetectorWorker(
            detector=self.detector,
            in_queue=self.detector_in_q,
            out_queue=self.classifier_in_q,
            stop_event=self.stop_event,
            lock=self.detector_lock,
            latest_state=self.latest_detector,
        )
        self.classifier_thread = ClassifierWorker(
            classifier=self.classifier,
            in_queue=self.classifier_in_q,
            stop_event=self.stop_event,
            lock=self.classifier_lock,
            latest_state=self.latest_classifier,
        )

        self.cap = None
        self.grabber = None

    def _open_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cam_fps_target)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabber = LatestFrameGrabber(self.cap)
        self.grabber.start()
        self.detector_thread.start()
        self.classifier_thread.start()

    def _update_loop_fps(self):
        current_time = time.perf_counter()
        delta = current_time - self.prev_time
        self.prev_time = current_time

        if delta > 0:
            instant_fps = 1.0 / delta
            self.fps = (0.9 * self.fps) + (0.1 * instant_fps) if self.fps > 0 else instant_fps

        now_print = time.perf_counter()
        if now_print - self.last_fps_print >= 1.0:
            #print(f"Loop FPS={self.fps:.1f} | Capture FPS={self.capture_fps:.1f}")
            self.last_fps_print = now_print

    def _cleanup(self):
        self.stop_event.set()
        if self.detector_thread.is_alive():
            self.detector_thread.join(timeout=1.0)
        if self.classifier_thread.is_alive():
            self.classifier_thread.join(timeout=1.0)
        if self.grabber is not None:
            self.grabber.stop()
        if self.cap is not None:
            self.cap.release()
        self.animator.close()
        cv2.destroyAllWindows()
        self.logger.flush()
        self.logger.close()

    def _print_summary(self):
        precision, correct_predictions, total_predictions = self.logger.calculate_precision(self.default_class)
        summary_file_path, decision = self.logger.save_run_decision(self.default_class, precision)

        print(f"Default class: {self.default_class}")
        print(f"Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"Precision: {precision:.4f}")
        print(f"Decision: {decision}")
        print(f"Saved summary to: {summary_file_path}")
        print(f"Current stable class at end of run: {self.current_stable_class}")

    def run(self):
        self._open_camera()

        print("Sharingan Activated! Looking for Jutsus... Press 'q' to quit.")
        print(f"Logging predictions to: {self.logger.log_file_path}")
        print(
            f"Device: {'GPU' if self.use_gpu else 'CPU'} | "
            f"imgsz={self.img_size} | cam={self.cam_width}x{self.cam_height}"
        )
        print(f"Loop FPS target: {self.loop_fps_target}")
        print(f"Detector model in use: {self.detector.active_model_path}")
        print(f"Classifier model in use: {self.classifier.active_model_path}")
        actual_cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera target FPS: {self.cam_fps_target} | Camera reported FPS: {actual_cam_fps:.2f}")

        try:
            while self.cap.isOpened():
                loop_start = time.perf_counter()

                frame, frame_time, threaded_capture_fps = self.grabber.read_latest()
                if frame is None:
                    time.sleep(0.001)
                    continue

                is_new_frame = (self.last_seen_frame_time is None) or (frame_time != self.last_seen_frame_time)
                if is_new_frame:
                    self.last_seen_frame_time = frame_time

                self.capture_fps = threaded_capture_fps
                frame = cv2.flip(frame, 1)
                self.frame_id_seq += 1
                frame_id = self.frame_id_seq

                should_infer = (not self.infer_on_new_frame_only) or is_new_frame
                if should_infer:
                    put_latest(
                        self.detector_in_q,
                        {
                            "frame": frame.copy(),
                            "frame_id": frame_id,
                            "frame_time": frame_time,
                        },
                    )

                with self.detector_lock:
                    detector_snapshot = dict(self.latest_detector)
                with self.classifier_lock:
                    classifier_snapshot = dict(self.latest_classifier)

                detected_class = classifier_snapshot.get("class_name")
                best_confidence = float(classifier_snapshot.get("class_confidence", 0.0))

                draw_box = classifier_snapshot.get("box_xyxy")
                if draw_box is None:
                    raw_box = detector_snapshot.get("box_xyxy")
                    if raw_box is not None:
                        draw_box = clamp_box(raw_box, frame.shape)

                detection_info = {
                    "has_detection": draw_box is not None and detected_class is not None,
                    "class_name": detected_class,
                    "confidence": best_confidence,
                    "box_xyxy": draw_box,
                    "raw_result": None,
                }
                self.annotator.draw_detection(frame, detection_info)

                cls_frame_id = int(classifier_snapshot.get("frame_id", -1))
                did_log = False
                if detected_class is not None:
                    did_log = self.logger.log_prediction(
                        detected_class,
                        self.fps,
                        confidence=best_confidence,
                        frame_time=classifier_snapshot.get("frame_time", frame_time),
                        is_new_frame=(cls_frame_id != self.last_logged_frame_id),
                    )
                    if did_log:
                        self.pending_logs += 1
                        self.last_logged_frame_id = cls_frame_id

                detected_confidence = best_confidence if detected_class is not None else 0.0
                if cls_frame_id >= 0 and cls_frame_id != self.last_stabilized_frame_id:
                    stable_class = self.stabilizer.update(detected_class, detected_confidence)
                    self.current_stable_class = stable_class
                    self.last_stabilized_frame_id = cls_frame_id
                chain_info = self.chainer.update(self.current_stable_class)
                self.current_sequence = chain_info["active_sequence"]

                if chain_info["completed_sequence"] is not None:
                    print(f"Completed sequence: {chain_info['completed_sequence']}")

                if chain_info["is_chain_completed"] and chain_info["completed_chain_name"] is not None:
                    completed_chain_name = chain_info["completed_chain_name"]
                    print(f"Chain completed: {completed_chain_name}")
                    self.animator.play(completed_chain_name)
                    self.chainer.clear()
                    self.current_sequence = []

                if self.pending_logs >= self.flush_every_n_logs:
                    self.logger.flush()
                    self.pending_logs = 0

                self._update_loop_fps()
                self.annotator.draw_stats(
                    frame,
                    loop_fps=self.fps,
                    capture_fps=self.capture_fps,
                    stable_class=self.current_stable_class,
                )

                cv2.imshow('Jutsu Detector - YOLOv8', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if self.loop_frame_budget > 0:
                    elapsed = time.perf_counter() - loop_start
                    remaining = self.loop_frame_budget - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self._cleanup()
            self._print_summary()


if __name__ == "__main__":
    detector = LiveDetector()
    detector.run()