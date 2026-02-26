import cv2
import time
from animator import Animator
from chainer import Chainer
from detector import Detector
from frame_annotator import FrameAnnotator
from stabilizer import Stabilizer
from logger import Logger
from frame_grabber import LatestFrameGrabber

class LiveDetector:
    def __init__(
        self,
        model_path='best.engine',
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
        self.model_path = model_path
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
            fallback_model_path="bests.pt",
        )
        self.use_gpu = self.detector.use_gpu
        self.annotator = FrameAnnotator()
        self.animator = Animator(width=self.cam_width, height=self.cam_height)
        self.stabilizer = Stabilizer(enter_point=3.0, confirm_threshold=8.0, exit_point=2.0, queue_size=16)
        self.chainer = Chainer()
        self.logger = Logger(model_path=self.model_path, logs_directory="logs", max_records=500)

        self.current_stable_class = None
        self.current_sequence = []
        self.fps = 0.0
        self.pending_logs = 0
        self.capture_fps = 0.0
        self.prev_time = time.perf_counter()
        self.last_fps_print = time.perf_counter()
        self.last_seen_frame_time = None
        self.last_inferred_frame_time = None
        self.last_detection_info = {
            "has_detection": False,
            "class_name": None,
            "confidence": 0.0,
            "box_xyxy": None,
            "raw_result": None,
        }

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
        print(f"Model in use: {self.detector.active_model_path}")
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

                should_infer = (not self.infer_on_new_frame_only) or is_new_frame
                if should_infer:
                    detection_info = self.detector.predict(frame)
                    self.last_detection_info = detection_info
                    self.last_inferred_frame_time = frame_time
                else:
                    detection_info = self.last_detection_info

                detected_class = detection_info["class_name"]
                best_confidence = detection_info["confidence"]
                self.annotator.draw_detection(frame, detection_info)

                did_log = self.logger.log_prediction(
                    detected_class,
                    self.fps,
                    confidence=best_confidence,
                    frame_time=self.last_inferred_frame_time,
                    is_new_frame=is_new_frame,
                )
                if did_log:
                    self.pending_logs += 1

                detected_confidence = best_confidence if detected_class is not None else 0.0
                stable_class = self.stabilizer.update(detected_class, detected_confidence)
                self.current_stable_class = stable_class
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