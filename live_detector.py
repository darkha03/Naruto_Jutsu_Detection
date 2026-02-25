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
        model_path='bests.engine',
        default_class="Tiger",
        img_size=320,
        confidence=0.5,
        iou=0.5,
        cam_width=640,
        cam_height=480,
        cam_fps_target=60,
        max_detections=1,
        flush_every_n_logs=10,
    ):
        self.model_path = model_path
        self.default_class = default_class
        self.img_size = img_size
        self.confidence = confidence
        self.iou = iou
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_fps_target = cam_fps_target
        self.max_detections = max_detections
        self.flush_every_n_logs = flush_every_n_logs

        self.detector = Detector(
            model_path=self.model_path,
            img_size=self.img_size,
            confidence=self.confidence,
            iou=self.iou,
            max_detections=self.max_detections,
            use_gpu=True,
            warmup_height=self.cam_height,
            warmup_width=self.cam_width,
        )
        self.use_gpu = self.detector.use_gpu
        self.annotator = FrameAnnotator()
        self.animator = Animator(width=self.cam_width, height=self.cam_height)
        self.stabilizer = Stabilizer(enter_point=2.50, confirm_threshold=12.0, exit_point=1.5, queue_size=24)
        self.chainer = Chainer()
        self.logger = Logger(model_path=self.model_path, logs_directory="logs", max_records=500)

        self.current_stable_class = None
        self.current_sequence = []
        self.fps = 0.0
        self.pending_logs = 0
        self.capture_fps = 0.0
        self.prev_time = time.perf_counter()
        self.last_fps_print = time.perf_counter()

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
        actual_cam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera target FPS: {self.cam_fps_target} | Camera reported FPS: {actual_cam_fps:.2f}")

        try:
            while self.cap.isOpened():
                frame, _frame_time, threaded_capture_fps = self.grabber.read_latest()
                if frame is None:
                    time.sleep(0.001)
                    continue

                self.capture_fps = threaded_capture_fps
                frame = cv2.flip(frame, 1)

                detection_info = self.detector.predict(frame)
                detected_class = detection_info["class_name"]
                best_confidence = detection_info["confidence"]
                self.annotator.draw_detection(frame, detection_info)

                if detected_class is not None:
                    did_log = self.logger.log_prediction(detected_class, self.fps, confidence=best_confidence)
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
        finally:
            self._cleanup()
            self._print_summary()


if __name__ == "__main__":
    detector = LiveDetector()
    detector.run()