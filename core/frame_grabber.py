import threading
import time


class LatestFrameGrabber:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.frame_time = 0.0
        self.capture_fps = 0.0
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        last_time = 0.0
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            now = time.perf_counter()
            if last_time > 0:
                dt = now - last_time
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.capture_fps = (0.9 * self.capture_fps) + (0.1 * instant_fps) if self.capture_fps > 0 else instant_fps
            last_time = now

            with self.lock:
                self.frame = frame
                self.frame_time = now

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0, self.capture_fps
            return self.frame.copy(), self.frame_time, self.capture_fps

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
