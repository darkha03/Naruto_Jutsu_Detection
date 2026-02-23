from collections import Counter, deque
import cv2

class Stabilizer:
    def __init__(self, window_frames=20, uptime_threshold=0.5):
        self.window_frames = window_frames
        self.uptime_threshold = uptime_threshold
        self.frame_history = deque(maxlen=window_frames)
        self.last_emitted_class = None

    def update(self, classes_in_frame):
        frame_classes = set(classes_in_frame) if classes_in_frame else set()
        self.frame_history.append(frame_classes)

        if not self.frame_history:
            return None

        frame_count = len(self.frame_history)
        class_counter = Counter()

        for classes in self.frame_history:
            for class_name in classes:
                class_counter[class_name] += 1

        stable_class = None
        best_uptime = 0.0

        for class_name, count in class_counter.items():
            uptime = count / frame_count
            if uptime > self.uptime_threshold and uptime > best_uptime:
                best_uptime = uptime
                stable_class = class_name

        if stable_class is None:
            return None

        if stable_class == self.last_emitted_class:
            return None

        self.last_emitted_class = stable_class
        
        cv2.putText(
            None,
            f"Stabilized class: {stable_class} (uptime: {best_uptime:.2f})",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        
        return stable_class

    
    
    def uptime(self, class_name):
        if not self.frame_history:
            return 0.0

        frame_count = len(self.frame_history)
        active_frames = sum(1 for classes in self.frame_history if class_name in classes)
        return active_frames / frame_count