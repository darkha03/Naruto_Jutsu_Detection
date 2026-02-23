from collections import Counter, deque

class Stabilizer:
    def __init__(self, window_frames=20, uptime_threshold=0.5):
        self.window_frames = window_frames
        self.uptime_threshold = uptime_threshold
        self.frame_history = deque(maxlen=window_frames)
        self.last_emitted_class = None

    def update(self, class_in_frame):
        frame_class = class_in_frame.strip() if isinstance(class_in_frame, str) else None
        if frame_class == "":
            frame_class = None

        self.frame_history.append(frame_class)

        if not self.frame_history:
            return None

        frame_count = len(self.frame_history)
        class_counter = Counter()

        for class_name in self.frame_history:
            if class_name is not None:
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
        return stable_class

    
    
    def uptime(self, class_name):
        if not self.frame_history:
            return 0.0

        frame_count = len(self.frame_history)
        active_frames = sum(1 for frame_class in self.frame_history if frame_class == class_name)
        return active_frames / frame_count