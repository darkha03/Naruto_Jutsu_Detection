from pathlib import Path
from datetime import datetime
import time
import cv2

class Register:
    def __init__(self, image_directory="images", file_prefix="no_class", cooldown_seconds=1.0):
        self.image_dir = Path(image_directory)
        self.image_dir.mkdir(exist_ok=True)
        self.file_prefix = file_prefix
        self.cooldown_seconds = cooldown_seconds
        self.last_saved_at = None
    
    def update(self, frame, classes_in_frame):
        classes = list(classes_in_frame) if classes_in_frame is not None else []

        has_missing_name = len(classes) == 0 or any(
            class_name is None or str(class_name).strip() == ""
            for class_name in classes
        )

        if not has_missing_name:
            return None

        current_time = time.monotonic()
        if self.last_saved_at is not None:
            elapsed = current_time - self.last_saved_at
            if elapsed < self.cooldown_seconds:
                return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = self.image_dir / f"{self.file_prefix}_{timestamp}.jpg"

        if cv2.imwrite(str(image_path), frame):
            self.last_saved_at = current_time
            return image_path

        return None