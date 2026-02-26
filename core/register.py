from pathlib import Path
from datetime import datetime
import time
import cv2

class Register:
    def __init__(self, image_directory="images", file_prefix="no_class", cooldown_seconds=2.0):
        image_dir = Path(image_directory)
        if not image_dir.is_absolute():
            image_dir = Path(__file__).resolve().parents[1] / image_dir
        self.image_dir = image_dir
        self.image_dir.mkdir(exist_ok=True)
        self.file_prefix = file_prefix
        self.cooldown_seconds = cooldown_seconds
        self.last_saved_at = None
    
    def update(self, frame, class_in_frame):
        has_missing_name = class_in_frame is None or str(class_in_frame).strip() == ""

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