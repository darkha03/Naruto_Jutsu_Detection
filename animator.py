from pathlib import Path
import cv2
import numpy as np


class Animator:
    def __init__(self, window_name="Jutsu Animation", width=640, height=480, fps=30, animations_dir="animations"):
        self.window_name = window_name
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.frame_delay_ms = max(1, int(1000 / self.fps))
        self.animations_dir = Path(animations_dir)

        self.video_map = {
            "Fireball": "fireball.mp4",
            "Chidori": "chidori.mp4",
        }

    def register_video(self, chain_name, file_name):
        self.video_map[chain_name] = file_name

    def _normalize_name(self, chain_name):
        if chain_name is None:
            return ""
        return "".join(char.lower() for char in str(chain_name) if char.isalnum())

    def _resolve_video_path(self, chain_name):
        mapped_file = self.video_map.get(chain_name)
        if mapped_file is not None:
            candidate_path = self.animations_dir / mapped_file
            if candidate_path.exists():
                return candidate_path

        normalized_target = self._normalize_name(chain_name)
        for video_file in self.animations_dir.glob("*.mp4"):
            if self._normalize_name(video_file.stem) == normalized_target:
                return video_file

        return None

    def play(self, chain_name):
        try:
            video_path = self._resolve_video_path(chain_name)
            if video_path is None:
                return self._animate_default(chain_name)
            return self._play_video(video_path)
        finally:
            self.close()

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _show_frame(self, frame):
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(self.frame_delay_ms) & 0xFF
        return key not in (27, ord("q"))

    def _play_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                if not self._show_frame(resized):
                    return False
        finally:
            cap.release()

        return True

    def _animate_default(self, chain_name):
        for _ in range(30):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                f"No animation for: {chain_name}",
                (20, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if not self._show_frame(frame):
                return False
        return True
