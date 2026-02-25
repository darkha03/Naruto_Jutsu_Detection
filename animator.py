import time
import cv2
import numpy as np


class Animator:
    def __init__(self, window_name="Jutsu Animation", width=640, height=480, fps=30):
        self.window_name = window_name
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.frame_delay_ms = max(1, int(1000 / self.fps))

        self.animations = {
            "Fireball": self._animate_fireball,
            "Water Dragon": self._animate_water_dragon,
        }

    def register_animation(self, chain_name, animation_callable):
        self.animations[chain_name] = animation_callable

    def play(self, chain_name):
        animation_callable = self.animations.get(chain_name, self._animate_default)
        return animation_callable(chain_name)

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _render_label(self, frame, chain_name):
        cv2.putText(
            frame,
            chain_name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _show_frame(self, frame):
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(self.frame_delay_ms) & 0xFF
        return key not in (27, ord("q"))

    def _animate_fireball(self, chain_name):
        center_x = self.width // 2
        center_y = self.height // 2

        for step in range(45):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            radius = 10 + step * 5
            glow_radius = radius + 18

            cv2.circle(frame, (center_x, center_y), glow_radius, (0, 90, 255), -1)
            cv2.circle(frame, (center_x, center_y), radius, (0, 170, 255), -1)
            self._render_label(frame, chain_name)

            if not self._show_frame(frame):
                return False
        return True

    def _animate_water_dragon(self, chain_name):
        x_positions = np.linspace(40, self.width - 40, 80)

        for step in range(70):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            amplitude = 35
            phase = step * 0.2

            points = []
            for x in x_positions:
                y = self.height // 2 + int(amplitude * np.sin((x / 40.0) + phase))
                points.append([int(x), int(y)])

            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=(255, 170, 0), thickness=8)
            self._render_label(frame, chain_name)

            if not self._show_frame(frame):
                return False
        return True

    def _animate_default(self, chain_name):
        for step in range(30):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            alpha = 80 + (step % 15) * 10

            cv2.rectangle(frame, (0, 0), (self.width, self.height), (alpha // 4, alpha // 4, alpha // 4), -1)
            self._render_label(frame, chain_name)

            if not self._show_frame(frame):
                return False
        return True
