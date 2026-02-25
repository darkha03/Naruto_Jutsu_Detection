import cv2


class FrameAnnotator:
    def draw_detection(self, frame, detection_info):
        detected_class = detection_info["class_name"]
        best_confidence = detection_info["confidence"]
        best_box_xyxy = detection_info["box_xyxy"]

        if best_box_xyxy is not None and detected_class is not None:
            x1, y1, x2, y2 = map(int, best_box_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                frame,
                f"{detected_class} {best_confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

    def draw_stats(self, frame, loop_fps, capture_fps, stable_class):
        cv2.putText(
            frame,
            f"Loop FPS: {loop_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Capture FPS: {capture_fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

        stable_text = stable_class if stable_class is not None else "None"
        cv2.putText(
            frame,
            f"Stable: {stable_text}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
