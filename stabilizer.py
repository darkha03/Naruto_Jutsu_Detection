from collections import deque

class Stabilizer:
    def __init__(
        self,
        enter_point=2.5,
        confirm_threshold=6.0,
        exit_point=1.5,
        queue_size=12,
        confirm_consecutive_frames=3,
        max_unconfirmed_frames=100,
    ):
        if not (exit_point <= enter_point <= confirm_threshold):
            raise ValueError("Expected exit_point <= enter_point <= confirm_threshold")
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")
        if confirm_consecutive_frames <= 0:
            raise ValueError("confirm_consecutive_frames must be > 0")
        if max_unconfirmed_frames <= 0:
            raise ValueError("max_unconfirmed_frames must be > 0")

        self.enter_point = float(enter_point)
        self.confirm_threshold = float(confirm_threshold)
        self.exit_point = float(exit_point)
        self.confirm_consecutive_frames = int(confirm_consecutive_frames)
        self.max_unconfirmed_frames = int(max_unconfirmed_frames)
        self.frame_queue = deque(maxlen=int(queue_size))
        self.frame_index = 0

        self.candidate_class = None
        self.candidate_score = 0.0
        self.last_emitted_class = None
        self.last_confirmed_frame_index = None

    def _normalize_class(self, class_in_frame):
        if isinstance(class_in_frame, str):
            normalized = class_in_frame.strip()
            return normalized if normalized else None
        return None

    def _parse_confidence(self, confidence):
        try:
            return float(confidence)
        except (TypeError, ValueError):
            return 0.0

    def _enqueue_frame(self, frame_class, score):
        self.frame_index += 1
        self.frame_queue.append((self.frame_index, frame_class, score))

    def _class_scores(self):
        scores = {}
        for _, frame_class, score in self.frame_queue:
            if frame_class is None:
                continue
            scores[frame_class] = scores.get(frame_class, 0.0) + score
        return scores

    def _promote_from_queue(self, class_scores, excluded_class=None):
        best_class = None
        best_score = float("-inf")

        for frame_class, score in class_scores.items():
            if excluded_class is not None and frame_class == excluded_class:
                continue
            if score >= self.enter_point and score > best_score:
                best_class = frame_class
                best_score = score

        if best_class is not None:
            self.candidate_class = best_class
            self.candidate_score = best_score
            return True

        self.candidate_class = None
        self.candidate_score = 0.0
        return False

    def _consecutive_trailing_frames(self, class_name):
        if class_name is None:
            return 0

        consecutive = 0
        for _, frame_class, _ in reversed(self.frame_queue):
            if frame_class == class_name:
                consecutive += 1
            else:
                break
        return consecutive

    def _expire_last_emitted_if_stale(self):
        if self.last_emitted_class is None or self.last_confirmed_frame_index is None:
            return

        frames_since_confirm = self.frame_index - self.last_confirmed_frame_index
        if frames_since_confirm >= self.max_unconfirmed_frames:
            self.last_emitted_class = None
            self.last_confirmed_frame_index = None

    def update(self, class_in_frame, confidence=0.0):
        frame_class = self._normalize_class(class_in_frame)
        score = self._parse_confidence(confidence)

        if frame_class is None:
            score = 0.0

        self._enqueue_frame(frame_class, score)
        class_scores = self._class_scores()

        if self.candidate_class is None:
            self._promote_from_queue(class_scores)
        else:
            self.candidate_score = class_scores.get(self.candidate_class, 0.0)

        if self.candidate_class is None:
            self._expire_last_emitted_if_stale()
            return self.last_emitted_class

        if self.candidate_score < self.exit_point:
            rejected_class = self.candidate_class
            self.candidate_class = None
            self.candidate_score = 0.0
            # Try to promote another class from the queue, excluding the recently rejected class
            self._promote_from_queue(class_scores, excluded_class=rejected_class)

        consecutive_frames = self._consecutive_trailing_frames(self.candidate_class)
        has_confirmed_class = (
            self.candidate_score >= self.confirm_threshold
            and consecutive_frames >= self.confirm_consecutive_frames
        )

        if has_confirmed_class:
            self.last_confirmed_frame_index = self.frame_index
            if self.candidate_class != self.last_emitted_class:
                self.last_emitted_class = self.candidate_class
                return self.candidate_class
            return self.last_emitted_class

        self._expire_last_emitted_if_stale()

        return self.last_emitted_class
