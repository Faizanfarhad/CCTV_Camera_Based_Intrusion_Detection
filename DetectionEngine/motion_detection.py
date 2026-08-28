import cv2
import numpy as np

class MotionDetector:
    """Persistent foreground detector based on MOG2."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = True,
        min_threshold: int = 200,
    ):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_threshold = min_threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return a cleaned binary foreground mask for the input frame."""
        fg_mask = self.subtractor.apply(frame)

        # Remove MOG2 shadow pixels and keep only strong foreground motion.
        _, fg_mask = cv2.threshold(
            fg_mask, self.min_threshold, 255, cv2.THRESH_BINARY
        )

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, self.kernel)
        return fg_mask

    def apply_with_color(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return both the motion mask and a color frame containing only motion."""
        fg_mask = self.apply(frame)
        motion_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
        return fg_mask, motion_frame

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.apply_with_color(frame)


def foreground_model(history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = True,
        min_threshold: int = 200) -> MotionDetector:
    """Factory used by the ingestion pipeline to create one detector instance."""
    return MotionDetector(history,var_threshold,detect_shadows,min_threshold)
