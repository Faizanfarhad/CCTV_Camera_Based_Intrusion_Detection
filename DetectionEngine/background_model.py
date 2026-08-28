"""Person-only detection on top of YOLOv8 (the "background model" stage).

Pipeline role (see what_to_do.txt):
    1. A frame is captured.
    2. MOG2 background subtraction flags pixel-level motion (motion_detection.py).
    3. When motion is present, this model runs YOLOv8n filtered to the COCO
       "person" class (id 0) to confirm the moving object is actually a human.
    4. For each person we return the bounding box, confidence and the box
       centroid so the Zone/ROI module can run its point-in-polygon check.
"""

from ultralytics import YOLO

import cv2


class BGDetect:
    """Detect people (and only people) in a frame using YOLOv8n.

    Usage:
        detector = BGDetect()
        detections, annotated = detector(frame)

    ``detections`` is a list of dicts with keys ``bbox``, ``confidence``,
    ``class`` and ``centroid``.
    """

    PERSON_CLASS = 0  # COCO class id for "person"

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        super().__init__()
        self.model = YOLO(model_path, task="detect")
        self.conf_threshold = conf_threshold

    @staticmethod
    def draw_border(
        img,
        top_left,
        bottom_right,
        color=(0, 255, 0),
        thickness=10,
        line_length_x=200,
        line_length_y=200,
    ):
        """Draw corner brackets around a bounding box."""
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Keep the bracket strokes inside the box (and the frame).
        line_length_x = min(line_length_x, x2 - x1)
        line_length_y = min(line_length_y, y2 - y1)

        cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
        cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

        cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
        cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

        cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
        cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

        cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
        cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

        return img

    def bgmodel(self, frame):
        """Run person-only detection on ``frame`` and annotate a copy of it.

        Returns:
            tuple: ``(detections, annotated_frame)``.
        """
        results = self.model(frame, classes=[self.PERSON_CLASS], verbose=False)

        detections = []
        annotated = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                if conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(conf),
                        "class": int(cls_id),
                        "centroid": centroid,
                    }
                )

                annotated = self.draw_border(annotated, (x1, y1), (x2, y2))
                annotated = self._draw_label(annotated, f"person {conf:.2f}", x1, y1)

        return detections, annotated

    @staticmethod
    def _draw_label(img, text, x, y, color=(0, 255, 0)):
        """Draw a small label tag above the box whose top-left corner is (x, y)."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tag_y1 = max(0, y - th - 6)
        cv2.rectangle(img, (x, tag_y1), (x + tw + 4, y), color, -1)
        cv2.putText(
            img,
            text,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        return img

    def __call__(self, frame):
        """Convenience alias for :meth:`bgmodel`."""
        return self.bgmodel(frame)


def background_model(model_path: str = "yolov8n.pt", conf_threshold: float = 0.25) -> BGDetect:
    """Factory used by the ingestion pipeline to create one detector instance."""
    return BGDetect(model_path=model_path, conf_threshold=conf_threshold)


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "testVideo/video.mp4"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        raise SystemExit(1)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read a frame.")
        raise SystemExit(1)

    detector = background_model()
    detections, annotated = detector(frame)

    print(f"Detected {len(detections)} person(s):")
    for det in detections:
        cx, cy = det["centroid"]
        print(
            f"  bbox={det['bbox']} conf={det['confidence']:.3f} "
            f"centroid=({cx:.1f}, {cy:.1f})"
        )

    out = "/tmp/person_detection.jpg"
    cv2.imwrite(out, annotated)
    print(f"Annotated frame saved to: {out}")
