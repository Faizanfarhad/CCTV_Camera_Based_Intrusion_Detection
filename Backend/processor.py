"""Headless per-camera stream processor.

Reuses the project's existing detection pipeline (``VideoIngestion`` and
``DetectionEngine``) but without any OpenCV GUI / mouse callbacks:

    * ``VideoIngestion.denoise.denoise_frame``
    * ``DetectionEngine.motion_detection.foreground_model``  (MOG2)
    * ``DetectionEngine.background_model.background_model``  (YOLOv8 person)
    * ``DetectionEngine.visualize_polyogn.extract_roi``

Each processor runs in its own daemon thread, encodes an annotated JPEG for the
MJPEG live-feed endpoint and invokes ``on_alert`` when a detected person enters
a configured zone.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from Backend.config import SNAPSHOT_DIR
from VideoIngestion.denoise import denoise_frame
from DetectionEngine.motion_detection import foreground_model
from DetectionEngine.visualize_polyogn import extract_roi


def point_in_polygon(pt, poly) -> bool:
    """Ray-casting point-in-polygon test (``pt`` = ``(x, y)``)."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    h = value.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


class StreamProcessor(threading.Thread):
    def __init__(
        self,
        camera: dict,
        get_zones,
        on_alert,
        bg_detector,
        bg_lock: threading.Lock,
        config,
    ):
        super().__init__(daemon=True, name=f"stream-{camera['id']}")
        self.camera = camera
        self.get_zones = get_zones
        self.on_alert = on_alert
        self.bg_detector = bg_detector
        self.bg_lock = bg_lock

        self.process_width = getattr(config, "PROCESS_WIDTH", 960)
        self.detect_every = max(1, int(getattr(config, "DETECT_EVERY", 3)))
        self.person_conf = float(getattr(config, "PERSON_CONF_THRESHOLD", 0.35))
        self.anomaly_threshold = float(getattr(config, "ANOMALY_THRESHOLD", 0.02))
        self.alert_debounce = float(getattr(config, "ALERT_DEBOUNCE_SECONDS", 30))
        self.target_fps = float(getattr(config, "STREAM_FPS", 12))
        self.jpeg_quality = int(getattr(config, "JPEG_QUALITY", 70))

        self._stop = threading.Event()
        self._frame_cond = threading.Condition()
        self._seq = 0
        self._latest_jpeg = None
        self._last_alert = {}  # zone_id -> monotonic timestamp
        self.fps = 0.0

        self.motion = foreground_model(var_threshold=20, min_threshold=200)

    # ------------------------------------------------------------------ #
    # Capture helpers
    # ------------------------------------------------------------------ #
    def _open_capture(self):
        source = self.camera.get("source")
        if self.camera.get("kind") == "camera":
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(str(source))
        return cap if cap.isOpened() else None

    @staticmethod
    def _resize(frame, process_width):
        h, w = frame.shape[:2]
        if w > process_width:
            scale = process_width / float(w)
            frame = cv2.resize(frame, (process_width, int(h * scale)))
        return frame

    @staticmethod
    def _zone_to_pixels(pts, w, h):
        poly = []
        for px, py in pts:
            x = max(0, min(w - 1, int(round(px / 100.0 * w))))
            y = max(0, min(h - 1, int(round(py / 60.0 * h))))
            poly.append((x, y))
        return poly

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_zone(self, frame, poly, color_hex):
        color = hex_to_bgr(color_hex)
        pts = np.array(poly, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2)

    def _draw_status(self, frame, person_count, zone_count):
        h = frame.shape[0]
        cv2.putText(
            frame,
            f"Zones {zone_count} | Persons {person_count} | {self.fps:.1f} FPS",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            self.camera["name"],
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # ------------------------------------------------------------------ #
    # Detection / frame logic
    # ------------------------------------------------------------------ #
    def _motion_score(self, motion_mask, poly):
        """Fraction of moving pixels inside ``poly`` using the shared ROI crop."""
        try:
            roi, _ = extract_roi(motion_mask, poly)
        except Exception:
            return 0.0
        if roi is None or roi.size == 0:
            return 0.0
        return float((roi > 0).sum()) / max(1, roi.size)

    def _save_snapshot(self, frame, zone, conf):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = SNAPSHOT_DIR / f"{self.camera['id']}_{zone['id']}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    def _set_frame(self, jpeg_bytes):
        with self._frame_cond:
            self._seq += 1
            self._latest_jpeg = jpeg_bytes
            self._frame_cond.notify_all()

    def wait_for_frame(self, timeout=1.0):
        with self._frame_cond:
            self._frame_cond.wait(timeout)
            return self._seq, self._latest_jpeg

    def stop(self):
        self._stop.set()


    def run(self):
        cap = self._open_capture()
        if cap is None:
            return

        frame_idx = 0
        detections = []
        try:
            while not self._stop.is_set():
                started = time.time()
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    if self.camera.get("kind") == "camera":
                        time.sleep(2)  # brief reconnect back-off
                        cap = self._open_capture()
                        if cap is None:
                            break
                        continue
                    # Loop recorded files so the live view never goes dark.
                    cap = self._open_capture()
                    if cap is None:
                        break
                    continue

                frame = self._resize(frame, self.process_width)
                denoised = denoise_frame(
                    frame,
                    method=self.camera.get("denoise", "fast"),
                    strength=int(self.camera.get("denoise_strength", 10)),
                )
                motion_mask = self.motion.apply(denoised)

                zones = self.get_zones() or []
                h, w = denoised.shape[:2]

                # Refresh YOLO detections periodically; reuse them in between
                # so the stream stays smooth while inference is happening.
                if frame_idx % self.detect_every == 0:
                    with self.bg_lock:
                        detections, _ = self.bg_detector(denoised)
                    detections = [
                        d for d in detections if d["confidence"] >= self.person_conf
                    ]

                display = denoised.copy()
                pixel_zones = []
                for z in zones:
                    poly = self._zone_to_pixels(z["pts"], w, h)
                    if len(poly) >= 3:
                        pixel_zones.append((z, poly))
                        self._draw_zone(display, poly, z["color"])


                # Draw detected persons and figure out which zone (if any) each
                # person's centroid is inside.
                intrusions = []
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    cx, cy = det["centroid"]
                    hit_zone = None
                    for z, poly in pixel_zones:
                        if point_in_polygon((cx, cy), poly):
                            hit_zone = z
                            break
                    color = (0, 0, 255) if hit_zone else (0, 255, 0)
                    self.bg_detector.draw_border(
                        display, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=3
                    )
                    self.bg_detector._draw_label(
                        display, f"person {det['confidence']:.2f}", int(x1), int(y1), color=color
                    )
                    if hit_zone:
                        intrusions.append((det, hit_zone))

                # Emit debounced alerts for person-in-zone intrusions.
                now = time.time()
                for det, z in intrusions:
                    last = self._last_alert.get(z["id"], 0.0)
                    if now - last >= self.alert_debounce:
                        self._last_alert[z["id"]] = now
                        snapshot_path = self._save_snapshot(display, z, det["confidence"])
                        self.on_alert(
                            self.camera,
                            z,
                            det["confidence"],
                            "intrusion",
                            snapshot_path,
                        )

                # Compute per-zone anomaly score for the on-screen summary.
                for z, poly in pixel_zones:
                    score = self._motion_score(motion_mask, poly)
                    first = poly[0]
                    label = f"{z['name']} {score:.3f}"
                    cv2.putText(
                        display,
                        label,
                        (first[0], max(14, first[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        hex_to_bgr(z["color"]),
                        1,
                        cv2.LINE_AA,
                    )

                self._draw_status(display, len(detections), len(pixel_zones))

                ok, buf = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if ok:
                    self._set_frame(buf.tobytes())

                self.fps = 1.0 / max(0.001, time.time() - started)
                frame_idx += 1
                elapsed = time.time() - started
                time.sleep(max(0.0, 1.0 / self.target_fps - elapsed))
        finally:
            cap.release()



class StreamManager:
    """Lazily starts/stops one :class:`StreamProcessor` per camera."""

    def __init__(self, get_zones, on_alert, config):
        self.get_zones = get_zones
        self.on_alert = on_alert
        self.config = config
        self.processors = {}
        self._lock = threading.Lock()
        self.bg_lock = threading.Lock()
        self.bg_detector = None

    def _get_bg_detector(self):
        if self.bg_detector is None:
            from DetectionEngine.background_model import background_model

            self.bg_detector = background_model()
        return self.bg_detector

    def ensure(self, camera):
        with self._lock:
            proc = self.processors.get(camera["id"])
            if proc is not None and proc.is_alive():
                return proc
            proc = StreamProcessor(
                camera,
                self.get_zones,
                self.on_alert,
                self._get_bg_detector(),
                self.bg_lock,
                self.config,
            )
            self.processors[camera["id"]] = proc
            proc.start()
            return proc

    def stop(self, camera_id):
        with self._lock:
            proc = self.processors.pop(camera_id, None)
        if proc is not None:
            proc.stop()

    def stop_all(self):
        with self._lock:
            procs = list(self.processors.values())
            self.processors.clear()
        for proc in procs:
            proc.stop()

    def running_ids(self):
        with self._lock:
            return {cid for cid, p in self.processors.items() if p.is_alive()}

