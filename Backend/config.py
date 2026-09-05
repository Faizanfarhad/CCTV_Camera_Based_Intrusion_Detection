"""Central configuration for the Sentinel CCTV backend.

The dashboard draws one ROI in a normalized 100x60 coordinate space. The
backend stores those same normalized points and converts them to pixel
coordinates at detection time (``px/100 * width``, ``py/60 * height``).
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Persistence.
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "sentinel.db"
SNAPSHOT_DIR = BASE_DIR / "saved_videos" / "snapshots"

# Dashboard entry point served by FastAPI at "/".
DASHBOARD_PATH = BASE_DIR / "index.html"

# YOLO weights. Ultralytics downloads ``yolov8n.pt`` on first use when it is
# not already present. Set YOLO_MODEL_PATH on Render to a mounted/custom model.
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

# Single camera source. ``kind`` is either "video" (path relative to BASE_DIR)
# or "camera" (OpenCV device index).
CAMERA_SOURCE = os.getenv(
    "CAMERA_SOURCE",
    str(BASE_DIR / "testVideo" / "hit_and_run.mp4"),
)

SOURCES = [
    {
        "id": "cam-01",
        "name": "Perimeter Gate",
        "kind": "video",
        # Set CAMERA_SOURCE on Render to a reachable RTSP/IP-camera URL or
        # to a video file that exists inside the deployed container.
        "source": CAMERA_SOURCE,
        "enabled": True,
        "fps": 14,
        "res": "1080p",
    },
]

# Default ROI, matching the dashboard's initial configuration.
DEFAULT_ROI = {
    "id": "roi",
    "name": "ROI",
    "color": "#22d3ee",
    "visible": True,
    "pts": [[18, 22], [38, 14], [58, 18], [52, 38], [28, 44]],
}

# Detection / streaming tuning.
PROCESS_WIDTH = 960          # frames are downscaled to this width before inference
DETECT_EVERY = 3             # run YOLO every N frames (motion runs every frame)
PERSON_CONF_THRESHOLD = 0.35 # minimum person confidence for an intrusion
ANOMALY_THRESHOLD = 0.02     # motion fraction inside the ROI above which it is "anomalous"
ALERT_DEBOUNCE_SECONDS = 30  # minimum time between ROI alerts
ALERT_CONFIDENCE_THRESHOLD = 0.65  # notify only when person confidence is greater than 70%
STREAM_FPS = 12              # target processing/streaming rate
JPEG_QUALITY = 70

# Live camera input (OpenCV device index) and folders scanned for saved videos.
LIVE_CAMERA_INDEX = 0
VIDEO_DIRS = [
    BASE_DIR / "saved_videos",
    BASE_DIR / "testVideo",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
