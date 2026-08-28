"""Central configuration for the Sentinel CCTV backend.

The dashboard draws zones in a normalized 100x60 coordinate space. The backend
stores those same normalized points and converts them to pixel coordinates at
detection time (``px/100 * width``, ``py/60 * height``).
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Persistence.
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "sentinel.db"
SNAPSHOT_DIR = BASE_DIR / "saved_videos" / "snapshots"

# Dashboard entry point served by FastAPI at "/".
DASHBOARD_PATH = BASE_DIR / "dashboard" / "index.html"

# Camera sources. ``kind`` is either "video" (path relative to BASE_DIR) or
# "camera" (OpenCV device index). ``enabled=False`` marks a camera as offline.
SOURCES = [
    {
        "id": "cam-01",
        "name": "Perimeter Gate",
        "kind": "video",
        "source": str(BASE_DIR / "testVideo" / "hit_and_run.mp4"),
        "enabled": True,
        "fps": 14,
        "res": "1080p",
    },
    {
        "id": "cam-02",
        "name": "Server Room",
        "kind": "video",
        "source": str(BASE_DIR / "testVideo" / "bird_video.mp4"),
        "enabled": True,
        "fps": 12,
        "res": "720p",
    },
    {
        "id": "cam-03",
        "name": "Warehouse Bay",
        "kind": "video",
        "source": str(BASE_DIR / "testVideo" / "animal_video.mp4"),
        "enabled": True,
        "fps": 15,
        "res": "1080p",
    },
    {
        "id": "cam-04",
        "name": "Parking Lot",
        "kind": "video",
        "source": str(BASE_DIR / "testVideo" / "columbina-moonlit-requiem.3840x2160.mp4"),
        "enabled": False,
        "fps": 0,
        "res": "—",
    },
]

# Default zones, matching the dashboard's initial configuration.
DEFAULT_ZONES = [
    {
        "id": "z1",
        "name": "Zone A — Perimeter",
        "color": "#22d3ee",
        "pts": [[18, 22], [38, 14], [58, 18], [52, 38], [28, 44]],
    },
    {
        "id": "z2",
        "name": "Zone B — Restricted",
        "color": "#f43f5e",
        "pts": [[62, 16], [80, 18], [78, 40], [60, 42]],
    },
]

# Detection / streaming tuning.
PROCESS_WIDTH = 960          # frames are downscaled to this width before inference
DETECT_EVERY = 3             # run YOLO every N frames (motion runs every frame)
PERSON_CONF_THRESHOLD = 0.35 # minimum person confidence for an intrusion
ANOMALY_THRESHOLD = 0.02     # motion fraction inside a zone above which it is "anomalous"
ALERT_DEBOUNCE_SECONDS = 30  # one alert per zone within this window
STREAM_FPS = 12              # target processing/streaming rate
JPEG_QUALITY = 70
