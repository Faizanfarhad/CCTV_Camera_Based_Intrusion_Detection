"""FastAPI backend connecting the React dashboard to the CCTV detection pipeline.

Run from the project root:

    uvicorn Backend.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/ (the dashboard is served by the backend too).
"""

import asyncio
import base64
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from Backend import store
from Backend import config as config_module
from Backend.config import (
    BASE_DIR,
    DASHBOARD_PATH,
    LIVE_CAMERA_INDEX,
    SOURCES,
    VIDEO_DIRS,
    VIDEO_EXTENSIONS,
)
from Backend.processor import StreamManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("sentinel.backend")


class ZoneIn(BaseModel):
    name: str
    color: str = "#22d3ee"
    pts: list[list[float]] = Field(default_factory=list)


class HandleIn(BaseModel):
    handled: bool = True


class ConnectionManager:
    """Tracks open WebSocket clients and pushes JSON messages to them."""

    def __init__(self):
        self.active: set[WebSocket] = set()
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self.lock:
            self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        with self.lock:
            self.active.discard(ws)

    async def broadcast(self, message: dict):
        with self.lock:
            targets = list(self.active)
        text = json.dumps(message, default=str)
        for ws in targets:
            try:
                await ws.send_text(text)
            except Exception:
                self.disconnect(ws)

    def broadcast_threadsafe(self, message: dict):
        loop = self.loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)


ws_manager = ConnectionManager()


def find_camera(camera_id: str) -> Optional[dict]:
    return next((c for c in SOURCES if c["id"] == camera_id), None)


def _list_videos() -> list[dict]:
    """List saved/recorded videos available for playback."""
    videos = []
    for directory in VIDEO_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTENSIONS and path.is_file():
                rel = path.relative_to(BASE_DIR).as_posix()
                videos.append(
                    {
                        "id": base64.urlsafe_b64encode(rel.encode()).decode(),
                        "name": rel,
                        "path": str(path),
                    }
                )
    return videos


def _resolve_video(video_id: str) -> Optional[Path]:
    """Resolve a video id from ``/api/videos`` back to a safe file path."""
    try:
        rel = base64.urlsafe_b64decode(video_id.encode()).decode()
    except Exception:
        return None
    candidate = (BASE_DIR / rel).resolve()
    for directory in VIDEO_DIRS:
        root = directory.resolve()
        if candidate == root or str(candidate).startswith(str(root) + os.sep):
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                return candidate
    return None

    return next((c for c in SOURCES if c["id"] == camera_id), None)


def build_stats() -> dict:
    s = store.stats()
    online = sum(1 for c in SOURCES if c.get("enabled", True))
    return {
        "cameras_online": online,
        "cameras_total": len(SOURCES),
        "active_zones": s["active_zones"],
        "detections_24h": s["detections_24h"],
        "avg_confidence": s["avg_confidence"],
    }


def maybe_send_alerts(event: dict) -> None:
    """Forward an intrusion to the alert modules when explicitly enabled.

    Sending real email/WhatsApp messages on every intrusion is noisy, so this is
    opt-in via ``ENABLE_ALERTS=1`` in the environment.
    """
    if os.getenv("ENABLE_ALERTS") != "1":
        return
    message = (
        f"Intrusion detected: {event['cam']} / {event['zone']} "
        f"(confidence {event['conf']:.2f})"
    )
    if os.getenv("ALERT_MAIL_TO"):
        try:
            from AlertSystem.mail_alert import MailAlert

            MailAlert().send(
                sender=os.getenv("ALERT_MAIL_FROM", "onboarding@resend.dev"),
                reciever=os.getenv("ALERT_MAIL_TO"),
                subject="CCTV Intrusion Alert",
                message=message,
            )
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            logger.warning("Mail alert failed: %s", exc)
    if os.getenv("ALERT_WHATSAPP_TO"):
        try:
            from AlertSystem.whatsapp_alert import WhatsAPPAlert

            WhatsAPPAlert(os.getenv("ALERT_WHATSAPP_TO"), message).send()
        except Exception as exc:  # noqa: BLE001
            logger.warning("WhatsApp alert failed: %s", exc)


def on_alert(camera: dict, zone: dict, confidence: float, event_type: str, snapshot_path: str) -> None:
    event = store.add_event(
        camera_id=camera["id"],
        camera_name=camera["name"],
        zone_id=zone["id"],
        zone_name=zone["name"],
        confidence=confidence,
        event_type=event_type,
        snapshot_path=snapshot_path,
    )
    ws_manager.broadcast_threadsafe({"type": "alert", "data": event})
    ws_manager.broadcast_threadsafe({"type": "stats", "data": build_stats()})
    maybe_send_alerts(event)


manager = StreamManager(get_zones=store.list_zones, on_alert=on_alert, config=config_module)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    ws_manager.loop = asyncio.get_running_loop()
    logger.info("Sentinel backend ready (db=%s)", config_module.DB_PATH)
    yield
    manager.stop_all()


app = FastAPI(title="Sentinel CCTV Intrusion API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Dashboard / static entry point
# --------------------------------------------------------------------------- #
@app.get("/")
async def index():
    return FileResponse(DASHBOARD_PATH)


# --------------------------------------------------------------------------- #
# REST API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/cameras")
async def cameras():
    running = manager.running_ids()
    result = []
    for c in SOURCES:
        result.append(
            {
                "id": c["id"],
                "name": c["name"],
                "status": "online" if c.get("enabled", True) else "offline",
                "fps": c.get("fps", 0),
                "res": c.get("res", "—"),
                "streaming": c["id"] in running,
                "stream": f"/api/stream/{c['id']}",
            }
        )
    return result


@app.get("/api/stats")
async def get_stats():
    return build_stats()


@app.get("/api/zones")
async def get_zones():
    return store.list_zones()


@app.post("/api/zones")
async def post_zone(payload: ZoneIn):
    if len(payload.pts) < 3:
        raise HTTPException(status_code=422, detail="A zone needs at least 3 points.")
    zone = store.create_zone(payload.name, payload.color, payload.pts)
    await ws_manager.broadcast({"type": "zones", "data": store.list_zones()})
    await ws_manager.broadcast({"type": "stats", "data": build_stats()})
    return zone


@app.delete("/api/zones/{zone_id}")
async def delete_zone(zone_id: str):
    if not store.delete_zone(zone_id):
        raise HTTPException(status_code=404, detail="Zone not found.")
    await ws_manager.broadcast({"type": "zones", "data": store.list_zones()})
    await ws_manager.broadcast({"type": "stats", "data": build_stats()})
    return {"ok": True}


@app.get("/api/alerts")
async def get_alerts(limit: int = 200, type: Optional[str] = None, q: Optional[str] = None):
    return store.list_events(limit=limit, event_type=type, q=q)


@app.patch("/api/alerts/{event_id}")
async def patch_alert(event_id: int, payload: HandleIn):
    if not store.mark_handled(event_id, payload.handled):
        raise HTTPException(status_code=404, detail="Alert not found.")
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Live MJPEG stream
# --------------------------------------------------------------------------- #
def _mjpeg_generator(proc):
    last_seq = -1
    while proc.is_alive():
        seq, jpeg = proc.wait_for_frame(timeout=1.0)
        if jpeg is not None and seq != last_seq:
            last_seq = seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )




@app.get("/api/videos")
async def get_videos():
    """List saved/recorded videos that can be played from the dashboard."""
    return _list_videos()


@app.get("/api/stream/live")
async def stream_live():
    """Stream the live camera (default OpenCV device index)."""
    cap = cv2.VideoCapture(int(LIVE_CAMERA_INDEX))
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=503, detail="Live camera not available.")
    cap.release()

    camera = {
        "id": "live",
        "name": "Live Camera",
        "kind": "camera",
        "source": int(LIVE_CAMERA_INDEX),
        "enabled": True,
        "fps": 0,
        "res": "—",
    }
    proc = manager.ensure(camera)
    return StreamingResponse(
        _mjpeg_generator(proc),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/video/{video_id}")
async def stream_video(video_id: str):
    """Stream a saved video file by its id from ``/api/videos``."""
    path = _resolve_video(video_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Saved video not found.")
    camera = {
        "id": f"video:{video_id}",
        "name": path.name,
        "kind": "video",
        "source": str(path),
        "enabled": True,
        "fps": 0,
        "res": "—",
    }
    proc = manager.ensure(camera)
    return StreamingResponse(
        _mjpeg_generator(proc),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/api/stream/{camera_id}")
async def stream_camera(camera_id: str):
    camera = find_camera(camera_id)
    if camera is None or not camera.get("enabled", True):
        raise HTTPException(status_code=404, detail="Camera not found or offline.")
    proc = manager.ensure(camera)
    return StreamingResponse(
        _mjpeg_generator(proc),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------------------- #
# WebSocket: real-time alerts, zones and stats
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "zones", "data": store.list_zones()}))
        await ws.send_text(json.dumps({"type": "stats", "data": build_stats()}))
        while True:
            message = await ws.receive_text()
            if message == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)

