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
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import File, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # Keep the backend importable until requirements are installed.
    def load_dotenv():
        return False

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
from Backend.processor import StreamManager, StreamProcessor

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("sentinel.backend")


class ROIIn(BaseModel):
    color: str = "#22d3ee"
    pts: list[list[float]] = Field(default_factory=list)


class ROIVisibilityIn(BaseModel):
    visible: bool


class HandleIn(BaseModel):
    handled: bool = True


class RetentionIn(BaseModel):
    enabled: bool = False
    amount: int = Field(default=7, ge=1, le=3650)
    unit: Literal["hours", "days"] = "days"


class NotificationIn(BaseModel):
    enabled: bool = False
    email: str = Field(default="", max_length=320)
    whatsapp: str = Field(default="", max_length=32)


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

# Uploaded review videos are temporary. They are kept outside VIDEO_DIRS so
# they do not appear as permanent saved videos and are removed after playback.
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
_uploaded_videos: dict[str, tuple[Path, str]] = {}
_uploaded_videos_lock = threading.Lock()

# Batch review videos are processed one at a time. The YOLO instance is shared
# with the live processor, and the single worker prevents several large videos
# from competing for the same model/CPU at once.
_analysis_jobs: dict[str, dict] = {}
_analysis_jobs_lock = threading.Lock()
_analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analysis")


def _public_analysis_job(job: dict) -> dict:
    """Return a JSON-safe job view without exposing temporary server paths."""
    result = {key: value for key, value in job.items() if key != "items"}
    result["items"] = [
        {key: value for key, value in item.items() if key != "path"}
        for item in job.get("items", [])
    ]
    return result


def _update_analysis_job(job_id: str, **changes) -> None:
    with _analysis_jobs_lock:
        job = _analysis_jobs.get(job_id)
        if job is not None:
            job.update(changes)


def _update_analysis_item(job_id: str, item_index: int, **changes) -> None:
    with _analysis_jobs_lock:
        job = _analysis_jobs.get(job_id)
        if job is not None and item_index < len(job["items"]):
            job["items"][item_index].update(changes)


def _run_analysis_job(job_id: str) -> None:
    """Analyze queued videos sequentially and update their progress in memory."""
    with _analysis_jobs_lock:
        job = _analysis_jobs.get(job_id)
        if job is None:
            return
        items = list(job["items"])

    _update_analysis_job(job_id, status="running", current=None, error=None)
    logger.info("Started batch video analysis %s (%d video(s))", job_id, len(items))

    try:
        detector = manager._get_bg_detector()
        for item_index, item in enumerate(items):
            path = Path(item["path"])
            name = item["name"]
            _update_analysis_item(
                job_id,
                item_index,
                status="processing",
                progress=0,
                frames=0,
                detections=0,
                max_confidence=0.0,
                error=None,
            )
            _update_analysis_job(job_id, current=name)
            logger.info("Analyzing video %d/%d: %s", item_index + 1, len(items), name)

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                _update_analysis_item(
                    job_id,
                    item_index,
                    status="failed",
                    error="Video could not be opened",
                )
                path.unlink(missing_ok=True)
                continue

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fps = fps if fps > 0 else 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            _update_analysis_item(job_id, item_index, total_frames=total_frames)

            camera = {
                "id": f"analysis:{job_id}:{item_index}",
                "name": name,
                "kind": "video",
                "source": str(path),
                "enabled": True,
                "loop_video": False,
                "fps": fps,
                "res": "—",
            }
            alert_count = 0

            def batch_alert(camera, roi, confidence, event_type, snapshot_path):
                nonlocal alert_count
                alert_count += 1
                on_alert(camera, roi, confidence, event_type, snapshot_path)

            processor = StreamProcessor(
                camera,
                store.get_roi,
                batch_alert,
                detector,
                manager.bg_lock,
                config_module,
            )
            # Offline processing is faster than real time. Use video time for
            # the existing debounce so one long intrusion is not emitted once
            # per frame, and allow the first detected intrusion immediately.
            processor._last_alert = -processor.alert_debounce
            frame_index = 0
            intrusion_frames = 0
            max_confidence = 0.0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    processor.process_frame(
                        frame,
                        frame_index,
                        event_time=frame_index / fps,
                        encode=False,
                    )
                    if processor.last_intrusion_count:
                        intrusion_frames += 1
                        max_confidence = max(
                            max_confidence,
                            processor.last_intrusion_confidence,
                        )
                    frame_index += 1
                    if frame_index % 10 == 0 or frame_index == total_frames:
                        progress = (
                            min(100, round(frame_index * 100 / total_frames))
                            if total_frames
                            else 0
                        )
                        _update_analysis_item(
                            job_id,
                            item_index,
                            progress=progress,
                            frames=frame_index,
                            detections=intrusion_frames,
                            alerts=alert_count,
                            max_confidence=round(max_confidence, 4),
                        )
            finally:
                cap.release()

            _update_analysis_item(
                job_id,
                item_index,
                status="completed",
                progress=100,
                frames=frame_index,
                detections=intrusion_frames,
                alerts=alert_count,
                max_confidence=round(max_confidence, 4),
            )
            _update_analysis_job(job_id, completed=item_index + 1)
            logger.info(
                "Completed video %s: %d frame(s) with ROI intrusion, %d alert event(s)",
                name,
                intrusion_frames,
                alert_count,
            )
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary batch video %s", path)
    except Exception as exc:  # noqa: BLE001 - retain the failure in job status
        logger.exception("Batch video analysis %s failed", job_id)
        for item in items:
            Path(item["path"]).unlink(missing_ok=True)
        if items:
            try:
                Path(items[0]["path"]).parent.rmdir()
            except OSError:
                logger.warning("Could not remove temporary batch directory for %s", job_id)
        _update_analysis_job(job_id, status="failed", current=None, error=str(exc))
        return

    if items:
        try:
            Path(items[0]["path"]).parent.rmdir()
        except OSError:
            logger.warning("Could not remove temporary batch directory for %s", job_id)
    _update_analysis_job(job_id, status="completed", current=None, completed=len(items))
    logger.info("Finished batch video analysis %s", job_id)


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


def _get_uploaded_video(video_id: str) -> Optional[tuple[Path, str]]:
    with _uploaded_videos_lock:
        item = _uploaded_videos.get(video_id)
    if item is None or not item[0].is_file():
        return None
    return item


def _remove_uploaded_video(video_id: str) -> None:
    with _uploaded_videos_lock:
        item = _uploaded_videos.pop(video_id, None)
    if item is not None:
        try:
            item[0].unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temporary upload %s", item[0])


def build_stats() -> dict:
    s = store.stats()
    online = sum(1 for c in SOURCES if c.get("enabled", True))
    return {
        "cameras_online": online,
        "cameras_total": len(SOURCES),
        "roi_configured": s["roi_configured"],
        "detections_24h": s["detections_24h"],
        "avg_confidence": s["avg_confidence"],
    }


def _mask_recipient(value: str) -> str:
    """Keep alert logs useful without exposing full email/phone details."""
    recipient = str(value or "")
    if "@" in recipient:
        local, domain = recipient.split("@", 1)
        return f"{local[:2]}***@{domain}"
    digits = "".join(character for character in recipient if character.isdigit())
    if len(digits) <= 4:
        return "***"
    return f"+***{digits[-4:]}"


def maybe_send_alerts(event: dict) -> None:
    """Forward high-confidence intrusions to the alert modules when enabled.

    Dashboard/SQLite events are created for every intrusion. External messages
    are opt-in and are sent only when confidence is greater than the configured
    threshold.
    """
    notification_settings = store.get_notification_settings()
    if not notification_settings["enabled"]:
        logger.info("External alerts are disabled")
        return
    confidence = float(event.get("conf", 0.0))
    if confidence <= config_module.ALERT_CONFIDENCE_THRESHOLD:
        logger.info(
            "Skipping external alert for %s: confidence %.3f is not greater than %.3f",
            event.get("cam", "unknown"),
            confidence,
            config_module.ALERT_CONFIDENCE_THRESHOLD,
        )
        return

    message = (
        f"Intrusion detected: {event['cam']} / {event['roi']} "
        f"(confidence {confidence:.2f})"
    )
    email_recipient = notification_settings["email"]
    whatsapp_recipient = notification_settings["whatsapp"]
    if not email_recipient and not whatsapp_recipient:
        logger.warning("External alert triggered, but no recipients are registered")
        return

    logger.info(
        "External alert triggered for %s (confidence %.1f%%, threshold %.1f%%)",
        event.get("cam", "unknown"),
        confidence * 100,
        config_module.ALERT_CONFIDENCE_THRESHOLD * 100,
    )
    if email_recipient:
        try:
            from AlertSystem.mail_alert import MailAlert

            masked_email = _mask_recipient(email_recipient)
            logger.info("Sending email alert to %s", masked_email)
            result = MailAlert().send(
                sender=os.getenv("ALERT_MAIL_FROM", "onboarding@resend.dev"),
                reciever=email_recipient,
                subject="CCTV Intrusion Alert",
                message=message,
            )
            if result is None:
                logger.error("EMAIL ALERT FAILED for %s: provider returned no response", masked_email)
            else:
                logger.info("EMAIL ALERT ACCEPTED by provider for %s", masked_email)
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            logger.error("EMAIL ALERT FAILED for %s: %s", _mask_recipient(email_recipient), exc)
    if whatsapp_recipient:
        try:
            from AlertSystem.whatsapp_alert import WhatsAPPAlert

            masked_whatsapp = _mask_recipient(whatsapp_recipient)
            logger.info("Sending WhatsApp alert to %s", masked_whatsapp)
            result = WhatsAPPAlert(whatsapp_recipient, message).send()
            if isinstance(result, dict) and result.get("status") == "failed":
                logger.error(
                    "WHATSAPP ALERT FAILED for %s: %s",
                    masked_whatsapp,
                    result.get("error", "unknown error"),
                )
            else:
                logger.info("WHATSAPP ALERT ACCEPTED by provider for %s", masked_whatsapp)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "WHATSAPP ALERT FAILED for %s: %s",
                _mask_recipient(whatsapp_recipient),
                exc,
            )


def on_alert(camera: dict, roi: dict, confidence: float, event_type: str, snapshot_path: str) -> None:
    event = store.add_event(
        camera_id=camera["id"],
        camera_name=camera["name"],
        roi_id=roi["id"],
        roi_name=roi["name"],
        confidence=confidence,
        event_type=event_type,
        snapshot_path=snapshot_path,
    )
    ws_manager.broadcast_threadsafe({"type": "alert", "data": event})
    ws_manager.broadcast_threadsafe({"type": "stats", "data": build_stats()})
    if event["conf"] > config_module.ALERT_CONFIDENCE_THRESHOLD:
        threading.Thread(target=maybe_send_alerts, args=(event,), daemon=True).start()


manager = StreamManager(get_roi=store.get_roi, on_alert=on_alert, config=config_module)


async def retention_cleanup_loop():
    while True:
        await asyncio.sleep(60)
        deleted = store.purge_expired_events()
        if deleted:
            logger.info("Removed %d expired alert result(s)", deleted)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    ws_manager.loop = asyncio.get_running_loop()
    deleted = store.purge_expired_events()
    if deleted:
        logger.info("Removed %d expired alert result(s) at startup", deleted)
    cleanup_task = asyncio.create_task(retention_cleanup_loop())
    logger.info("Sentinel backend ready (db=%s)", config_module.DB_PATH)
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
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


@app.get("/api/detector")
async def detector_status():
    """Load the configured YOLO detector and report whether it is ready."""
    try:
        detector = await asyncio.to_thread(manager._get_bg_detector)
        # Execute one real inference so this endpoint verifies model execution,
        # not just that the weights file/import exists.
        test_frame = np.zeros((64, 64, 3), dtype=np.uint8)

        def run_test_inference():
            with manager.bg_lock:
                return detector(test_frame)

        detections, _ = await asyncio.to_thread(run_test_inference)
    except Exception as exc:  # noqa: BLE001 - expose a useful deployment status
        raise HTTPException(status_code=503, detail=f"YOLO detector unavailable: {exc}") from exc
    return {
        "status": "ready",
        "model": getattr(config_module, "YOLO_MODEL_PATH", "yolov8n.pt"),
        "foreground_model": "MOG2",
        "person_class": 0,
        "inference": "ok",
        "test_detections": len(detections),
    }


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


@app.get("/api/roi")
async def get_roi():
    return store.get_roi()


@app.put("/api/roi")
async def put_roi(payload: ROIIn):
    if len(payload.pts) < 3:
        raise HTTPException(status_code=422, detail="An ROI needs at least 3 points.")
    roi = store.save_roi(payload.color, payload.pts)
    await ws_manager.broadcast({"type": "roi", "data": roi})
    await ws_manager.broadcast({"type": "stats", "data": build_stats()})
    return roi


@app.patch("/api/roi/visibility")
async def patch_roi_visibility(payload: ROIVisibilityIn):
    roi = store.set_roi_visibility(payload.visible)
    if roi is None:
        raise HTTPException(status_code=404, detail="ROI is not configured.")
    await ws_manager.broadcast({"type": "roi", "data": roi})
    return roi


@app.delete("/api/roi")
async def delete_roi():
    store.delete_roi()
    await ws_manager.broadcast({"type": "roi", "data": None})
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


@app.get("/api/retention")
async def get_retention():
    return store.get_retention()


@app.put("/api/retention")
async def put_retention(payload: RetentionIn):
    retention = store.save_retention(payload.enabled, payload.amount, payload.unit)
    deleted = store.purge_expired_events()
    if deleted:
        logger.info("Removed %d expired alert result(s) after retention update", deleted)
    return retention


@app.get("/api/notifications")
async def get_notifications():
    return store.get_notification_settings()


@app.put("/api/notifications")
async def put_notifications(payload: NotificationIn):
    if payload.enabled and not (payload.email.strip() or payload.whatsapp.strip()):
        raise HTTPException(
            status_code=422,
            detail="Add an email address or WhatsApp number before enabling alerts.",
        )
    return store.save_notification_settings(
        payload.enabled,
        payload.email,
        payload.whatsapp,
    )


# --------------------------------------------------------------------------- #
# Live MJPEG stream
# --------------------------------------------------------------------------- #
def _mjpeg_generator(
    proc,
    cleanup_upload_id: Optional[str] = None,
    stop_when_disconnected: bool = False,
):
    last_seq = -1
    try:
        while True:
            seq, jpeg = proc.wait_for_frame(timeout=1.0)
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            elif not proc.is_alive():
                break
    finally:
        if stop_when_disconnected:
            manager.stop(proc.camera["id"])
        if cleanup_upload_id:
            _remove_uploaded_video(cleanup_upload_id)




@app.get("/api/videos")
async def get_videos():
    """List saved/recorded videos that can be played from the dashboard."""
    return _list_videos()


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept one video for on-demand detection and return a temporary id."""
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type. Use: {', '.join(sorted(VIDEO_EXTENSIONS))}",
        )

    video_id = uuid.uuid4().hex
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{video_id}{suffix}"
    try:
        with path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)
    except Exception as exc:  # noqa: BLE001 - convert upload failures to HTTP errors
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not save uploaded video.") from exc
    finally:
        await file.close()

    cap = cv2.VideoCapture(str(path))
    valid = cap.isOpened()
    cap.release()
    if not valid:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable video.")

    with _uploaded_videos_lock:
        _uploaded_videos[video_id] = (path, filename)
    logger.info("Accepted temporary video upload %s", filename)
    return {"id": video_id, "name": filename}


@app.post("/api/videos/analyze")
async def analyze_videos(files: list[UploadFile] = File(...)):
    """Queue multiple uploaded videos for sequential ROI/anomaly analysis."""
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one video.")

    job_id = uuid.uuid4().hex
    job_dir = UPLOAD_DIR / f"batch-{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    items = []
    try:
        for index, file in enumerate(files):
            filename = Path(file.filename or "").name
            suffix = Path(filename).suffix.lower()
            if not filename or suffix not in VIDEO_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported video type for {filename or 'unnamed file'}. "
                    f"Use: {', '.join(sorted(VIDEO_EXTENSIONS))}",
                )

            path = job_dir / f"{index}{suffix}"
            try:
                with path.open("wb") as destination:
                    while chunk := await file.read(1024 * 1024):
                        destination.write(chunk)
            finally:
                await file.close()

            cap = cv2.VideoCapture(str(path))
            valid = cap.isOpened()
            cap.release()
            if not valid:
                raise HTTPException(status_code=400, detail=f"{filename} is not a readable video.")

            items.append(
                {
                    "id": f"{job_id}-{index}",
                    "name": filename,
                    "path": str(path),
                    "status": "queued",
                    "progress": 0,
                    "frames": 0,
                    "total_frames": 0,
                    "detections": 0,
                    "alerts": 0,
                    "max_confidence": 0.0,
                    "error": None,
                }
            )
    except HTTPException:
        for path in job_dir.glob("*"):
            path.unlink(missing_ok=True)
        job_dir.rmdir()
        raise
    except Exception as exc:  # noqa: BLE001 - convert upload failures to HTTP errors
        for path in job_dir.glob("*"):
            path.unlink(missing_ok=True)
        job_dir.rmdir()
        raise HTTPException(status_code=500, detail="Could not save batch videos.") from exc

    job = {
        "id": job_id,
        "status": "queued",
        "total": len(items),
        "completed": 0,
        "current": None,
        "error": None,
        "items": items,
    }
    with _analysis_jobs_lock:
        _analysis_jobs[job_id] = job
    _analysis_executor.submit(_run_analysis_job, job_id)
    logger.info("Queued batch video analysis %s with %d video(s)", job_id, len(items))
    return _public_analysis_job(job)


@app.get("/api/videos/analysis/{job_id}")
async def get_analysis_job(job_id: str):
    """Return the current status and per-video results for a batch job."""
    with _analysis_jobs_lock:
        job = _analysis_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video analysis job not found.")
        return _public_analysis_job(job)


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
        _mjpeg_generator(proc, stop_when_disconnected=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/upload/{video_id}")
async def stream_uploaded_video(video_id: str):
    """Run detection once on a temporary video upload."""
    item = _get_uploaded_video(video_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Uploaded video not found or expired.")
    path, filename = item
    camera = {
        "id": f"upload:{video_id}",
        "name": filename,
        "kind": "video",
        "source": str(path),
        "enabled": True,
        "loop_video": False,
        "fps": 0,
        "res": "—",
    }
    proc = manager.ensure(camera)
    return StreamingResponse(
        _mjpeg_generator(proc, cleanup_upload_id=video_id),
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
        _mjpeg_generator(proc, stop_when_disconnected=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/api/stream/{camera_id}")
async def stream_camera(camera_id: str):
    camera = find_camera(camera_id)
    if camera is None or not camera.get("enabled", True):
        raise HTTPException(status_code=404, detail="Camera not found or offline.")
    proc = manager.ensure(camera)
    return StreamingResponse(
        _mjpeg_generator(proc, stop_when_disconnected=True),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------------------- #
# Browser integrated-camera detection
# --------------------------------------------------------------------------- #
@app.websocket("/ws/live")
async def live_video_websocket(ws: WebSocket):
    """Receive browser webcam JPEGs and return annotated detection JPEGs."""
    await ws.accept()
    processor = None
    camera = {
        "id": "browser-live",
        "name": "Integrated Camera",
        "kind": "browser",
        "source": "browser",
        "enabled": True,
        "fps": 0,
        "res": "—",
    }
    try:
        processor = await asyncio.to_thread(manager.create_processor, camera)
        logger.info(
            "Browser integrated-camera detection connected (MOG2 foreground + YOLO person detection)"
        )
        frame_idx = 0
        while True:
            payload = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            jpeg = await asyncio.to_thread(processor.process_frame, frame, frame_idx)
            if jpeg:
                await ws.send_bytes(jpeg)
            frame_idx += 1
    except WebSocketDisconnect:
        logger.info("Browser integrated-camera detection disconnected")
    except Exception:
        logger.exception("Browser integrated-camera detection failed")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
    finally:
        if processor is not None:
            processor.stop()


# --------------------------------------------------------------------------- #
# WebSocket: real-time alerts, ROI and stats
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "roi", "data": store.get_roi()}))
        await ws.send_text(json.dumps({"type": "stats", "data": build_stats()}))
        while True:
            message = await ws.receive_text()
            if message == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)
