"""SQLite persistence for one ROI and alert events.

SQLite is used directly (no ORM) so the backend has no extra dependencies beyond
the standard library. Every operation opens a short-lived connection protected
by a process-wide lock; this is safe for the single-process FastAPI + worker
threads design used here.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta

from Backend.config import DB_PATH, DEFAULT_ROI

_lock = threading.RLock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        roi_table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'roi'"
        ).fetchone() is not None
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roi (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                color TEXT NOT NULL,
                visible INTEGER NOT NULL DEFAULT 1,
                points TEXT NOT NULL
            )
            """
        )
        roi_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(roi)").fetchall()
        }
        if "visible" not in roi_columns:
            conn.execute("ALTER TABLE roi ADD COLUMN visible INTEGER NOT NULL DEFAULT 1")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                camera_name TEXT NOT NULL,
                roi_id TEXT,
                roi_name TEXT,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                type TEXT NOT NULL,
                handled INTEGER NOT NULL DEFAULT 0,
                snapshot_path TEXT
            )
            """
        )
        event_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        if "zone_id" in event_columns:
            conn.execute("ALTER TABLE events RENAME COLUMN zone_id TO roi_id")
        if "zone_name" in event_columns:
            conn.execute("ALTER TABLE events RENAME COLUMN zone_name TO roi_name")
        conn.commit()

        # Migrate the old multi-zone table by keeping its first polygon as the
        # single ROI. The legacy table is no longer used by the application.
        legacy_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'zones'"
        ).fetchone() is not None
        roi = conn.execute("SELECT * FROM roi WHERE id = 'roi'").fetchone()
        if legacy_exists:
            if roi is None:
                legacy = conn.execute("SELECT * FROM zones ORDER BY rowid LIMIT 1").fetchone()
                if legacy is not None:
                    conn.execute(
                        "INSERT INTO roi (id, name, color, visible, points) VALUES (?, ?, ?, 1, ?)",
                        ("roi", "ROI", legacy["color"], legacy["points"]),
                    )
            conn.execute("DROP TABLE zones")
        elif roi is None and not roi_table_exists:
            conn.execute(
                "INSERT INTO roi (id, name, color, visible, points) VALUES (?, ?, ?, ?, ?)",
                (DEFAULT_ROI["id"], DEFAULT_ROI["name"], DEFAULT_ROI["color"], int(DEFAULT_ROI["visible"]), json.dumps(DEFAULT_ROI["pts"])),
            )
        conn.commit()
        conn.close()


# --------------------------------------------------------------------------- #
# ROI
# --------------------------------------------------------------------------- #
def _roi_from_row(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "color": row["color"],
        "visible": bool(row["visible"]),
        "pts": json.loads(row["points"]),
    }


def get_roi() -> dict | None:
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT * FROM roi WHERE id = 'roi'").fetchone()
        conn.close()
    return _roi_from_row(row) if row else None


def save_roi(color: str, pts: list) -> dict:
    with _lock:
        conn = _connect()
        existing = conn.execute("SELECT visible FROM roi WHERE id = 'roi'").fetchone()
        visible = bool(existing["visible"]) if existing else True
        roi = {"id": "roi", "name": "ROI", "color": color, "visible": visible, "pts": pts}
        conn.execute(
            """
            INSERT INTO roi (id, name, color, visible, points) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                color = excluded.color,
                points = excluded.points
            """,
            (roi["id"], roi["name"], roi["color"], int(roi["visible"]), json.dumps(roi["pts"])),
        )
        conn.commit()
        conn.close()
    return roi


def set_roi_visibility(visible: bool) -> dict | None:
    with _lock:
        conn = _connect()
        conn.execute("UPDATE roi SET visible = ? WHERE id = 'roi'", (int(visible),))
        row = conn.execute("SELECT * FROM roi WHERE id = 'roi'").fetchone()
        conn.commit()
        conn.close()
    return _roi_from_row(row) if row else None


def delete_roi() -> bool:
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM roi WHERE id = 'roi'")
        conn.commit()
        deleted = cur.rowcount > 0
        conn.close()
    return deleted



# --------------------------------------------------------------------------- #
# Events / alerts
# --------------------------------------------------------------------------- #
def _event_from_row(row) -> dict:
    ts = row["timestamp"]
    try:
        time_str = datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        time_str = ts or ""
    return {
        "id": row["id"],
        "cam": row["camera_name"],
        "roi": row["roi_name"] or "ROI",
        "time": time_str,
        "timestamp": ts,
        "conf": row["confidence"],
        "type": row["type"],
        "handled": bool(row["handled"]),
        "snapshot_path": row["snapshot_path"],
        "camera_id": row["camera_id"],
        "roi_id": row["roi_id"],
    }


def add_event(
    camera_id: str,
    camera_name: str,
    roi_id: str,
    roi_name: str,
    confidence: float,
    event_type: str = "intrusion",
    snapshot_path: str | None = None,
) -> dict:
    ts = datetime.now().isoformat()
    with _lock:
        conn = _connect()
        cur = conn.execute(
            """
            INSERT INTO events
                (camera_id, camera_name, roi_id, roi_name, timestamp,
                 confidence, type, handled, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (camera_id, camera_name, roi_id, roi_name, ts, confidence, event_type, snapshot_path),
        )
        conn.commit()
        event_id = cur.lastrowid
        conn.close()
    return _event_from_row(_get_event_row(event_id))


def _get_event_row(event_id: int):
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        conn.close()
    return row


def list_events(limit: int = 200, event_type: str | None = None, q: str | None = None) -> list[dict]:
    sql = "SELECT * FROM events"
    clauses = []
    params = []

    if event_type and event_type != "all":
        clauses.append("type = ?")
        params.append(event_type)
    if q:
        clauses.append("(camera_name LIKE ? OR roi_name LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like])

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    with _lock:
        conn = _connect()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    return [_event_from_row(r) for r in rows]


def mark_handled(event_id: int, handled: bool = True) -> bool:
    with _lock:
        conn = _connect()
        cur = conn.execute("UPDATE events SET handled = ? WHERE id = ?", (1 if handled else 0, event_id))
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()
    return updated


def stats() -> dict:
    with _lock:
        conn = _connect()
        active_roi = conn.execute("SELECT COUNT(*) FROM roi WHERE id = 'roi'").fetchone()[0]
        since = (datetime.now() - timedelta(hours=24)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*), AVG(confidence) FROM events WHERE timestamp >= ?",
            (since,),
        ).fetchone()
        conn.close()
    detections = row[0] or 0
    avg_conf = round(row[1] or 0.0, 3)
    return {
        "roi_configured": bool(active_roi),
        "detections_24h": detections,
        "avg_confidence": avg_conf,
    }
