"""SQLite persistence for zones and alert events.

SQLite is used directly (no ORM) so the backend has no extra dependencies beyond
the standard library. Every operation opens a short-lived connection protected
by a process-wide lock; this is safe for the single-process FastAPI + worker
threads design used here.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta

from Backend.config import DB_PATH, DEFAULT_ZONES

_lock = threading.RLock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                color TEXT NOT NULL,
                points TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                camera_name TEXT NOT NULL,
                zone_id TEXT,
                zone_name TEXT,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                type TEXT NOT NULL,
                handled INTEGER NOT NULL DEFAULT 0,
                snapshot_path TEXT
            )
            """
        )
        conn.commit()

        # Seed the default zones the first time the database is created.
        count = conn.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
        if count == 0:
            for z in DEFAULT_ZONES:
                conn.execute(
                    "INSERT INTO zones (id, name, color, points) VALUES (?, ?, ?, ?)",
                    (z["id"], z["name"], z["color"], json.dumps(z["pts"])),
                )
            conn.commit()
        conn.close()


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #
def _zone_from_row(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "color": row["color"],
        "pts": json.loads(row["points"]),
    }


def list_zones() -> list[dict]:
    with _lock:
        conn = _connect()
        rows = conn.execute("SELECT * FROM zones ORDER BY rowid").fetchall()
        conn.close()
    return [_zone_from_row(r) for r in rows]


def get_zone(zone_id: str) -> dict | None:
    with _lock:
        conn = _connect()
        row = conn.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)).fetchone()
        conn.close()
    return _zone_from_row(row) if row else None


def create_zone(name: str, color: str, pts: list) -> dict:
    zone_id = f"z-{uuid.uuid4().hex[:8]}"
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT INTO zones (id, name, color, points) VALUES (?, ?, ?, ?)",
            (zone_id, name, color, json.dumps(pts)),
        )
        conn.commit()
        conn.close()
    return {"id": zone_id, "name": name, "color": color, "pts": pts}


def delete_zone(zone_id: str) -> bool:
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
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
        "zone": row["zone_name"],
        "time": time_str,
        "timestamp": ts,
        "conf": row["confidence"],
        "type": row["type"],
        "handled": bool(row["handled"]),
        "snapshot_path": row["snapshot_path"],
        "camera_id": row["camera_id"],
        "zone_id": row["zone_id"],
    }


def add_event(
    camera_id: str,
    camera_name: str,
    zone_id: str,
    zone_name: str,
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
                (camera_id, camera_name, zone_id, zone_name, timestamp,
                 confidence, type, handled, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (camera_id, camera_name, zone_id, zone_name, ts, confidence, event_type, snapshot_path),
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
        clauses.append("(camera_name LIKE ? OR zone_name LIKE ?)")
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
        active_zones = conn.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
        since = (datetime.now() - timedelta(hours=24)).isoformat()
        row = conn.execute(
            "SELECT COUNT(*), AVG(confidence) FROM events WHERE timestamp >= ?",
            (since,),
        ).fetchone()
        conn.close()
    detections = row[0] or 0
    avg_conf = round(row[1] or 0.0, 3)
    return {
        "active_zones": active_zones,
        "detections_24h": detections,
        "avg_confidence": avg_conf,
    }

