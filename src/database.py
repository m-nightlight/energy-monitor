"""
SQLite database module for energy meter readings.

Schema:
  readings
    id           INTEGER  Primary key
    timestamp    TEXT     ISO-8601 UTC timestamp of the reading
    value        REAL     Meter reading
    unit         TEXT     Unit label (e.g. 'MWh')
    raw_text     TEXT     Raw OCR output (for debugging)
    confidence   REAL     Tesseract confidence score (0–100)
    image_path   TEXT     Path to source image (may be NULL if deleted)
    sane         INTEGER  1 = passed sanity check, 0 = flagged
    created_at   TEXT     Row insertion timestamp
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    value       REAL    NOT NULL,
    unit        TEXT    NOT NULL DEFAULT 'MWh',
    raw_text    TEXT,
    confidence  REAL,
    image_path  TEXT,
    sane        INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings(timestamp);
"""


def init_db(db_path: str) -> None:
    """Create the database and table if they don't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
    logger.debug("Database ready: %s", db_path)


def insert_reading(
    db_path: str,
    value: float,
    unit: str = "MWh",
    raw_text: str = "",
    confidence: float = 0.0,
    image_path: Optional[str] = None,
    sane: bool = True,
    timestamp: Optional[datetime] = None,
) -> int:
    """
    Insert a new reading row.

    Returns:
        The row id of the inserted record.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    sane_int = 1 if sane else 0

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO readings (timestamp, value, unit, raw_text, confidence, image_path, sane)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts_str, value, unit, raw_text, confidence, image_path, sane_int),
        )
        row_id = cursor.lastrowid

    logger.info(
        "Stored reading id=%d  %.3f %s  ts=%s  sane=%s",
        row_id, value, unit, ts_str, sane,
    )
    return row_id


def get_last_reading(db_path: str) -> Optional[dict]:
    """Return the most recent sane reading, or None if the table is empty."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, timestamp, value, unit FROM readings WHERE sane=1 ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

    if row is None:
        return None

    return {"id": row[0], "timestamp": row[1], "value": row[2], "unit": row[3]}


def get_readings(
    db_path: str,
    limit: int = 100,
    sane_only: bool = True,
) -> list:
    """
    Return the N most recent readings as a list of dicts.

    Args:
        db_path: Path to SQLite file.
        limit: Max number of rows to return.
        sane_only: If True, exclude rows flagged as insane.
    """
    where = "WHERE sane=1" if sane_only else ""
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT id, timestamp, value, unit, confidence, sane FROM readings "
            f"{where} ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "value": r[2],
            "unit": r[3],
            "confidence": r[4],
            "sane": bool(r[5]),
        }
        for r in rows
    ]


def export_csv(db_path: str, output_path: str, sane_only: bool = True) -> int:
    """
    Export readings to a CSV file.

    Returns:
        Number of rows written.
    """
    import csv

    rows = get_readings(db_path, limit=10_000_000, sane_only=sane_only)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "timestamp", "value", "unit", "confidence", "sane"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d rows to %s", len(rows), output_path)
    return len(rows)


# ── Internal ─────────────────────────────────────────────────────────────────

@contextmanager
def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
