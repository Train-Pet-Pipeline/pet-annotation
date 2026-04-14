"""Shared test fixtures for pet-annotation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

SCHEMA_SQL = Path(__file__).parent.parent / "migrations" / "001_create_annotation_tables.sql"

FRAMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL,
    source          TEXT NOT NULL,
    frame_path      TEXT NOT NULL,
    data_root       TEXT NOT NULL,
    timestamp_ms    INTEGER,
    species         TEXT,
    breed           TEXT,
    lighting        TEXT CHECK(lighting IN ('bright','dim','infrared_night','unknown')
                               OR lighting IS NULL),
    bowl_type       TEXT,
    quality_flag    TEXT NOT NULL DEFAULT 'normal' CHECK(quality_flag IN ('normal','low','failed')),
    blur_score      REAL,
    phash           BLOB,
    aug_quality     TEXT CHECK(aug_quality IN ('ok','failed') OR aug_quality IS NULL),
    aug_seed        INTEGER,
    parent_frame_id TEXT,
    is_anomaly_candidate INTEGER NOT NULL DEFAULT 0,
    anomaly_score   REAL,
    annotation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(annotation_status IN ('pending','annotating','auto_checked',
                                    'approved','needs_review','reviewed','rejected','exported')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@pytest.fixture
def db_conn() -> sqlite3.Connection:
    """In-memory SQLite with frames + annotation tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(FRAMES_SCHEMA)
    if SCHEMA_SQL.exists():
        conn.executescript(SCHEMA_SQL.read_text())
    conn.commit()
    return conn
