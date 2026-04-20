"""Tests for DPO modality filtering (B13)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pet_annotation.dpo.generate_pairs import generate_cross_model_pairs
from pet_annotation.store import AnnotationStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    lighting        TEXT,
    bowl_type       TEXT,
    quality_flag    TEXT NOT NULL DEFAULT 'normal',
    blur_score      REAL,
    phash           BLOB,
    aug_quality     TEXT,
    aug_seed        INTEGER,
    parent_frame_id TEXT,
    is_anomaly_candidate INTEGER NOT NULL DEFAULT 0,
    anomaly_score   REAL,
    annotation_status TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

MIGRATIONS = [
    "001_create_annotation_tables.sql",
    "002_add_modality.sql",
    "003_create_audio_annotations.sql",
]

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


@pytest.fixture()
def modality_conn() -> sqlite3.Connection:
    """In-memory SQLite with frames + all migrations applied (including modality column)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(FRAMES_SCHEMA)
    for migration in MIGRATIONS:
        sql_path = MIGRATIONS_DIR / migration
        if sql_path.exists():
            conn.executescript(sql_path.read_text())
    conn.commit()
    return conn


def _insert_frame(conn: sqlite3.Connection, frame_id: str) -> None:
    """Insert a minimal frame row."""
    conn.execute(
        """
        INSERT INTO frames (frame_id, video_id, source, frame_path, data_root, annotation_status)
        VALUES (?, 'vid1', 'test', '/tmp/f.jpg', '/tmp', 'approved')
        """,
        (frame_id,),
    )
    conn.commit()


def _insert_annotation(
    conn: sqlite3.Connection,
    frame_id: str,
    annotation_id: str,
    modality: str = "vision",
    review_status: str = "approved",
    model_name: str = "gpt-4o",
    confidence: float = 0.95,
) -> None:
    """Insert an annotation row with explicit modality."""
    conn.execute(
        """
        INSERT INTO annotations (
            annotation_id, frame_id, model_name, prompt_hash,
            raw_response, parsed_output, schema_valid, confidence_overall,
            review_status, prompt_tokens, completion_tokens, total_tokens,
            api_latency_ms, modality
        ) VALUES (?, ?, ?, 'hash1', ?, '{"species":"cat"}', 1, ?, ?, 5, 10, 15, 100, ?)
        """,
        (
            annotation_id,
            frame_id,
            model_name,
            json.dumps({"species": "cat", "confidence": confidence}),
            confidence,
            review_status,
            modality,
        ),
    )
    conn.commit()


def _insert_comparison(
    conn: sqlite3.Connection,
    frame_id: str,
    comparison_id: str,
    modality: str = "vision",
    model_name: str = "claude-3",
    confidence: float = 0.80,
) -> None:
    """Insert a model_comparisons row with explicit modality."""
    conn.execute(
        """
        INSERT INTO model_comparisons (
            comparison_id, frame_id, model_name, prompt_hash,
            raw_response, parsed_output, schema_valid, confidence_overall,
            prompt_tokens, completion_tokens, total_tokens, api_latency_ms, modality
        ) VALUES (?, ?, ?, 'hash1', ?, '{"species":"dog"}', 1, ?, 5, 10, 15, 100, ?)
        """,
        (
            comparison_id,
            frame_id,
            model_name,
            json.dumps({"species": "dog", "confidence": confidence}),
            confidence,
            modality,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_cross_model_pairs_audio_raises_not_implemented(
    modality_conn: sqlite3.Connection,
) -> None:
    """Calling generate_cross_model_pairs with modality='audio' raises NotImplementedError."""
    store = AnnotationStore(conn=modality_conn)
    try:
        with pytest.raises(NotImplementedError, match="audio DPO"):
            generate_cross_model_pairs(store, "gpt-4o", "1.0", modality="audio")
    finally:
        store.close()


def test_generate_cross_model_pairs_default_modality_is_vision(
    modality_conn: sqlite3.Connection,
) -> None:
    """generate_cross_model_pairs default only returns vision pairs, not audio rows."""
    conn = modality_conn
    _insert_frame(conn, "f-vision")
    _insert_frame(conn, "f-audio")

    # Vision annotation + comparison → should produce a pair
    _insert_annotation(
        conn,
        "f-vision",
        "ann-vision",
        modality="vision",
        review_status="approved",
        model_name="gpt-4o",
        confidence=0.95,
    )
    _insert_comparison(
        conn, "f-vision", "cmp-vision", modality="vision", model_name="claude-3", confidence=0.80
    )

    # Audio annotation + comparison → should NOT appear in output
    _insert_annotation(
        conn,
        "f-audio",
        "ann-audio",
        modality="audio",
        review_status="approved",
        model_name="gpt-4o",
        confidence=0.95,
    )
    _insert_comparison(
        conn, "f-audio", "cmp-audio", modality="audio", model_name="claude-3", confidence=0.80
    )

    store = AnnotationStore(conn=conn)
    try:
        pairs = generate_cross_model_pairs(store, "gpt-4o", "1.0")
        # Audio frame should not appear in any pair
        frame_ids = {p["frame_id"] for p in pairs}
        assert "f-audio" not in frame_ids, "Audio frame should not appear in vision DPO pairs"
    finally:
        store.close()


def test_fetch_approved_annotations_filters_modality(
    modality_conn: sqlite3.Connection,
) -> None:
    """fetch_approved_annotations(modality='vision') returns only vision rows."""
    conn = modality_conn
    _insert_frame(conn, "fv")
    _insert_frame(conn, "fa")

    _insert_annotation(conn, "fv", "ann-v", modality="vision", review_status="approved")
    _insert_annotation(conn, "fa", "ann-a", modality="audio", review_status="approved")

    store = AnnotationStore(conn=conn)
    try:
        rows = store.fetch_approved_annotations(limit=100, modality="vision")
        ids = [r["annotation_id"] for r in rows]
        assert "ann-v" in ids
        assert "ann-a" not in ids
    finally:
        store.close()
