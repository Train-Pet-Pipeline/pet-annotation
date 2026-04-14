"""Tests for AnnotationStore (TDD)."""
from __future__ import annotations

import sqlite3

import pytest

from pet_annotation.store import AnnotationRecord, AnnotationStore, ComparisonRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_frame(conn: sqlite3.Connection, frame_id: str, status: str = "pending") -> None:
    """Insert a minimal frame row for testing."""
    conn.execute(
        """
        INSERT INTO frames (frame_id, video_id, source, frame_path, data_root, annotation_status)
        VALUES (?, 'vid1', 'test', '/tmp/f.jpg', '/tmp', ?)
        """,
        (frame_id, status),
    )
    conn.commit()


def _make_annotation(frame_id: str = "f1", model_name: str = "gpt-4o",
                     prompt_hash: str = "abc123") -> AnnotationRecord:
    """Build a minimal AnnotationRecord."""
    return AnnotationRecord(
        annotation_id=f"ann-{frame_id}-{model_name}",
        frame_id=frame_id,
        model_name=model_name,
        prompt_hash=prompt_hash,
        raw_response='{"ok": true}',
        parsed_output='{"species": "cat"}',
        schema_valid=1,
        validation_errors=None,
        confidence_overall=0.95,
        review_status="pending",
        reviewer=None,
        review_notes=None,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        api_latency_ms=250,
    )


def _make_comparison(frame_id: str = "f1", model_name: str = "gpt-4o",
                     prompt_hash: str = "abc123") -> ComparisonRecord:
    """Build a minimal ComparisonRecord."""
    return ComparisonRecord(
        comparison_id=f"cmp-{frame_id}-{model_name}",
        frame_id=frame_id,
        model_name=model_name,
        prompt_hash=prompt_hash,
        raw_response='{"ok": true}',
        parsed_output='{"species": "dog"}',
        schema_valid=1,
        validation_errors=None,
        confidence_overall=0.8,
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
        api_latency_ms=100,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_creates_tables(db_conn: sqlite3.Connection) -> None:
    """Tables annotations and model_comparisons exist after store init."""
    store = AnnotationStore(conn=db_conn)
    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "annotations" in tables
    assert "model_comparisons" in tables
    store.close()


def test_insert_annotation(db_conn: sqlite3.Connection) -> None:
    """Inserted annotation can be retrieved by cache key."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)
    rec = _make_annotation("f1")
    store.insert_annotation(rec)

    got = store.get_annotation("f1", "gpt-4o", "abc123")
    assert got is not None
    assert got.annotation_id == rec.annotation_id
    assert got.confidence_overall == pytest.approx(0.95)
    store.close()


def test_cache_hit(db_conn: sqlite3.Connection) -> None:
    """cache_hit returns False before insert, True after."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)

    assert store.cache_hit("f1", "gpt-4o", "abc123") is False
    store.insert_annotation(_make_annotation("f1"))
    assert store.cache_hit("f1", "gpt-4o", "abc123") is True
    store.close()


def test_insert_comparison(db_conn: sqlite3.Connection) -> None:
    """Inserted comparison can be retrieved by cache key."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)
    rec = _make_comparison("f1")
    store.insert_comparison(rec)

    got = store.get_comparison("f1", "gpt-4o", "abc123")
    assert got is not None
    assert got.comparison_id == rec.comparison_id
    assert got.confidence_overall == pytest.approx(0.8)
    store.close()


def test_fetch_pending_frames(db_conn: sqlite3.Connection) -> None:
    """fetch_pending_frames only returns frames with annotation_status='pending'."""
    # Create the store first so that the recovery pass runs before we insert test frames.
    store = AnnotationStore(conn=db_conn)

    _insert_frame(db_conn, "f-pending", "pending")
    # Insert directly with annotation_status='annotating' to simulate a mid-run state
    # that was NOT caused by this store's startup (i.e. written after init recovery).
    db_conn.execute(
        """
        INSERT INTO frames (frame_id, video_id, source, frame_path, data_root, annotation_status)
        VALUES ('f-annotating', 'vid1', 'test', '/tmp/f.jpg', '/tmp', 'annotating')
        """
    )
    db_conn.commit()
    _insert_frame(db_conn, "f-approved", "approved")

    rows = store.fetch_pending_frames(limit=10)
    ids = [r["frame_id"] for r in rows]
    assert "f-pending" in ids
    assert "f-annotating" not in ids
    assert "f-approved" not in ids
    store.close()


def test_update_annotation_status_atomic(db_conn: sqlite3.Connection) -> None:
    """insert_annotation_and_update_status atomically inserts and updates frame."""
    _insert_frame(db_conn, "f1", "pending")
    store = AnnotationStore(conn=db_conn)
    rec = _make_annotation("f1")
    store.insert_annotation_and_update_status(rec, "annotating")

    # annotation inserted
    assert store.cache_hit("f1", "gpt-4o", "abc123") is True
    # frame status updated
    row = db_conn.execute(
        "SELECT annotation_status FROM frames WHERE frame_id='f1'"
    ).fetchone()
    assert row["annotation_status"] == "annotating"
    store.close()


def test_recover_annotating_on_init(db_conn: sqlite3.Connection) -> None:
    """Frames stuck in 'annotating' are reset to 'pending' on store init."""
    _insert_frame(db_conn, "stuck1", "annotating")
    _insert_frame(db_conn, "stuck2", "annotating")
    _insert_frame(db_conn, "normal", "pending")

    AnnotationStore(conn=db_conn).close()

    rows = db_conn.execute(
        "SELECT frame_id, annotation_status FROM frames"
    ).fetchall()
    status_map = {r["frame_id"]: r["annotation_status"] for r in rows}
    assert status_map["stuck1"] == "pending"
    assert status_map["stuck2"] == "pending"
    assert status_map["normal"] == "pending"


def test_fetch_approved_annotations(db_conn: sqlite3.Connection) -> None:
    """fetch_approved_annotations JOINs frames and returns only approved/reviewed."""
    _insert_frame(db_conn, "f-app", "approved")
    _insert_frame(db_conn, "f-rev", "reviewed")
    _insert_frame(db_conn, "f-pen", "pending")

    store = AnnotationStore(conn=db_conn)

    rec_app = _make_annotation("f-app", prompt_hash="h1")
    rec_app = AnnotationRecord(**{**rec_app.__dict__, "annotation_id": "ann-app",
                                  "review_status": "approved"})
    store.insert_annotation(rec_app)

    rec_rev = _make_annotation("f-rev", prompt_hash="h2")
    rec_rev = AnnotationRecord(**{**rec_rev.__dict__, "annotation_id": "ann-rev",
                                  "review_status": "reviewed"})
    store.insert_annotation(rec_rev)

    rec_pen = _make_annotation("f-pen", prompt_hash="h3")
    rec_pen = AnnotationRecord(**{**rec_pen.__dict__, "annotation_id": "ann-pen",
                                  "review_status": "pending"})
    store.insert_annotation(rec_pen)

    rows = store.fetch_approved_annotations(limit=10)
    ids = [r["annotation_id"] for r in rows]
    assert "ann-app" in ids
    assert "ann-rev" in ids
    assert "ann-pen" not in ids
    store.close()


def test_update_review_and_frame_status(db_conn: sqlite3.Connection) -> None:
    """update_review_and_frame_status atomically updates both tables."""
    _insert_frame(db_conn, "f1", "pending")
    store = AnnotationStore(conn=db_conn)
    rec = _make_annotation("f1")
    store.insert_annotation(rec)

    store.update_review_and_frame_status(rec.annotation_id, "approved", "f1", "approved")

    ann = store.get_annotation("f1", "gpt-4o", "abc123")
    assert ann is not None
    assert ann.review_status == "approved"

    frame_row = db_conn.execute(
        "SELECT annotation_status FROM frames WHERE frame_id='f1'"
    ).fetchone()
    assert frame_row["annotation_status"] == "approved"
    store.close()
