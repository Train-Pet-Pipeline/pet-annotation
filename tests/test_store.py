"""Tests for AnnotationStore (TDD)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from pet_annotation.store import (
    AnnotationRecord,
    AnnotationStore,
    AudioAnnotationRow,
    ComparisonRecord,
    VisionAnnotationRow,
)

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


def _make_annotation(
    frame_id: str = "f1", model_name: str = "gpt-4o", prompt_hash: str = "abc123"
) -> AnnotationRecord:
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


def _make_comparison(
    frame_id: str = "f1", model_name: str = "gpt-4o", prompt_hash: str = "abc123"
) -> ComparisonRecord:
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
        for row in db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
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
    row = db_conn.execute("SELECT annotation_status FROM frames WHERE frame_id='f1'").fetchone()
    assert row["annotation_status"] == "annotating"
    store.close()


def test_recover_annotating_on_init(db_conn: sqlite3.Connection) -> None:
    """Frames stuck in 'annotating' are reset to 'pending' on store init."""
    _insert_frame(db_conn, "stuck1", "annotating")
    _insert_frame(db_conn, "stuck2", "annotating")
    _insert_frame(db_conn, "normal", "pending")

    AnnotationStore(conn=db_conn).close()

    rows = db_conn.execute("SELECT frame_id, annotation_status FROM frames").fetchall()
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
    rec_app = AnnotationRecord(
        **{**rec_app.__dict__, "annotation_id": "ann-app", "review_status": "approved"}
    )
    store.insert_annotation(rec_app)

    rec_rev = _make_annotation("f-rev", prompt_hash="h2")
    rec_rev = AnnotationRecord(
        **{**rec_rev.__dict__, "annotation_id": "ann-rev", "review_status": "reviewed"}
    )
    store.insert_annotation(rec_rev)

    rec_pen = _make_annotation("f-pen", prompt_hash="h3")
    rec_pen = AnnotationRecord(
        **{**rec_pen.__dict__, "annotation_id": "ann-pen", "review_status": "pending"}
    )
    store.insert_annotation(rec_pen)

    rows = store.fetch_approved_annotations(limit=10)
    ids = [r["annotation_id"] for r in rows]
    assert "ann-app" in ids
    assert "ann-rev" in ids
    assert "ann-pen" not in ids
    store.close()


def test_insert_annotation_modality_default(db_conn: sqlite3.Connection) -> None:
    """Inserted annotation has modality='vision' by default (migration 002 column)."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)
    store.insert_annotation(_make_annotation("f1"))

    row = db_conn.execute("SELECT modality FROM annotations WHERE frame_id='f1'").fetchone()
    assert row is not None
    assert row["modality"] == "vision"
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


# ---------------------------------------------------------------------------
# B4 tests — VisionAnnotationRow rename + AudioAnnotationRow routing
# ---------------------------------------------------------------------------


def test_annotation_record_alias_is_vision_row() -> None:
    """AnnotationRecord must be the same class as VisionAnnotationRow (back-compat alias)."""
    assert AnnotationRecord is VisionAnnotationRow


def test_vision_annotation_row_has_modality_field() -> None:
    """VisionAnnotationRow must have a modality field defaulting to 'vision'."""
    row = VisionAnnotationRow(
        annotation_id="ann-1",
        frame_id="f1",
        model_name="gpt-4o",
        prompt_hash="hash1",
        raw_response="{}",
        schema_valid=1,
    )
    assert row.modality == "vision"


def test_vision_annotation_row_has_storage_uri_field() -> None:
    """VisionAnnotationRow must have a storage_uri field defaulting to None."""
    row = VisionAnnotationRow(
        annotation_id="ann-1",
        frame_id="f1",
        model_name="gpt-4o",
        prompt_hash="hash1",
        raw_response="{}",
        schema_valid=1,
    )
    assert row.storage_uri is None


def test_insert_vision_annotation_writes_modality_column(db_conn: sqlite3.Connection) -> None:
    """insert_annotation with VisionAnnotationRow writes modality='vision' and storage_uri."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)
    row = VisionAnnotationRow(
        annotation_id="ann-vision-1",
        frame_id="f1",
        model_name="gpt-4o",
        prompt_hash="hash1",
        raw_response='{"ok": true}',
        schema_valid=1,
        storage_uri="gs://bucket/frame-001.jpg",
    )
    store.insert_annotation(row)

    db_row = db_conn.execute(
        "SELECT modality, storage_uri FROM annotations WHERE annotation_id='ann-vision-1'"
    ).fetchone()
    assert db_row is not None
    assert db_row["modality"] == "vision"
    assert db_row["storage_uri"] == "gs://bucket/frame-001.jpg"
    store.close()


def test_insert_audio_annotation_routes_to_audio_table(db_conn: sqlite3.Connection) -> None:
    """insert_annotation with AudioAnnotationRow writes to audio_annotations, not annotations."""
    store = AnnotationStore(conn=db_conn)
    row = AudioAnnotationRow(
        annotation_id="ann-audio-1",
        sample_id="sample-001",
        annotator_type="cnn",
        annotator_id="cnn-v2",
        predicted_class="bark",
        class_probs=json.dumps({"bark": 0.9, "silence": 0.1}),
    )
    store.insert_annotation(row)

    # Must appear in audio_annotations
    audio_row = db_conn.execute(
        "SELECT * FROM audio_annotations WHERE annotation_id='ann-audio-1'"
    ).fetchone()
    assert audio_row is not None
    assert audio_row["sample_id"] == "sample-001"
    assert audio_row["predicted_class"] == "bark"

    # Must NOT appear in annotations (vision table)
    ann_row = db_conn.execute(
        "SELECT * FROM annotations WHERE annotation_id='ann-audio-1'"
    ).fetchone()
    assert ann_row is None
    store.close()


def test_insert_annotation_and_update_status_raises_for_audio(db_conn: sqlite3.Connection) -> None:
    """insert_annotation_and_update_status must raise ValueError for AudioAnnotationRow."""
    store = AnnotationStore(conn=db_conn)
    row = AudioAnnotationRow(
        annotation_id="ann-audio-2",
        sample_id="sample-002",
        annotator_type="cnn",
        annotator_id="cnn-v2",
        predicted_class="silence",
        class_probs=json.dumps({"silence": 1.0}),
    )
    with pytest.raises(ValueError, match="vision-only"):
        store.insert_annotation_and_update_status(row, "annotating")  # type: ignore[arg-type]
    store.close()


def test_get_annotation_vision_returns_vision_row(db_conn: sqlite3.Connection) -> None:
    """get_annotation with modality='vision' returns a VisionAnnotationRow."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)
    rec = _make_annotation("f1")
    store.insert_annotation(rec)

    result = store.get_annotation("f1", "gpt-4o", "abc123", modality="vision")
    assert isinstance(result, VisionAnnotationRow)
    store.close()


def test_get_annotation_audio_returns_audio_row(db_conn: sqlite3.Connection) -> None:
    """get_annotation with modality='audio' returns an AudioAnnotationRow."""
    store = AnnotationStore(conn=db_conn)
    row = AudioAnnotationRow(
        annotation_id="ann-audio-get",
        sample_id="sample-get",
        annotator_type="human",
        annotator_id="human-1",
        predicted_class="bark",
        class_probs=json.dumps({"bark": 1.0}),
    )
    store.insert_annotation(row)

    result = store.get_annotation("sample-get", modality="audio")
    assert isinstance(result, AudioAnnotationRow)
    assert result.annotation_id == "ann-audio-get"
    assert result.predicted_class == "bark"
    store.close()


def test_comparison_record_has_modality_field() -> None:
    """ComparisonRecord must have a modality field defaulting to 'vision'."""
    rec = ComparisonRecord(
        comparison_id="cmp-1",
        frame_id="f1",
        model_name="gpt-4o",
        prompt_hash="hash1",
        raw_response="{}",
        schema_valid=1,
    )
    assert rec.modality == "vision"


def test_insert_comparison_writes_modality_column(db_conn: sqlite3.Connection) -> None:
    """insert_comparison persists the modality column to model_comparisons."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)
    rec = ComparisonRecord(
        comparison_id="cmp-modality-1",
        frame_id="f1",
        model_name="claude-3-5",
        prompt_hash="hash2",
        raw_response='{"ok": true}',
        schema_valid=1,
    )
    store.insert_comparison(rec)

    db_row = db_conn.execute(
        "SELECT modality FROM model_comparisons WHERE comparison_id='cmp-modality-1'"
    ).fetchone()
    assert db_row is not None
    assert db_row["modality"] == "vision"
    store.close()


def test_get_comparison_returns_stored_modality(db_conn: sqlite3.Connection) -> None:
    """get_comparison populates modality from the DB row, not the dataclass default."""
    _insert_frame(db_conn, "f1")
    store = AnnotationStore(conn=db_conn)
    rec = ComparisonRecord(
        comparison_id="cmp-modal-read",
        frame_id="f1",
        model_name="gpt-4o",
        prompt_hash="hash-modal",
        raw_response='{"ok": true}',
        schema_valid=1,
        modality="vision",
    )
    store.insert_comparison(rec)

    got = store.get_comparison("f1", "gpt-4o", "hash-modal")
    assert got is not None
    assert got.modality == "vision"
    store.close()
