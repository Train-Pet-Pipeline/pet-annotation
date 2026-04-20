"""Tests for auto_check module."""

from __future__ import annotations

import uuid

from pet_annotation.quality.auto_check import run_auto_check
from pet_annotation.store import AnnotationRecord, AnnotationStore


def _insert_frame(conn, frame_id: str = "f1"):
    """Insert a minimal frame row."""
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) VALUES (?,?,?,?,?)",
        (frame_id, "v1", "selfshot", "f.jpg", "/data"),
    )
    conn.commit()


class TestAutoCheck:
    def test_approved_high_confidence(self, db_conn):
        """High-confidence valid annotations are auto-approved."""
        store = AnnotationStore(conn=db_conn)
        _insert_frame(db_conn, "f1")
        rec = AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            frame_id="f1",
            model_name="primary",
            prompt_hash="h1",
            raw_response="{}",
            schema_valid=True,
            confidence_overall=0.90,
        )
        store.insert_annotation(rec)
        store.update_frame_status_batch(["f1"], "auto_checked")

        run_auto_check(store, sampling_rate=0.0, threshold=0.70, primary_model="primary")

        ann = store.get_annotation("f1", "primary", "h1")
        assert ann.review_status == "approved"

    def test_needs_review_low_confidence(self, db_conn):
        """Low-confidence annotations are flagged for review."""
        store = AnnotationStore(conn=db_conn)
        _insert_frame(db_conn, "f1")
        rec = AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            frame_id="f1",
            model_name="primary",
            prompt_hash="h1",
            raw_response="{}",
            schema_valid=True,
            confidence_overall=0.50,
        )
        store.insert_annotation(rec)
        store.update_frame_status_batch(["f1"], "auto_checked")

        run_auto_check(store, sampling_rate=0.0, threshold=0.70, primary_model="primary")

        ann = store.get_annotation("f1", "primary", "h1")
        assert ann.review_status == "needs_review"
