"""Tests for migration 003 — audio_annotations table (Task B3)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pet_annotation.store import AnnotationStore

# Minimal frames schema required by AnnotationStore._recover_stuck_frames.
# Copied from test_migrations.py (YAGNI — no shared conftest needed yet).
_FRAMES_STUB_DDL = (
    "CREATE TABLE IF NOT EXISTS frames "
    "(frame_id TEXT PRIMARY KEY, annotation_status TEXT NOT NULL DEFAULT 'pending')"
)


def _seed_frames(db_file: Path) -> None:
    """Create a minimal frames table in *db_file* before AnnotationStore opens it."""
    conn = sqlite3.connect(str(db_file))
    conn.execute(_FRAMES_STUB_DDL)
    conn.commit()
    conn.close()


def _column_names(store: AnnotationStore, table: str) -> set[str]:
    """Return the set of column names for a given table."""
    rows = store._conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _index_names(store: AnnotationStore) -> set[str]:
    """Return all index names from sqlite_master."""
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_store(db_file: Path) -> AnnotationStore:
    """Seed frames then open AnnotationStore."""
    _seed_frames(db_file)
    return AnnotationStore(db_path=db_file)


# ---------------------------------------------------------------------------
# Tests — table existence and columns
# ---------------------------------------------------------------------------

def test_audio_annotations_table_exists(tmp_path: Path) -> None:
    """Migration 003 must create the audio_annotations table."""
    store = _open_store(tmp_path / "test.db")
    cols = _column_names(store, "audio_annotations")
    assert cols, "audio_annotations table should exist and have columns"
    store.close()


def test_audio_annotations_expected_columns(tmp_path: Path) -> None:
    """audio_annotations must have all columns defined in the spec."""
    store = _open_store(tmp_path / "test.db")
    cols = _column_names(store, "audio_annotations")
    expected = {
        "annotation_id",
        "sample_id",
        "annotator_type",
        "annotator_id",
        "modality",
        "created_at",
        "schema_version",
        "predicted_class",
        "class_probs",
        "logits",
    }
    missing = expected - cols
    assert not missing, f"Missing columns: {missing}; got: {cols}"
    store.close()


# ---------------------------------------------------------------------------
# Tests — indexes
# ---------------------------------------------------------------------------

def test_audio_annotations_indexes_exist(tmp_path: Path) -> None:
    """Migration 003 must create both supporting indexes."""
    store = _open_store(tmp_path / "test.db")
    indexes = _index_names(store)
    assert "idx_audio_ann_sample" in indexes, (
        f"'idx_audio_ann_sample' missing; got: {indexes}"
    )
    assert "idx_audio_ann_class" in indexes, (
        f"'idx_audio_ann_class' missing; got: {indexes}"
    )
    store.close()


# ---------------------------------------------------------------------------
# Tests — CHECK constraints
# ---------------------------------------------------------------------------

def test_audio_annotations_modality_check_rejects_vision(tmp_path: Path) -> None:
    """INSERT with modality='vision' must raise IntegrityError (CHECK constraint)."""
    store = _open_store(tmp_path / "test.db")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO audio_annotations
                (annotation_id, sample_id, annotator_type, annotator_id,
                 modality, schema_version, predicted_class, class_probs)
            VALUES ('aa1', 's1', 'cnn', 'cnn-v1', 'vision', '2.0.0', 'bark', '{}')
            """
        )
    store.close()


def test_audio_annotations_annotator_type_check_rejects_invalid(tmp_path: Path) -> None:
    """INSERT with annotator_type='bot' must raise IntegrityError (CHECK constraint)."""
    store = _open_store(tmp_path / "test.db")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO audio_annotations
                (annotation_id, sample_id, annotator_type, annotator_id,
                 schema_version, predicted_class, class_probs)
            VALUES ('aa2', 's1', 'bot', 'bot-v1', '2.0.0', 'bark', '{}')
            """
        )
    store.close()


def test_audio_annotations_annotator_type_valid_values(tmp_path: Path) -> None:
    """All four valid annotator_type values must insert without error."""
    store = _open_store(tmp_path / "test.db")
    for i, atype in enumerate(("vlm", "cnn", "human", "rule")):
        store._conn.execute(
            """
            INSERT INTO audio_annotations
                (annotation_id, sample_id, annotator_type, annotator_id,
                 schema_version, predicted_class, class_probs)
            VALUES (?, 's1', ?, 'some-id', '2.0.0', 'bark', '{}')
            """,
            (f"aa-valid-{i}", atype),
        )
    store._conn.commit()
    count = store._conn.execute(
        "SELECT COUNT(*) FROM audio_annotations"
    ).fetchone()[0]
    assert count == 4
    store.close()


# ---------------------------------------------------------------------------
# Tests — JSON round-trip
# ---------------------------------------------------------------------------

def test_audio_annotations_json_roundtrip(tmp_path: Path) -> None:
    """Insert a row with class_probs and logits JSON; SELECT and parse back correctly."""
    store = _open_store(tmp_path / "test.db")

    class_probs_in = {"bark": 0.9, "silence": 0.1}
    logits_in = [0.1, 0.2, 0.3]

    store._conn.execute(
        """
        INSERT INTO audio_annotations
            (annotation_id, sample_id, annotator_type, annotator_id,
             schema_version, predicted_class, class_probs, logits)
        VALUES ('aa-rt', 's42', 'cnn', 'cnn-v2', '2.0.0', 'bark', ?, ?)
        """,
        (json.dumps(class_probs_in), json.dumps(logits_in)),
    )
    store._conn.commit()

    row = store._conn.execute(
        "SELECT class_probs, logits FROM audio_annotations WHERE annotation_id='aa-rt'"
    ).fetchone()
    assert row is not None, "Row not found after INSERT"

    class_probs_out = json.loads(row[0])
    logits_out = json.loads(row[1])

    assert class_probs_out == class_probs_in, (
        f"class_probs mismatch: {class_probs_out!r} != {class_probs_in!r}"
    )
    assert logits_out == logits_in, (
        f"logits mismatch: {logits_out!r} != {logits_in!r}"
    )
    store.close()


def test_audio_annotations_logits_nullable(tmp_path: Path) -> None:
    """logits column must accept NULL."""
    store = _open_store(tmp_path / "test.db")
    store._conn.execute(
        """
        INSERT INTO audio_annotations
            (annotation_id, sample_id, annotator_type, annotator_id,
             schema_version, predicted_class, class_probs, logits)
        VALUES ('aa-null-logits', 's1', 'human', 'human-1', '2.0.0', 'silence', '{}', NULL)
        """
    )
    store._conn.commit()
    row = store._conn.execute(
        "SELECT logits FROM audio_annotations WHERE annotation_id='aa-null-logits'"
    ).fetchone()
    assert row is not None
    assert row[0] is None, f"Expected NULL logits, got: {row[0]!r}"
    store.close()


def test_audio_annotations_modality_default_is_audio(tmp_path: Path) -> None:
    """modality column must default to 'audio' when omitted."""
    store = _open_store(tmp_path / "test.db")
    store._conn.execute(
        """
        INSERT INTO audio_annotations
            (annotation_id, sample_id, annotator_type, annotator_id,
             schema_version, predicted_class, class_probs)
        VALUES ('aa-default-mod', 's1', 'rule', 'rule-v1', '2.0.0', 'bark', '{}')
        """
    )
    store._conn.commit()
    row = store._conn.execute(
        "SELECT modality FROM audio_annotations WHERE annotation_id='aa-default-mod'"
    ).fetchone()
    assert row is not None
    assert row[0] == "audio", f"Expected default modality 'audio', got: {row[0]!r}"
    store.close()
