"""Tests for migration globbing and idempotency (Task B2)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from pet_annotation.store import AnnotationStore

# Minimal frames schema required by AnnotationStore._recover_stuck_frames.
# frame_id is the FK target for annotations; annotation_status is updated on recovery.
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


def test_migration_002_adds_columns(tmp_path: Path) -> None:
    """Fresh DB should have modality + storage_uri on annotations; modality on model_comparisons."""
    db_file = tmp_path / "test.db"
    _seed_frames(db_file)
    store = AnnotationStore(db_path=db_file)

    ann_cols = _column_names(store, "annotations")
    assert "modality" in ann_cols, f"'modality' missing from annotations; got: {ann_cols}"
    assert "storage_uri" in ann_cols, f"'storage_uri' missing from annotations; got: {ann_cols}"

    cmp_cols = _column_names(store, "model_comparisons")
    assert "modality" in cmp_cols, f"'modality' missing from model_comparisons; got: {cmp_cols}"

    store.close()


def test_migration_002_creates_index(tmp_path: Path) -> None:
    """Fresh DB should have idx_annotations_modality index."""
    db_file = tmp_path / "test.db"
    _seed_frames(db_file)
    store = AnnotationStore(db_path=db_file)

    indexes = _index_names(store)
    assert "idx_annotations_modality" in indexes, (
        f"'idx_annotations_modality' missing; got: {indexes}"
    )

    store.close()


def test_migration_002_modality_default(tmp_path: Path) -> None:
    """The modality column defaults to 'vision'."""
    db_file = tmp_path / "test.db"
    _seed_frames(db_file)
    store = AnnotationStore(db_path=db_file)

    store._conn.execute("INSERT INTO frames (frame_id) VALUES ('f1')")
    store._conn.execute(
        """
        INSERT INTO annotations
            (annotation_id, frame_id, model_name, prompt_hash, raw_response, schema_valid)
        VALUES ('a1', 'f1', 'gpt-4o', 'h1', '{}', 1)
        """
    )
    store._conn.commit()

    row = store._conn.execute(
        "SELECT modality FROM annotations WHERE annotation_id='a1'"
    ).fetchone()
    assert row[0] == "vision", f"Expected default 'vision', got: {row[0]}"

    store.close()


def test_migration_idempotent_reopen(tmp_path: Path) -> None:
    """Opening the store twice on the same DB must not raise (idempotent re-apply)."""
    db_file = tmp_path / "test.db"
    _seed_frames(db_file)

    store1 = AnnotationStore(db_path=db_file)
    store1.close()

    # Second open should re-run all migrations without error
    store2 = AnnotationStore(db_path=db_file)
    ann_cols = _column_names(store2, "annotations")
    assert "modality" in ann_cols
    store2.close()
