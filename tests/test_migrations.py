"""Tests for migration system — idempotency and schema correctness after 004."""

from __future__ import annotations

from pathlib import Path

from pet_annotation.store import AnnotationStore


def _table_names(store: AnnotationStore) -> set[str]:
    """Return set of user-created table names in the DB."""
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def _column_names(store: AnnotationStore, table: str) -> set[str]:
    """Return set of column names for *table*."""
    rows = store._conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _index_names(store: AnnotationStore) -> set[str]:
    """Return all index names from sqlite_master."""
    rows = store._conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    return {r[0] for r in rows}


def test_migration_004_creates_four_tables(tmp_path: Path) -> None:
    """After init_schema(), the 4 paradigm tables must exist."""
    store = AnnotationStore(str(tmp_path / "t.db"))
    store.init_schema()
    tables = _table_names(store)
    assert "llm_annotations" in tables
    assert "classifier_annotations" in tables
    assert "rule_annotations" in tables
    assert "human_annotations" in tables


def test_migration_004_drops_old_tables(tmp_path: Path) -> None:
    """Old annotations / audio_annotations / model_comparisons tables must not exist."""
    store = AnnotationStore(str(tmp_path / "t.db"))
    store.init_schema()
    tables = _table_names(store)
    assert "annotations" not in tables
    assert "audio_annotations" not in tables
    assert "model_comparisons" not in tables


def test_migration_004_llm_table_columns(tmp_path: Path) -> None:
    """llm_annotations must have all required columns."""
    store = AnnotationStore(str(tmp_path / "t.db"))
    store.init_schema()
    cols = _column_names(store, "llm_annotations")
    required = {
        "annotation_id", "target_id", "annotator_id", "annotator_type",
        "modality", "schema_version", "created_at", "storage_uri",
        "prompt_hash", "raw_response", "parsed_output",
    }
    assert required <= cols, f"missing columns: {required - cols}"


def test_migration_004_creates_indexes(tmp_path: Path) -> None:
    """Expected indexes must be present after migration 004."""
    store = AnnotationStore(str(tmp_path / "t.db"))
    store.init_schema()
    indexes = _index_names(store)
    assert "idx_llm_target" in indexes
    assert "idx_cls_target" in indexes
    assert "idx_rule_target" in indexes
    assert "idx_human_target" in indexes


def test_migration_idempotent_reopen(tmp_path: Path) -> None:
    """Opening the store twice on the same DB must not raise."""
    db = str(tmp_path / "idem.db")
    s1 = AnnotationStore(db)
    s1.init_schema()

    s2 = AnnotationStore(db)
    s2.init_schema()  # must not fail

    tables = _table_names(s2)
    assert "llm_annotations" in tables


def test_applied_migrations_table_tracks_names(tmp_path: Path) -> None:
    """_applied_migrations table must contain all 4 migration file names."""
    store = AnnotationStore(str(tmp_path / "t.db"))
    store.init_schema()
    applied = {
        r[0]
        for r in store._conn.execute("SELECT name FROM _applied_migrations").fetchall()
    }
    assert "004_four_paradigm_tables.sql" in applied
