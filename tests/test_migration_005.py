"""Tests for migration 005 — annotation_targets table."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pet_annotation.store import AnnotationStore


def _table_names(store: AnnotationStore) -> set[str]:
    """Return set of user-created table names in the DB."""
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def _column_info(store: AnnotationStore, table: str) -> dict[str, dict]:
    """Return column metadata keyed by column name."""
    rows = store._conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1]: {"notnull": r[3], "pk": r[5], "type": r[2]} for r in rows}


def _index_names(store: AnnotationStore) -> set[str]:
    """Return all index names from sqlite_master."""
    rows = store._conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    return {r[0] for r in rows}


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    """Fresh AnnotationStore with full schema including migration 005."""
    s = AnnotationStore(str(tmp_path / "t.db"))
    s.init_schema()
    return s


def test_migration_005_creates_annotation_targets_table(store: AnnotationStore) -> None:
    """annotation_targets table must exist after init_schema()."""
    assert "annotation_targets" in _table_names(store)


def test_migration_005_column_schema(store: AnnotationStore) -> None:
    """annotation_targets must have all required columns with correct NOT NULL."""
    cols = _column_info(store, "annotation_targets")
    required_cols = {
        "target_id", "annotator_id", "annotator_type", "state",
        "claimed_at", "finished_at", "error_msg",
    }
    assert required_cols <= set(cols.keys()), f"missing: {required_cols - set(cols.keys())}"

    # NOT NULL columns
    for col in ("target_id", "annotator_id", "annotator_type", "state"):
        assert cols[col]["notnull"] == 1, f"{col} must be NOT NULL"

    # Nullable columns
    for col in ("claimed_at", "finished_at", "error_msg"):
        assert cols[col]["notnull"] == 0, f"{col} must be nullable"


def test_migration_005_composite_primary_key(store: AnnotationStore) -> None:
    """(target_id, annotator_id) must be the composite primary key."""
    cols = _column_info(store, "annotation_targets")
    pk_cols = {name for name, info in cols.items() if info["pk"] > 0}
    assert pk_cols == {"target_id", "annotator_id"}


def test_migration_005_indexes_created(store: AnnotationStore) -> None:
    """idx_targets_state and idx_targets_type indexes must exist."""
    indexes = _index_names(store)
    assert "idx_targets_state" in indexes
    assert "idx_targets_type" in indexes


def test_migration_005_check_state_constraint(store: AnnotationStore) -> None:
    """CHECK(state IN ...) must reject invalid state values."""
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO annotation_targets(target_id, annotator_id, annotator_type, state) "
            "VALUES (?, ?, ?, ?)",
            ("t1", "a1", "llm", "unknown_state"),
        )
        store._conn.commit()


def test_migration_005_check_annotator_type_constraint(store: AnnotationStore) -> None:
    """CHECK(annotator_type IN ...) must reject invalid annotator_type values."""
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO annotation_targets(target_id, annotator_id, annotator_type, state) "
            "VALUES (?, ?, ?, ?)",
            ("t1", "a1", "robot", "pending"),
        )
        store._conn.commit()


def test_migration_005_valid_insert_all_states(store: AnnotationStore) -> None:
    """All valid state values and annotator_type values must insert without error."""
    for i, state in enumerate(("pending", "in_progress", "done", "failed")):
        store._conn.execute(
            "INSERT INTO annotation_targets(target_id, annotator_id, annotator_type, state) "
            "VALUES (?, ?, ?, ?)",
            (f"t{i}", "ann-1", "llm", state),
        )
    for i, atype in enumerate(("llm", "classifier", "rule", "human")):
        store._conn.execute(
            "INSERT INTO annotation_targets(target_id, annotator_id, annotator_type, state) "
            "VALUES (?, ?, ?, ?)",
            (f"type-{i}", atype, atype, "pending"),
        )
    store._conn.commit()
    count = store._conn.execute("SELECT COUNT(*) FROM annotation_targets").fetchone()[0]
    assert count == 8


def test_migration_005_pk_rejects_duplicate(store: AnnotationStore) -> None:
    """PRIMARY KEY (target_id, annotator_id) must reject duplicates."""
    store._conn.execute(
        "INSERT INTO annotation_targets(target_id, annotator_id, annotator_type, state) "
        "VALUES ('t1', 'a1', 'llm', 'pending')"
    )
    store._conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO annotation_targets(target_id, annotator_id, annotator_type, state) "
            "VALUES ('t1', 'a1', 'llm', 'pending')"
        )
        store._conn.commit()


def test_migration_005_applied_migration_recorded(store: AnnotationStore) -> None:
    """005_add_annotation_targets.sql must appear in _applied_migrations."""
    applied = {
        r[0]
        for r in store._conn.execute("SELECT name FROM _applied_migrations").fetchall()
    }
    assert "005_add_annotation_targets.sql" in applied
