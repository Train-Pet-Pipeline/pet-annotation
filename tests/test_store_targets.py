"""Tests for AnnotationStore pending-target API (Phase 4 orchestrator wire).

Covers: ingest_pending_from_petdata, claim_pending_targets,
        mark_target_done, mark_target_failed, get_target_state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pet_annotation.store import AnnotationStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_petdata_db(path: Path, pending_ids: list[str], non_pending_ids: list[str]) -> str:
    """Create a minimal pet-data SQLite fixture with a frames table.

    Args:
        path: Directory to write the db file into.
        pending_ids: frame_id values with annotation_status='pending'.
        non_pending_ids: frame_id values with annotation_status='done'.

    Returns:
        Absolute path string to the created db file.
    """
    db_path = str(path / "petdata.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE frames (frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT)"
    )
    for fid in pending_ids:
        conn.execute(
            "INSERT INTO frames VALUES (?, 'pending', 'vision')", (fid,)
        )
    for fid in non_pending_ids:
        conn.execute(
            "INSERT INTO frames VALUES (?, 'done', 'vision')", (fid,)
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    """Fresh AnnotationStore with full schema."""
    s = AnnotationStore(str(tmp_path / "ann.db"))
    s.init_schema()
    return s


@pytest.fixture
def petdata_db(tmp_path: Path) -> str:
    """pet-data fixture: 5 pending + 2 non-pending frames."""
    return _make_petdata_db(
        tmp_path,
        pending_ids=[f"frame-{i}" for i in range(5)],
        non_pending_ids=["done-1", "done-2"],
    )


# ---------------------------------------------------------------------------
# ingest_pending_from_petdata
# ---------------------------------------------------------------------------


def test_ingest_creates_rows_for_all_annotators(
    store: AnnotationStore, petdata_db: str
) -> None:
    """5 pending frames × 2 annotator_ids should create 10 new annotation_targets rows."""
    count = store.ingest_pending_from_petdata(
        petdata_db, ["ann-a", "ann-b"], annotator_type="llm"
    )
    assert count == 10

    total = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets"
    ).fetchone()[0]
    assert total == 10


def test_ingest_skips_non_pending_frames(store: AnnotationStore, petdata_db: str) -> None:
    """Non-pending frames (annotation_status != 'pending') must not be ingested."""
    count = store.ingest_pending_from_petdata(
        petdata_db, ["ann-a"], annotator_type="llm"
    )
    assert count == 5  # only 5 pending frames, not the 2 done ones


def test_ingest_is_idempotent(store: AnnotationStore, petdata_db: str) -> None:
    """Re-running ingest on the same DB must not insert duplicate rows (INSERT OR IGNORE)."""
    count1 = store.ingest_pending_from_petdata(
        petdata_db, ["ann-a"], annotator_type="llm"
    )
    count2 = store.ingest_pending_from_petdata(
        petdata_db, ["ann-a"], annotator_type="llm"
    )
    assert count1 == 5
    assert count2 == 0  # already tracked

    total = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets"
    ).fetchone()[0]
    assert total == 5


def test_ingest_initial_state_is_pending(store: AnnotationStore, petdata_db: str) -> None:
    """All ingested rows must have state='pending'."""
    store.ingest_pending_from_petdata(petdata_db, ["ann-a"], annotator_type="llm")
    states = {
        r[0]
        for r in store._conn.execute(
            "SELECT DISTINCT state FROM annotation_targets"
        ).fetchall()
    }
    assert states == {"pending"}


def test_ingest_modality_filter(tmp_path: Path, store: AnnotationStore) -> None:
    """modality filter passes through correctly to the WHERE clause."""
    db_path = str(tmp_path / "modality_petdata.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE frames (frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT)"
    )
    conn.execute("INSERT INTO frames VALUES ('v1', 'pending', 'vision')")
    conn.execute("INSERT INTO frames VALUES ('a1', 'pending', 'audio')")
    conn.commit()
    conn.close()

    count = store.ingest_pending_from_petdata(
        db_path, ["ann-a"], annotator_type="llm", modality="vision"
    )
    assert count == 1
    state = store.get_target_state("v1", "ann-a")
    assert state == "pending"
    assert store.get_target_state("a1", "ann-a") is None


# ---------------------------------------------------------------------------
# claim_pending_targets
# ---------------------------------------------------------------------------


def test_claim_returns_correct_count_and_changes_state(
    store: AnnotationStore, petdata_db: str
) -> None:
    """claim_pending_targets must return target_ids and transition state to in_progress."""
    store.ingest_pending_from_petdata(petdata_db, ["ann-a"], annotator_type="llm")
    claimed = store.claim_pending_targets("ann-a", batch_size=3)
    assert len(claimed) == 3
    for tid in claimed:
        assert store.get_target_state(tid, "ann-a") == "in_progress"


def test_claim_leaves_unclaimed_as_pending(
    store: AnnotationStore, petdata_db: str
) -> None:
    """Remaining targets not in the batch must stay pending."""
    store.ingest_pending_from_petdata(petdata_db, ["ann-a"], annotator_type="llm")
    claimed = store.claim_pending_targets("ann-a", batch_size=3)
    pending_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE annotator_id='ann-a' AND state='pending'"
    ).fetchone()[0]
    assert pending_count == 5 - len(claimed)


def test_claim_empty_when_none_pending(store: AnnotationStore) -> None:
    """claim_pending_targets returns [] when no pending targets exist."""
    result = store.claim_pending_targets("ann-a", batch_size=10)
    assert result == []


def test_claim_disjoint_across_two_annotators(
    store: AnnotationStore, petdata_db: str
) -> None:
    """Two annotator_ids must claim disjoint target sets (same frame, different annotators)."""
    store.ingest_pending_from_petdata(
        petdata_db, ["ann-a", "ann-b"], annotator_type="llm"
    )
    claimed_a = store.claim_pending_targets("ann-a", batch_size=5)
    claimed_b = store.claim_pending_targets("ann-b", batch_size=5)

    # Both should have claimed all 5 frames (for their own annotator rows)
    assert len(claimed_a) == 5
    assert len(claimed_b) == 5

    # No overlap in in_progress for ann-a vs ann-b rows (they're different rows)
    for tid in claimed_a:
        assert store.get_target_state(tid, "ann-a") == "in_progress"
    for tid in claimed_b:
        assert store.get_target_state(tid, "ann-b") == "in_progress"


# ---------------------------------------------------------------------------
# mark_target_done / mark_target_failed
# ---------------------------------------------------------------------------


def test_mark_target_done_transitions_state(
    store: AnnotationStore, petdata_db: str
) -> None:
    """mark_target_done must transition state to 'done' and set finished_at."""
    store.ingest_pending_from_petdata(petdata_db, ["ann-a"], annotator_type="llm")
    claimed = store.claim_pending_targets("ann-a", batch_size=1)
    tid = claimed[0]

    store.mark_target_done(tid, "ann-a")
    assert store.get_target_state(tid, "ann-a") == "done"

    row = store._conn.execute(
        "SELECT finished_at FROM annotation_targets WHERE target_id=? AND annotator_id='ann-a'",
        (tid,),
    ).fetchone()
    assert row[0] is not None  # finished_at set


def test_mark_target_failed_sets_error_msg(
    store: AnnotationStore, petdata_db: str
) -> None:
    """mark_target_failed must set state='failed', finished_at, and error_msg."""
    store.ingest_pending_from_petdata(petdata_db, ["ann-a"], annotator_type="llm")
    claimed = store.claim_pending_targets("ann-a", batch_size=1)
    tid = claimed[0]

    store.mark_target_failed(tid, "ann-a", "provider timeout")
    assert store.get_target_state(tid, "ann-a") == "failed"

    row = store._conn.execute(
        "SELECT finished_at, error_msg FROM annotation_targets "
        "WHERE target_id=? AND annotator_id='ann-a'",
        (tid,),
    ).fetchone()
    assert row[0] is not None  # finished_at set
    assert row[1] == "provider timeout"


def test_mark_done_and_failed_independence(
    store: AnnotationStore, petdata_db: str
) -> None:
    """Marking one annotator's target done must not affect another annotator's row."""
    store.ingest_pending_from_petdata(
        petdata_db, ["ann-a", "ann-b"], annotator_type="llm"
    )
    claimed_a = store.claim_pending_targets("ann-a", batch_size=5)
    store.claim_pending_targets("ann-b", batch_size=5)

    # Mark ann-a's first target done
    store.mark_target_done(claimed_a[0], "ann-a")

    # ann-b's same target_id should still be in_progress
    assert store.get_target_state(claimed_a[0], "ann-b") == "in_progress"


# ---------------------------------------------------------------------------
# get_target_state
# ---------------------------------------------------------------------------


def test_get_target_state_returns_none_for_missing(store: AnnotationStore) -> None:
    """get_target_state returns None for a (target_id, annotator_id) that doesn't exist."""
    result = store.get_target_state("nonexistent", "ann-a")
    assert result is None
