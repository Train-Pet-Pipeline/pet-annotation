"""Tests for AnnotationStore — 4 annotator-paradigm tables."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from pet_schema import ClassifierAnnotation, HumanAnnotation, LLMAnnotation, RuleAnnotation

from pet_annotation.store import AnnotationStore


@pytest.fixture
def store(tmp_path: Path) -> AnnotationStore:
    """Fresh AnnotationStore with schema initialised."""
    db = tmp_path / "ann.db"
    s = AnnotationStore(str(db))
    s.init_schema()
    return s


def _base(**overrides):
    """Base annotation kwargs."""
    data = dict(
        target_id="t1",
        annotator_id="ann-1",
        modality="vision",
        schema_version="2.1.0",
        created_at=datetime(2026, 4, 21),
        storage_uri=None,
    )
    data.update(overrides)
    return data


# ---- LLM table ----


def test_llm_insert_roundtrip(store: AnnotationStore) -> None:
    """LLMAnnotation insert and fetch roundtrip preserves all fields."""
    ann = LLMAnnotation(
        annotation_id="l1",
        **_base(),
        prompt_hash="ph",
        raw_response="r",
        parsed_output={"ev": "eat"},
    )
    store.insert_llm(ann)
    rows = store.fetch_llm_by_target("t1")
    assert len(rows) == 1
    assert rows[0].annotation_id == "l1"
    assert rows[0].parsed_output == {"ev": "eat"}
    assert rows[0].prompt_hash == "ph"


def test_llm_fetch_by_target_filters(store: AnnotationStore) -> None:
    """fetch_llm_by_target returns only rows for the specified target_id."""
    ann_a = LLMAnnotation(
        annotation_id="la", **_base(target_id="a"), prompt_hash="h1", raw_response="r1",
        parsed_output={},
    )
    ann_b = LLMAnnotation(
        annotation_id="lb", **_base(target_id="b"), prompt_hash="h2", raw_response="r2",
        parsed_output={},
    )
    store.insert_llm(ann_a)
    store.insert_llm(ann_b)
    assert len(store.fetch_llm_by_target("a")) == 1
    assert len(store.fetch_llm_by_target("b")) == 1
    assert store.fetch_llm_by_target("a")[0].annotation_id == "la"


def test_llm_unique_target_annotator_prompt_hash(store: AnnotationStore) -> None:
    """UNIQUE(target_id, annotator_id, prompt_hash) rejects duplicate."""
    ann = LLMAnnotation(
        annotation_id="l1", **_base(), prompt_hash="h", raw_response="r", parsed_output={}
    )
    store.insert_llm(ann)
    with pytest.raises(sqlite3.IntegrityError):
        dup = LLMAnnotation(
            annotation_id="l2", **_base(), prompt_hash="h", raw_response="r2", parsed_output={}
        )
        store.insert_llm(dup)


# ---- Classifier table ----


def test_classifier_insert_roundtrip(store: AnnotationStore) -> None:
    """ClassifierAnnotation insert and fetch roundtrip preserves class_probs."""
    ann = ClassifierAnnotation(
        annotation_id="c1",
        **_base(modality="audio"),
        predicted_class="bark",
        class_probs={"bark": 0.9, "meow": 0.1},
        logits=[1.2, -0.3],
    )
    store.insert_classifier(ann)
    rows = store.fetch_classifier_by_target("t1")
    assert len(rows) == 1
    assert rows[0].predicted_class == "bark"
    assert abs(rows[0].class_probs["bark"] - 0.9) < 1e-6
    assert rows[0].logits is not None


def test_classifier_null_logits_roundtrip(store: AnnotationStore) -> None:
    """ClassifierAnnotation with logits=None roundtrips correctly."""
    ann = ClassifierAnnotation(
        annotation_id="c2", **_base(), predicted_class="cat",
        class_probs={"cat": 1.0}, logits=None,
    )
    store.insert_classifier(ann)
    rows = store.fetch_classifier_by_target("t1")
    assert rows[0].logits is None


def test_classifier_unique_target_annotator(store: AnnotationStore) -> None:
    """UNIQUE(target_id, annotator_id) rejects duplicate classifier annotation."""
    ann = ClassifierAnnotation(
        annotation_id="c1", **_base(), predicted_class="cat",
        class_probs={"cat": 1.0}, logits=None,
    )
    store.insert_classifier(ann)
    with pytest.raises(sqlite3.IntegrityError):
        dup = ClassifierAnnotation(
            annotation_id="c2", **_base(), predicted_class="dog",
            class_probs={"dog": 1.0}, logits=None,
        )
        store.insert_classifier(dup)


# ---- Rule table ----


def test_rule_insert_roundtrip(store: AnnotationStore) -> None:
    """RuleAnnotation insert and fetch roundtrip preserves rule_output."""
    ann = RuleAnnotation(
        annotation_id="r1", **_base(),
        rule_id="rule_confidence", rule_output={"passed": True, "score": 0.8},
    )
    store.insert_rule(ann)
    rows = store.fetch_rule_by_target("t1")
    assert len(rows) == 1
    assert rows[0].rule_output == {"passed": True, "score": 0.8}


def test_rule_unique_target_annotator_rule_id(store: AnnotationStore) -> None:
    """UNIQUE(target_id, annotator_id, rule_id) rejects duplicate."""
    ann = RuleAnnotation(annotation_id="r1", **_base(), rule_id="r1", rule_output={})
    store.insert_rule(ann)
    with pytest.raises(sqlite3.IntegrityError):
        dup = RuleAnnotation(annotation_id="r2", **_base(), rule_id="r1", rule_output={})
        store.insert_rule(dup)


# ---- Human table ----


def test_human_insert_roundtrip(store: AnnotationStore) -> None:
    """HumanAnnotation insert and fetch roundtrip preserves decision and notes."""
    ann = HumanAnnotation(
        annotation_id="h1",
        **_base(annotator_id="alice"),
        reviewer="alice",
        decision="accept",
        notes="looks good",
    )
    store.insert_human(ann)
    rows = store.fetch_human_by_target("t1")
    assert len(rows) == 1
    assert rows[0].reviewer == "alice"
    assert rows[0].decision == "accept"
    assert rows[0].notes == "looks good"


def test_human_null_notes_roundtrip(store: AnnotationStore) -> None:
    """HumanAnnotation with notes=None roundtrips correctly."""
    ann = HumanAnnotation(
        annotation_id="h2",
        **_base(annotator_id="bob"),
        reviewer="bob",
        decision="reject",
        notes=None,
    )
    store.insert_human(ann)
    rows = store.fetch_human_by_target("t1")
    assert rows[0].notes is None


def test_human_unique_target_annotator(store: AnnotationStore) -> None:
    """UNIQUE(target_id, annotator_id) rejects duplicate human annotation."""
    ann = HumanAnnotation(
        annotation_id="h1", **_base(annotator_id="alice"),
        reviewer="alice", decision="accept", notes=None,
    )
    store.insert_human(ann)
    with pytest.raises(sqlite3.IntegrityError):
        dup = HumanAnnotation(
            annotation_id="h2", **_base(annotator_id="alice"),
            reviewer="alice", decision="reject", notes=None,
        )
        store.insert_human(dup)


# ---- CHECK constraints ----


def test_modality_check_rejects_invalid_on_llm_table(store: AnnotationStore) -> None:
    """CHECK(modality IN ...) rejects bad value at SQL layer for llm_annotations."""
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO llm_annotations"
            "(annotation_id, target_id, annotator_id, annotator_type, "
            "modality, schema_version, created_at, prompt_hash, raw_response, parsed_output) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("x", "t", "a", "llm", "infrared", "2.1.0", "now", "h", "r", "{}"),
        )
        store._conn.commit()


def test_annotator_type_check_on_llm_table(store: AnnotationStore) -> None:
    """CHECK(annotator_type = 'llm') rejects any other value."""
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO llm_annotations"
            "(annotation_id, target_id, annotator_id, annotator_type, "
            "modality, schema_version, created_at, prompt_hash, raw_response, parsed_output) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("x", "t", "a", "vlm", "vision", "2.1.0", "now", "h", "r", "{}"),
        )
        store._conn.commit()


# ---- Migration idempotency ----


def test_init_schema_idempotent(tmp_path: Path) -> None:
    """Calling init_schema() twice on the same DB must not raise."""
    db = tmp_path / "idem.db"
    s1 = AnnotationStore(str(db))
    s1.init_schema()

    s2 = AnnotationStore(str(db))
    s2.init_schema()  # must not fail

    tables = {
        r[0]
        for r in s2._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "llm_annotations" in tables
    assert "classifier_annotations" in tables
    assert "rule_annotations" in tables
    assert "human_annotations" in tables
