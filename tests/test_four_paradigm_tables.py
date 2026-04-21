"""4 annotator-paradigm 表 insert / select / CHECK 约束。

Spec: 2026-04-21-phase-2-debt-repayment-design.md §2 + §5
"""

import sqlite3
from datetime import datetime

import pytest

from pet_annotation.store import AnnotationStore
from pet_schema import ClassifierAnnotation, HumanAnnotation, LLMAnnotation, RuleAnnotation


@pytest.fixture
def store(tmp_path):
    """Fresh AnnotationStore backed by a temp SQLite file, schema initialised."""
    db = tmp_path / "ann.db"
    s = AnnotationStore(str(db))
    s.init_schema()  # 跑到 migration 004
    return s


def _base():
    return dict(
        target_id="t1",
        annotator_id="ann-1",
        modality="vision",
        schema_version="2.1.0",
        created_at=datetime(2026, 4, 21),
        storage_uri=None,
    )


def test_insert_llm_roundtrip(store):
    """LLMAnnotation insert + fetch roundtrip preserves parsed_output."""
    ann = LLMAnnotation(
        annotation_id="a1", **_base(), prompt_hash="h", raw_response="r", parsed_output={"ev": "eat"}
    )
    store.insert_llm(ann)
    rows = store.fetch_llm_by_target("t1")
    assert len(rows) == 1
    assert rows[0].parsed_output == {"ev": "eat"}


def test_insert_classifier_roundtrip(store):
    """ClassifierAnnotation insert + fetch roundtrip."""
    ann = ClassifierAnnotation(
        annotation_id="c1",
        **{**_base(), "modality": "audio"},
        predicted_class="bark",
        class_probs={"bark": 0.9, "meow": 0.1},
        logits=[1.2, -0.3],
    )
    store.insert_classifier(ann)
    rows = store.fetch_classifier_by_target("t1")
    assert len(rows) == 1


def test_insert_rule_roundtrip(store):
    """RuleAnnotation insert + fetch roundtrip."""
    ann = RuleAnnotation(annotation_id="r1", **_base(), rule_id="rule1", rule_output={"passed": True})
    store.insert_rule(ann)
    assert len(store.fetch_rule_by_target("t1")) == 1


def test_insert_human_roundtrip(store):
    """HumanAnnotation insert + fetch roundtrip."""
    ann = HumanAnnotation(
        annotation_id="h1",
        **{**_base(), "annotator_id": "alice"},
        reviewer="alice",
        decision="accept",
        notes=None,
    )
    store.insert_human(ann)
    assert len(store.fetch_human_by_target("t1")) == 1


def test_modality_check_rejects_invalid(store):
    """SQL CHECK constraint rejects invalid modality value at DB layer."""
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO llm_annotations(annotation_id, target_id, annotator_id, annotator_type, "
            "modality, schema_version, created_at, prompt_hash, raw_response, parsed_output) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("x", "t", "a", "llm", "infrared", "2.1.0", "now", "h", "r", "{}"),
        )
        store._conn.commit()


def test_unique_on_target_annotator_prompt_hash(store):
    """UNIQUE(target_id, annotator_id, prompt_hash) rejects duplicate LLM annotation."""
    ann = LLMAnnotation(
        annotation_id="a1", **_base(), prompt_hash="h", raw_response="r", parsed_output={}
    )
    store.insert_llm(ann)
    with pytest.raises(sqlite3.IntegrityError):
        dup = LLMAnnotation(
            annotation_id="a2", **_base(), prompt_hash="h", raw_response="r2", parsed_output={}
        )
        store.insert_llm(dup)
