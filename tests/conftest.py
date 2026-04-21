"""Shared test fixtures for pet-annotation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pet_schema import ClassifierAnnotation, HumanAnnotation, LLMAnnotation, RuleAnnotation


def _base_kw(**overrides):
    """Return base annotation field dict with optional overrides."""
    base = dict(
        annotation_id="a",
        target_id="t",
        annotator_id="x",
        modality="vision",
        schema_version="2.1.0",
        created_at=datetime(2026, 4, 21),
        storage_uri=None,
    )
    base.update(overrides)
    return base


@pytest.fixture
def store(tmp_path: Path):
    """Fresh AnnotationStore backed by a temp SQLite file, schema initialised."""
    from pet_annotation.store import AnnotationStore

    db = tmp_path / "ann.db"
    s = AnnotationStore(str(db))
    s.init_schema()
    return s


@pytest.fixture
def llm_annotation_factory():
    """Factory fixture for LLMAnnotation instances."""

    def make(**kw):
        """Build an LLMAnnotation with sensible defaults.

        Args:
            **kw: Field overrides including prompt_hash, raw_response, parsed_output.
        """
        prompt_hash = kw.pop("prompt_hash", "h")
        raw_response = kw.pop("raw_response", "r")
        parsed_output = kw.pop("parsed_output", {})
        return LLMAnnotation(
            **_base_kw(**kw),
            prompt_hash=prompt_hash,
            raw_response=raw_response,
            parsed_output=parsed_output,
        )

    return make


@pytest.fixture
def classifier_annotation_factory():
    """Factory fixture for ClassifierAnnotation instances."""

    def make(**kw):
        """Build a ClassifierAnnotation with sensible defaults.

        Args:
            **kw: Field overrides including predicted_class, class_probs, logits.
        """
        predicted_class = kw.pop("predicted_class", "x")
        class_probs = kw.pop("class_probs", {"x": 1.0})
        logits = kw.pop("logits", None)
        return ClassifierAnnotation(
            **_base_kw(**kw),
            predicted_class=predicted_class,
            class_probs=class_probs,
            logits=logits,
        )

    return make


@pytest.fixture
def rule_annotation_factory():
    """Factory fixture for RuleAnnotation instances."""

    def make(**kw):
        """Build a RuleAnnotation with sensible defaults.

        Args:
            **kw: Field overrides including rule_id, rule_output.
        """
        rule_id = kw.pop("rule_id", "r1")
        rule_output = kw.pop("rule_output", {})
        return RuleAnnotation(**_base_kw(**kw), rule_id=rule_id, rule_output=rule_output)

    return make


@pytest.fixture
def human_annotation_factory():
    """Factory fixture for HumanAnnotation instances."""

    def make(**kw):
        """Build a HumanAnnotation with sensible defaults.

        Args:
            **kw: Field overrides including reviewer, decision, notes.
        """
        reviewer = kw.pop("reviewer", "alice")
        decision = kw.pop("decision", "accept")
        notes = kw.pop("notes", None)
        return HumanAnnotation(
            **_base_kw(**kw), reviewer=reviewer, decision=decision, notes=notes
        )

    return make
