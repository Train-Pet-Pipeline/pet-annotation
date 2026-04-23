"""Tests for classifier annotator plugin interface (TDD — Commit 2)."""

from __future__ import annotations

import pytest

from pet_annotation.classifiers.base import BaseClassifierAnnotator, NoopClassifier

# ---------------------------------------------------------------------------
# NoopClassifier tests
# ---------------------------------------------------------------------------


def test_noop_classifier_returns_unknown_class() -> None:
    """NoopClassifier.annotate() returns 'unknown' predicted_class."""
    clf = NoopClassifier()
    predicted_class, class_probs, logits = clf.annotate(b"fake-data")
    assert predicted_class == "unknown"


def test_noop_classifier_returns_uniform_probs() -> None:
    """NoopClassifier.annotate() returns {'unknown': 1.0} class_probs."""
    clf = NoopClassifier()
    _, class_probs, _ = clf.annotate(b"fake-data")
    assert class_probs == {"unknown": 1.0}


def test_noop_classifier_returns_none_logits() -> None:
    """NoopClassifier.annotate() returns None for logits."""
    clf = NoopClassifier()
    _, _, logits = clf.annotate(b"fake-data")
    assert logits is None


def test_noop_classifier_accepts_str_path() -> None:
    """NoopClassifier.annotate() accepts a file path string as target_data."""
    clf = NoopClassifier()
    predicted_class, class_probs, logits = clf.annotate("/tmp/fake.wav")
    assert predicted_class == "unknown"
    assert class_probs == {"unknown": 1.0}
    assert logits is None


def test_noop_classifier_accepts_extra_kwargs() -> None:
    """NoopClassifier.annotate() ignores extra kwargs without error."""
    clf = NoopClassifier()
    result = clf.annotate(b"data", top_k=3, threshold=0.5)
    assert result[0] == "unknown"


def test_noop_classifier_has_plugin_name() -> None:
    """NoopClassifier.plugin_name is 'noop_classifier'."""
    assert NoopClassifier.plugin_name == "noop_classifier"


# ---------------------------------------------------------------------------
# BaseClassifierAnnotator abstraction tests
# ---------------------------------------------------------------------------


def test_base_classifier_is_abstract() -> None:
    """BaseClassifierAnnotator cannot be instantiated directly (abstract)."""
    with pytest.raises(TypeError):
        BaseClassifierAnnotator()  # type: ignore[abstract]


def test_concrete_classifier_must_implement_annotate() -> None:
    """A concrete subclass without annotate() raises TypeError on instantiation."""

    class IncompleteClassifier(BaseClassifierAnnotator):
        plugin_name = "incomplete"
        # missing annotate()

    with pytest.raises(TypeError):
        IncompleteClassifier()  # type: ignore[abstract]


def test_concrete_classifier_with_annotate_instantiates() -> None:
    """A concrete subclass that implements annotate() can be instantiated."""

    class FixedClassifier(BaseClassifierAnnotator):
        plugin_name = "fixed_classifier"

        def annotate(self, target_data, **kwargs):
            """Return fixed prediction."""
            return "feeding", {"feeding": 0.9, "idle": 0.1}, [0.9, 0.1]

    clf = FixedClassifier()
    predicted_class, class_probs, logits = clf.annotate(b"audio")
    assert predicted_class == "feeding"
    assert class_probs["feeding"] == 0.9
    assert logits == [0.9, 0.1]
