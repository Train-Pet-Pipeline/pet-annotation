"""Tests for rule annotator plugin interface (TDD — Commit 3)."""

from __future__ import annotations

import pytest

from pet_annotation.rules.base import BaseRuleAnnotator, BrightnessRule


# ---------------------------------------------------------------------------
# BrightnessRule tests
# ---------------------------------------------------------------------------


def test_brightness_rule_dim_scene() -> None:
    """BrightnessRule labels dim_scene when brightness_score < threshold."""
    rule = BrightnessRule(threshold=0.3)
    result = rule.apply({"brightness_score": 0.1})
    assert result["label"] == "dim_scene"
    assert result["threshold"] == 0.3
    assert result["score"] == pytest.approx(0.1)


def test_brightness_rule_normal_scene() -> None:
    """BrightnessRule labels normal_scene when brightness_score >= threshold."""
    rule = BrightnessRule(threshold=0.3)
    result = rule.apply({"brightness_score": 0.5})
    assert result["label"] == "normal_scene"


def test_brightness_rule_at_boundary() -> None:
    """BrightnessRule labels normal_scene at exact threshold (not < threshold)."""
    rule = BrightnessRule(threshold=0.3)
    result = rule.apply({"brightness_score": 0.3})
    assert result["label"] == "normal_scene"


def test_brightness_rule_missing_field_returns_empty() -> None:
    """BrightnessRule returns empty dict when brightness_score is absent."""
    rule = BrightnessRule(threshold=0.3)
    result = rule.apply({"other_field": 0.5})
    assert result == {}


def test_brightness_rule_default_threshold() -> None:
    """BrightnessRule default threshold is 0.3."""
    rule = BrightnessRule()
    assert rule.threshold == 0.3


def test_brightness_rule_has_rule_id() -> None:
    """BrightnessRule.rule_id is 'brightness_threshold'."""
    assert BrightnessRule.rule_id == "brightness_threshold"


def test_brightness_rule_accepts_extra_metadata() -> None:
    """BrightnessRule.apply() ignores unrelated metadata fields."""
    rule = BrightnessRule(threshold=0.3)
    result = rule.apply({"brightness_score": 0.2, "frame_id": "f1", "modality": "vision"})
    assert result["label"] == "dim_scene"


def test_brightness_rule_result_is_json_serializable() -> None:
    """BrightnessRule.apply() result can be JSON-serialised."""
    import json

    rule = BrightnessRule(threshold=0.3)
    result = rule.apply({"brightness_score": 0.1})
    # Must not raise
    json.dumps(result)


# ---------------------------------------------------------------------------
# BaseRuleAnnotator abstraction tests
# ---------------------------------------------------------------------------


def test_base_rule_is_abstract() -> None:
    """BaseRuleAnnotator cannot be instantiated directly (abstract)."""
    with pytest.raises(TypeError):
        BaseRuleAnnotator()  # type: ignore[abstract]


def test_concrete_rule_must_implement_apply() -> None:
    """A concrete subclass without apply() raises TypeError on instantiation."""

    class IncompleteRule(BaseRuleAnnotator):
        rule_id = "incomplete"
        # missing apply()

    with pytest.raises(TypeError):
        IncompleteRule()  # type: ignore[abstract]


def test_concrete_rule_with_apply_instantiates() -> None:
    """A concrete subclass that implements apply() can be instantiated."""

    class FixedRule(BaseRuleAnnotator):
        rule_id = "fixed_rule"

        def apply(self, target_metadata: dict) -> dict:
            """Return fixed output."""
            return {"label": "fixed"}

    rule = FixedRule()
    result = rule.apply({"anything": 1})
    assert result["label"] == "fixed"


def test_concrete_rule_empty_output_is_valid() -> None:
    """A rule may legitimately return {} (rule did not trigger)."""

    class NeverFiresRule(BaseRuleAnnotator):
        rule_id = "never_fires"

        def apply(self, target_metadata: dict) -> dict:
            """Return empty dict — rule did not trigger."""
            return {}

    rule = NeverFiresRule()
    result = rule.apply({"brightness_score": 0.9})
    assert result == {}
