"""Tests for DPO pair validation — 5 rules from DEVELOPMENT_GUIDE."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from pet_annotation.dpo.validate_pairs import validate_pair


def _valid_output(**overrides) -> dict:
    """Build a valid annotation output dict for testing."""
    base = {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat", "breed_estimate": "british_shorthair",
            "id_tag": "grey_medium", "id_confidence": 0.85,
            "action": {
                "primary": "eating",
                "distribution": {"eating": 0.70, "drinking": 0.05, "sniffing_only": 0.10,
                                 "leaving_bowl": 0.05, "sitting_idle": 0.05, "other": 0.05},
            },
            "eating_metrics": {"speed": {"fast": 0.1, "normal": 0.7, "slow": 0.2},
                               "engagement": 0.8, "abandoned_midway": 0.1},
            "mood": {"alertness": 0.3, "anxiety": 0.1, "engagement": 0.8},
            "body_signals": {"posture": "relaxed", "ear_position": "forward"},
            "anomaly_signals": {"vomit_gesture": 0.0, "food_rejection": 0.0,
                                "excessive_sniffing": 0.0, "lethargy": 0.0, "aggression": 0.0},
        },
        "bowl": {"food_fill_ratio": 0.5, "water_fill_ratio": None, "food_type_visible": "dry"},
        "scene": {"lighting": "bright", "image_quality": "clear", "confidence_overall": 0.90},
        "narrative": "Grey cat eating normally.",
    }
    base.update(overrides)
    return base


def _mock_valid_result():
    """Return a mock validation result that passes."""
    r = MagicMock()
    r.valid = True
    r.errors = []
    return r


class TestValidatePair:
    @patch("pet_annotation.dpo.validate_pairs.validate_output", return_value=_mock_valid_result())
    def test_valid_pair(self, mock_validate):
        """Valid pair with different narratives and chosen > rejected confidence."""
        chosen = _valid_output(narrative="Cat eating dry food calmly.")
        rejected = _valid_output(narrative="Kitty is having a wonderful dinner!", **{
            "scene": {"lighting": "bright", "image_quality": "clear", "confidence_overall": 0.75}
        })
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "model_comparison"})
        assert ok
        assert len(errors) == 0

    @patch("pet_annotation.dpo.validate_pairs.validate_output", return_value=_mock_valid_result())
    def test_rule3_identical_narrative_fails(self, mock_validate):
        """Identical narratives fail validation."""
        chosen = _valid_output()
        rejected = _valid_output()  # same narrative
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "model_comparison"})
        assert not ok
        assert any("narrative" in e for e in errors)

    @patch("pet_annotation.dpo.validate_pairs.validate_output", return_value=_mock_valid_result())
    def test_rule4_user_feedback_needs_inference_id(self, mock_validate):
        """User feedback pairs require inference_id in metadata."""
        chosen = _valid_output(narrative="Corrected by user.")
        rejected = _valid_output(narrative="Model got it wrong.")
        rejected["scene"]["confidence_overall"] = 0.60
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "user_feedback"})
        assert not ok
        assert any("inference_id" in e for e in errors)

    @patch("pet_annotation.dpo.validate_pairs.validate_output", return_value=_mock_valid_result())
    def test_rule5_chosen_confidence_must_be_higher(self, mock_validate):
        """chosen confidence must be >= rejected confidence."""
        chosen = _valid_output(narrative="A")
        chosen["scene"]["confidence_overall"] = 0.50
        rejected = _valid_output(narrative="B")
        rejected["scene"]["confidence_overall"] = 0.90
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "model_comparison"})
        assert not ok
        assert any("confidence" in e for e in errors)
