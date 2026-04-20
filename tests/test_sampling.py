"""Tests for sampling module."""

from __future__ import annotations

from unittest.mock import patch

from pet_annotation.quality.sampling import decide_review


class TestDecideReview:
    def test_invalid_schema_forces_review(self):
        """Invalid schema always forces human review."""
        assert (
            decide_review(schema_valid=False, confidence=0.95, sampling_rate=0.0, threshold=0.70)
            == "needs_review"
        )

    def test_low_confidence_forces_review(self):
        """Low confidence forces human review."""
        assert (
            decide_review(schema_valid=True, confidence=0.50, sampling_rate=0.0, threshold=0.70)
            == "needs_review"
        )

    def test_random_sampling(self):
        """Random sampling triggers review when within sampling rate."""
        with patch("pet_annotation.quality.sampling.random.random", return_value=0.05):
            assert (
                decide_review(
                    schema_valid=True, confidence=0.90, sampling_rate=0.15, threshold=0.70
                )
                == "needs_review"
            )

    def test_approved_when_passing(self):
        """High confidence passing all checks gets approved."""
        with patch("pet_annotation.quality.sampling.random.random", return_value=0.99):
            assert (
                decide_review(
                    schema_valid=True, confidence=0.90, sampling_rate=0.15, threshold=0.70
                )
                == "approved"
            )
