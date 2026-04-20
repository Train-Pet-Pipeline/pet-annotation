"""Tests for CostTracker."""

from __future__ import annotations

from pet_annotation.teacher.cost_tracker import CostTracker


class TestCostTracker:
    def test_under_limit(self):
        tracker = CostTracker(max_daily_tokens=1000)
        assert tracker.check_and_record(500) is True
        assert tracker.remaining == 500

    def test_at_limit(self):
        tracker = CostTracker(max_daily_tokens=1000)
        tracker.check_and_record(999)
        assert tracker.check_and_record(1) is False

    def test_over_limit(self):
        tracker = CostTracker(max_daily_tokens=100)
        assert tracker.check_and_record(150) is False

    def test_per_model_tracking(self):
        tracker = CostTracker(max_daily_tokens=10000)
        tracker.check_and_record(100, model_name="qwen")
        tracker.check_and_record(200, model_name="doubao")
        stats = tracker.get_stats()
        assert stats["qwen"] == 100
        assert stats["doubao"] == 200
        assert stats["total"] == 300
