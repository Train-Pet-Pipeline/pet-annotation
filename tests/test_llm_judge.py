"""Tests for quality.llm_judge — anomaly threshold parameterisation."""

from __future__ import annotations

from pet_annotation.quality.llm_judge import run_llm_judge


def _make_output(anomaly_val: float) -> dict:
    return {
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat",
            "action": {"primary": "eating"},
            "anomaly_signals": {"vomit_gesture": anomaly_val, "food_rejection": 0.0},
        },
    }


def test_anomaly_threshold_param_override():
    """anomaly_threshold kwarg controls when anomaly disagreements are flagged."""
    primary = _make_output(0.0)
    comparison = _make_output(0.25)  # abs diff = 0.25

    # With default threshold 0.3: abs(0.0-0.25)=0.25 < 0.3 → no disagreement
    result_default = run_llm_judge(primary, comparison)
    assert "anomaly_signals.vomit_gesture" not in result_default["disagreements"]

    # With tight threshold 0.2: abs(0.0-0.25)=0.25 >= 0.2 → disagreement flagged
    result_tight = run_llm_judge(primary, comparison, anomaly_threshold=0.2)
    assert "anomaly_signals.vomit_gesture" in result_tight["disagreements"]


def test_anomaly_threshold_from_params_yaml_default():
    """Default threshold 0.3 matches params.yaml quality.anomaly_threshold."""
    from pathlib import Path

    from pet_annotation.config import load_config

    cfg = load_config(Path(__file__).parent.parent / "params.yaml")
    assert cfg.quality.anomaly_threshold == 0.3
