"""LLM-based quality judge for cross-checking annotation outputs.

Compares primary and comparison model outputs to detect inconsistencies.
When both models agree on key fields (pet_present, species, action.primary),
the annotation is considered consistent. Disagreements flag the annotation
for human review.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_llm_judge(
    primary_output: dict[str, Any],
    comparison_output: dict[str, Any] | None,
    anomaly_threshold: float = 0.3,
) -> dict[str, Any]:
    """Cross-check primary annotation against comparison model output.

    Compares key fields between the two model outputs and scores consistency.
    When no comparison output is available, returns a neutral score.

    Args:
        primary_output: Parsed JSON output from the primary model.
        comparison_output: Parsed JSON output from the comparison model,
            or None if unavailable.
        anomaly_threshold: Maximum allowed absolute difference for anomaly signals
            before flagging a disagreement. Loaded from params.yaml
            quality.anomaly_threshold (default 0.3).

    Returns:
        Dict with keys:
            judge_score (float): 0.0–1.0, higher = more consistent.
            judge_notes (str): Human-readable summary of findings.
            disagreements (list[str]): List of fields that disagree.
    """
    if comparison_output is None:
        return {
            "judge_score": 0.5,
            "judge_notes": "No comparison model output available",
            "disagreements": [],
        }

    disagreements: list[str] = []
    checks_total = 0
    checks_passed = 0

    # Check pet_present
    checks_total += 1
    if primary_output.get("pet_present") == comparison_output.get("pet_present"):
        checks_passed += 1
    else:
        disagreements.append("pet_present")

    # Check pet_count
    checks_total += 1
    if primary_output.get("pet_count") == comparison_output.get("pet_count"):
        checks_passed += 1
    else:
        disagreements.append("pet_count")

    # Check species (nested under pet)
    primary_pet = primary_output.get("pet") or {}
    comparison_pet = comparison_output.get("pet") or {}

    checks_total += 1
    if primary_pet.get("species") == comparison_pet.get("species"):
        checks_passed += 1
    else:
        disagreements.append("pet.species")

    # Check action.primary
    primary_action = (primary_pet.get("action") or {}).get("primary")
    comparison_action = (comparison_pet.get("action") or {}).get("primary")
    checks_total += 1
    if primary_action == comparison_action:
        checks_passed += 1
    else:
        disagreements.append("action.primary")

    # Check anomaly signals — high-confidence anomaly disagreement is critical
    primary_anomaly = primary_pet.get("anomaly_signals") or {}
    comparison_anomaly = comparison_pet.get("anomaly_signals") or {}
    for signal_name in ("vomit_gesture", "food_rejection"):
        checks_total += 1
        p_val = primary_anomaly.get(signal_name, 0.0) or 0.0
        c_val = comparison_anomaly.get(signal_name, 0.0) or 0.0
        if abs(p_val - c_val) < anomaly_threshold:
            checks_passed += 1
        else:
            disagreements.append(f"anomaly_signals.{signal_name}")

    score = checks_passed / checks_total if checks_total > 0 else 0.5

    notes_parts = []
    if disagreements:
        notes_parts.append(f"Disagreements on: {', '.join(disagreements)}")
    else:
        notes_parts.append("Models agree on all checked fields")
    notes_parts.append(f"({checks_passed}/{checks_total} checks passed)")

    return {
        "judge_score": round(score, 2),
        "judge_notes": " ".join(notes_parts),
        "disagreements": disagreements,
    }
