"""Sampling strategy for human review routing."""

from __future__ import annotations

import random


def decide_review(
    schema_valid: bool,
    confidence: float | None,
    sampling_rate: float,
    threshold: float,
) -> str:
    """Decide whether a frame needs human review.

    Args:
        schema_valid: Whether the annotation passed schema validation.
        confidence: confidence_overall value (0-1), or None.
        sampling_rate: Random sampling rate (0-1).
        threshold: Minimum confidence to auto-approve.

    Returns:
        'approved' or 'needs_review'.
    """
    if not schema_valid:
        return "needs_review"
    if confidence is not None and confidence < threshold:
        return "needs_review"
    if random.random() < sampling_rate:
        return "needs_review"
    return "approved"
