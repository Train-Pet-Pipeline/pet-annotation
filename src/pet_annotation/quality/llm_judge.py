"""LLM-based quality judge for cross-checking annotation outputs.

Stub for future implementation — will use a second LLM to evaluate
whether annotations are consistent and accurate.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_llm_judge(annotation_id: str, raw_response: str) -> dict:
    """Run LLM judge on a single annotation.

    Args:
        annotation_id: The annotation to evaluate.
        raw_response: The raw annotation output to judge.

    Returns:
        Dict with judge_score, judge_notes fields.
    """
    raise NotImplementedError("LLM judge not yet implemented")
