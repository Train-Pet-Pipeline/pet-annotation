"""Auto quality check — runs sampling decisions on auto_checked annotations.

Updates annotation review_status and frame annotation_status based on
the sampling decision.
"""
from __future__ import annotations

import logging

from pet_annotation.quality.sampling import decide_review
from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def run_auto_check(
    store: AnnotationStore,
    sampling_rate: float,
    threshold: float,
    primary_model: str,
) -> dict[str, int]:
    """Run auto check on all auto_checked frames.

    Queries annotations with review_status='pending' for the primary model,
    applies the sampling decision, and updates both annotation review_status
    and frame annotation_status.

    Args:
        store: AnnotationStore instance.
        sampling_rate: Random sampling rate for human review.
        threshold: Low confidence threshold for forced review.
        primary_model: Name of the primary model.

    Returns:
        Stats dict with counts of approved/needs_review.
    """
    rows = store.fetch_auto_checked_annotations(primary_model)

    stats = {"approved": 0, "needs_review": 0}

    for row in rows:
        decision = decide_review(
            schema_valid=bool(row["schema_valid"]),
            confidence=row["confidence_overall"],
            sampling_rate=sampling_rate,
            threshold=threshold,
        )

        new_frame_status = "approved" if decision == "approved" else "needs_review"
        store.update_review_and_frame_status(
            annotation_id=row["annotation_id"],
            review_status=decision,
            frame_id=row["frame_id"],
            frame_status=new_frame_status,
        )

        stats[decision] += 1
        logger.info(
            '{"event": "auto_check", "frame_id": "%s", "decision": "%s", "confidence": %s}',
            row["frame_id"], decision,
            row["confidence_overall"] if row["confidence_overall"] is not None else "null",
        )

    return stats
