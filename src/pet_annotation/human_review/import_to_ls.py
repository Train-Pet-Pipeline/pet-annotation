"""Import VLM outputs to Label Studio as review tasks.

Pre-fills VLM output as predictions so reviewers see the model's answer
and can quickly confirm or correct.
"""
from __future__ import annotations

import logging

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def import_needs_review(store: AnnotationStore, ls_url: str, ls_api_key: str) -> int:
    """Create Label Studio tasks for annotations needing review.

    Queries annotations with review_status='needs_review', builds LS tasks
    with VLM output pre-filled as predictions, and creates them via LS API.

    Args:
        store: AnnotationStore instance.
        ls_url: Label Studio server URL.
        ls_api_key: Label Studio API key.

    Returns:
        Number of tasks created.
    """
    raise NotImplementedError("Requires Label Studio integration — implement when LS is deployed")
