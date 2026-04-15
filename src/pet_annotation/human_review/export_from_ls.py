"""Export completed Label Studio annotations back to the database.

Updates review_status and optionally overwrites parsed_output if the
reviewer made corrections.
"""
from __future__ import annotations

import logging

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def export_reviewed(store: AnnotationStore, ls_url: str, ls_api_key: str) -> int:
    """Pull completed annotations from Label Studio and update DB.

    Args:
        store: AnnotationStore instance.
        ls_url: Label Studio server URL.
        ls_api_key: Label Studio API key.

    Returns:
        Number of annotations updated.
    """
    raise NotImplementedError("Requires Label Studio integration — implement when LS is deployed")
