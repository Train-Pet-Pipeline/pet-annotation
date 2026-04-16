"""Import user feedback from APP into Label Studio for DPO pair generation.

Post-launch feature: pulls feeding_events with user_feedback='inaccurate'
from cloud sync, creates Label Studio tasks for human confirmation. Requires
the cloud sync backend and APP feedback API to be deployed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def import_user_corrections(ls_url: str, ls_api_key: str) -> int:
    """Pull user corrections and create Label Studio tasks.

    Args:
        ls_url: Label Studio server URL.
        ls_api_key: Label Studio API key.

    Returns:
        Number of correction tasks created.

    Raises:
        NotImplementedError: App feedback pipeline is a post-launch feature
            requiring cloud sync infrastructure.
    """
    raise NotImplementedError(
        "App feedback import is a post-launch feature. "
        "Requires: (1) cloud sync backend for feeding events, "
        "(2) APP feedback API integration, "
        "(3) Label Studio project for feedback review. "
        "See: https://github.com/Train-Pet-Pipeline/pet-annotation/issues "
        "for tracking."
    )
