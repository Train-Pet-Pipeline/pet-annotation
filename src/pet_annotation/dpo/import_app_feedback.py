"""Import user feedback from APP into Label Studio for DPO pair generation.

Pulls feeding_events with user_feedback='inaccurate' from cloud sync,
creates Label Studio tasks for human confirmation.
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
    """
    raise NotImplementedError("Requires cloud sync + Label Studio — implement post-launch")
