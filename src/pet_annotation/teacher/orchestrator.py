"""Annotation orchestrator — stub for v2.0.0 4-paradigm schema.

The orchestrator is being rewritten for the 4-table annotator-paradigm model.
The old frame-based pipeline has been removed in v2.0.0 (destructive rebuild).
"""

from __future__ import annotations

import hashlib
import logging

from pet_annotation.config import AnnotationConfig
from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def compute_prompt_hash(system_prompt: str, user_prompt: str, schema_version: str) -> str:
    """Compute cache key hash from prompt template text + schema version.

    Args:
        system_prompt: System prompt template text.
        user_prompt: User prompt template text.
        schema_version: Schema version string.

    Returns:
        SHA-256 hex digest.
    """
    content = f"{system_prompt}|{user_prompt}|{schema_version}"
    return hashlib.sha256(content.encode()).hexdigest()


class AnnotationOrchestrator:
    """Orchestrator stub — v2.0.0 rewrite pending.

    The v1 frame-based pipeline has been removed.
    Implement per-paradigm dispatch in a follow-up PR.

    Args:
        config: Loaded annotation configuration.
        store: AnnotationStore instance.
    """

    def __init__(self, config: AnnotationConfig, store: AnnotationStore) -> None:
        """Initialise the orchestrator.

        Args:
            config: Validated annotation config.
            store: AnnotationStore for DB access.
        """
        self._config = config
        self._store = store

    async def run(self) -> dict:
        """Placeholder run — returns empty stats until 4-paradigm pipeline is wired.

        Returns:
            Stats dict.
        """
        logger.warning('{"event": "orchestrator_stub", "msg": "v2 pipeline not yet wired"}')
        return {"processed": 0, "skipped": 0, "failed": 0}
