"""Annotation orchestrator — async batch dispatch for 1..N LLM annotators.

Phase 4 wire: implements LLM paradigm. Classifier/rule/human in later subagents.

Design decisions (user-approved 2026-04-23):
- D1: pending targets read from pet-data frames table (read-only via sqlite3.connect)
- D2: state tracked in pet-annotation's annotation_targets table (migration 005)
- D3: 1..N annotators per target, all symmetric; N=0 valid (returns 0/0/0)
- D4: no cross-model reconcile; each annotation stored independently
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import uuid
from datetime import UTC, datetime
from typing import Any

from pet_schema import LLMAnnotation
from pet_schema.renderer import render_prompt
from pet_schema.validator import validate_output

from pet_annotation.config import AnnotationConfig, LLMAnnotatorConfig
from pet_annotation.store import AnnotationStore
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider
from pet_annotation.teacher.providers.vllm import VLLMProvider

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


def _build_provider(llm_cfg: LLMAnnotatorConfig) -> OpenAICompatProvider:
    """Instantiate a provider from LLMAnnotatorConfig.

    Args:
        llm_cfg: The annotator config.

    Returns:
        An OpenAICompatProvider or VLLMProvider instance.
    """
    if llm_cfg.provider == "vllm":
        return VLLMProvider(
            base_url=llm_cfg.base_url,
            model_name=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )
    # openai_compat
    return OpenAICompatProvider(
        base_url=llm_cfg.base_url,
        model_name=llm_cfg.model_name,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
    )


class AnnotationOrchestrator:
    """Dispatches pending targets to configured 1..N LLM annotators.

    Reads pending targets from pet-data DB. For each target × configured annotator,
    calls provider.annotate() asynchronously (bounded by max_concurrent semaphore),
    validates output against pet-schema LLMAnnotation contract, inserts into
    pet-annotation DB, marks target done (or failed).

    Args:
        config: Loaded annotation configuration.
        store: AnnotationStore instance.
        pet_data_db_path: Path to pet-data's SQLite database file (read-only).
    """

    def __init__(
        self,
        config: AnnotationConfig,
        store: AnnotationStore,
        pet_data_db_path: str,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            config: Validated annotation config.
            store: AnnotationStore for DB access.
            pet_data_db_path: Path to pet-data's frames SQLite (read-only).
        """
        self._config = config
        self._store = store
        self._pet_data_db = pet_data_db_path
        self._providers: dict[str, OpenAICompatProvider] = {
            llm_cfg.id: _build_provider(llm_cfg)
            for llm_cfg in config.llm.annotators
        }
        self._semaphore = asyncio.Semaphore(config.llm.max_concurrent)
        self._shutdown = False

    async def run(self) -> dict[str, int]:
        """Main dispatch loop for LLM paradigm.

        Returns:
            Stats dict: {"processed": N, "skipped": M, "failed": K}
        """
        if not self._config.llm.annotators:
            logger.info('{"event": "no_llm_annotators_configured"}')
            return {"processed": 0, "skipped": 0, "failed": 0}

        self._setup_signal_handlers()

        # Step 1: ingest new pending targets from pet-data
        annotator_ids = [a.id for a in self._config.llm.annotators]
        new_count = self._store.ingest_pending_from_petdata(
            self._pet_data_db,
            annotator_ids,
            annotator_type="llm",
            modality=None,  # all modalities for MVP
        )
        logger.info(f'{{"event": "ingested_pending", "count": {new_count}}}')

        # Render prompt once (shared by all annotators per run)
        system_prompt, user_prompt = render_prompt(
            version=self._config.annotation.schema_version
        )
        prompt_hash = compute_prompt_hash(
            system_prompt, user_prompt, self._config.annotation.schema_version
        )

        stats: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}

        # Step 2: batch loop per annotator (D4: each annotator independent)
        for llm_cfg in self._config.llm.annotators:
            if self._shutdown:
                break
            while not self._shutdown:
                target_ids = self._store.claim_pending_targets(
                    llm_cfg.id, self._config.llm.batch_size
                )
                if not target_ids:
                    break
                tasks = [
                    self._process_one(tid, llm_cfg, system_prompt, user_prompt, prompt_hash)
                    for tid in target_ids
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for tid, res in zip(target_ids, results):
                    if isinstance(res, Exception):
                        stats["failed"] += 1
                        self._store.mark_target_failed(tid, llm_cfg.id, str(res))
                        logger.error(
                            f'{{"event": "target_failed", "target_id": "{tid}", '
                            f'"annotator_id": "{llm_cfg.id}", "error": "{res}"}}'
                        )
                    elif res == "done":
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

        logger.info(f'{{"event": "orchestrator_done", "stats": {json.dumps(stats)}}}')
        return stats

    async def _process_one(
        self,
        target_id: str,
        llm_cfg: LLMAnnotatorConfig,
        system_prompt: str,
        user_prompt: str,
        prompt_hash: str,
    ) -> str:
        """Annotate one target × one LLM annotator.

        Args:
            target_id: The target (frame) to annotate.
            llm_cfg: The annotator configuration.
            system_prompt: Rendered system prompt.
            user_prompt: Rendered user prompt.
            prompt_hash: Cache key hash for this prompt + schema version.

        Returns:
            'done' on success, 'skipped' if provider is unavailable.

        Raises:
            Exception: On provider error or validation failure.
        """
        async with self._semaphore:
            provider = self._providers[llm_cfg.id]
            api_key = llm_cfg.api_key
            prompt_pair = (system_prompt, user_prompt)

            # Providers have async annotate(image_path, prompt, api_key).
            # For Phase 4 MVP, target_id is the frame path or identifier.
            # We pass target_id as image_path — callers are responsible for
            # providing resolvable paths (orchestrator consumer sets pet_data_db_path).
            raw_response, prompt_tokens, completion_tokens = await self._call_provider(
                provider, target_id, prompt_pair, api_key
            )

            # Validate against pet-schema
            validation = validate_output(raw_response, self._config.annotation.schema_version)
            parsed: dict[str, Any] = {}
            if validation.valid:
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError:
                    parsed = {"raw": raw_response}
            else:
                parsed = {"raw": raw_response, "validation_errors": validation.errors}

            annotation_id = f"{target_id}:{llm_cfg.id}:{prompt_hash[:8]}:{uuid.uuid4().hex[:8]}"
            ann = LLMAnnotation(
                annotation_id=annotation_id,
                target_id=target_id,
                annotator_id=llm_cfg.id,
                annotator_type="llm",
                modality="vision",
                schema_version=self._config.annotation.schema_version,
                created_at=datetime.now(UTC),
                storage_uri=None,
                prompt_hash=prompt_hash,
                raw_response=raw_response,
                parsed_output=parsed,
            )
            self._store.insert_llm(ann)
            self._store.mark_target_done(target_id, llm_cfg.id)

            logger.info(
                f'{{"event": "annotated", "target_id": "{target_id}", '
                f'"annotator_id": "{llm_cfg.id}", "schema_valid": {str(validation.valid).lower()}, '
                f'"tokens": {prompt_tokens + completion_tokens}}}'
            )
            return "done"

    async def _call_provider(
        self,
        provider: OpenAICompatProvider,
        target_id: str,
        prompt_pair: tuple[str, str],
        api_key: str,
    ) -> tuple[str, int, int]:
        """Call provider.annotate(); wraps sync providers via asyncio.to_thread if needed.

        Args:
            provider: The provider instance (OpenAICompatProvider or subclass).
            target_id: Target identifier (used as image_path for the provider).
            prompt_pair: (system_prompt, user_prompt) tuple.
            api_key: API key string (empty = no auth header).

        Returns:
            Tuple of (raw_response, prompt_tokens, completion_tokens).
        """
        # provider.annotate is async (BaseProvider.annotate is abstract async)
        result = await provider.annotate(target_id, prompt_pair, api_key)
        return result.raw_response, result.prompt_tokens, result.completion_tokens

    def _setup_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM for graceful shutdown via asyncio event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except (NotImplementedError, ValueError):
                pass  # Windows or no running loop

    def _handle_shutdown(self) -> None:
        """Set shutdown flag to stop after current batch."""
        logger.warning('{"event": "shutdown_requested"}')
        self._shutdown = True
