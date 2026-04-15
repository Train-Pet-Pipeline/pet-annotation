"""Annotation orchestrator — the core batch annotation engine.

Pulls pending frames, dispatches to all configured models concurrently,
validates results, writes to DB, and advances the annotation state machine.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pet_schema.renderer import render_prompt
from pet_schema.validator import validate_output

from pet_annotation.config import AnnotationConfig
from pet_annotation.store import AnnotationRecord, AnnotationStore, ComparisonRecord
from pet_annotation.teacher.cost_tracker import CostTracker
from pet_annotation.teacher.provider import (
    BaseProvider,
    PromptPair,
    ProviderRegistry,
    ProviderResult,
)
from pet_annotation.teacher.rate_tracker import RateTracker

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
    """Orchestrates multi-model annotation with async concurrency.

    Pulls pending frames in batches, dispatches each frame to all configured
    models concurrently, validates results with pet-schema, writes to DB,
    and advances frames through the annotation state machine.

    Args:
        config: Loaded annotation configuration.
        store: AnnotationStore instance.
    """

    def __init__(self, config: AnnotationConfig, store: AnnotationStore) -> None:
        """Initialize the orchestrator.

        Args:
            config: Validated annotation config.
            store: AnnotationStore for DB access.
        """
        self._config = config
        self._store = store
        self._registry = ProviderRegistry(config)
        self._cost_tracker = CostTracker(config.annotation.max_daily_tokens)
        self._semaphore = asyncio.Semaphore(config.annotation.max_concurrent)
        self._db_executor = ThreadPoolExecutor(max_workers=1)
        self._shutdown = False

    async def run(self) -> dict:
        """Main entry point: process all pending frames in batches.

        Returns:
            Stats dict with counts of processed/skipped/failed frames.
        """
        self._setup_signal_handlers()
        prompt = render_prompt(version=self._config.annotation.schema_version)
        prompt_hash = compute_prompt_hash(
            prompt[0], prompt[1], self._config.annotation.schema_version
        )
        batch_size = self._config.annotation.batch_size

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        failed_ids: set[str] = set()

        while not self._shutdown:
            frames = await self._run_in_executor(
                self._store.fetch_pending_frames, batch_size
            )
            # Filter out frames that already failed in this run to avoid retry loops
            frames = [f for f in frames if f["frame_id"] not in failed_ids]
            if not frames:
                break

            frame_ids = [f["frame_id"] for f in frames]
            await self._run_in_executor(
                self._store.update_frame_status_batch, frame_ids, "annotating"
            )

            results = await asyncio.gather(
                *[self._process_frame(f, prompt, prompt_hash) for f in frames],
                return_exceptions=True,
            )

            for frame, result in zip(frames, results):
                fid = frame["frame_id"]
                if isinstance(result, Exception):
                    logger.error(
                        '{"event": "frame_failed", "frame_id": "%s", "error": "%s"}',
                        fid, str(result),
                    )
                    await self._run_in_executor(
                        self._store.update_frame_status_batch, [fid], "pending"
                    )
                    failed_ids.add(fid)
                    stats["failed"] += 1
                elif result == "skipped":
                    await self._run_in_executor(
                        self._store.update_frame_status_batch, [fid], "auto_checked"
                    )
                    stats["skipped"] += 1
                else:
                    await self._run_in_executor(
                        self._store.update_frame_status_batch, [fid], "auto_checked"
                    )
                    stats["processed"] += 1

            if not self._cost_tracker.check_and_record(0):
                logger.warning('{"event": "daily_limit_reached"}')
                break

        logger.info('{"event": "orchestrator_done", "stats": %s}', json.dumps(stats))
        return stats

    async def _process_frame(
        self, frame: sqlite3.Row, prompt: PromptPair, prompt_hash: str
    ) -> str:
        """Process a single frame across all configured models.

        Args:
            frame: Frame row from DB.
            prompt: (system, user) prompt pair.
            prompt_hash: Cache key hash.

        Returns:
            'processed' or 'skipped'.
        """
        image_path = str(Path(frame["data_root"]) / frame["frame_path"])
        primary_name = self._config.annotation.primary_model

        tasks = []
        for name, provider, tracker in self._registry.get_all():
            is_primary = name == primary_name
            tasks.append(
                self._annotate_one(
                    frame["frame_id"], name, provider, tracker,
                    image_path, prompt, prompt_hash, is_primary,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # If primary model failed, propagate the exception
        primary_idx = next(
            i for i, (n, _, _) in enumerate(self._registry.get_all()) if n == primary_name
        )
        primary_result = results[primary_idx]
        if isinstance(primary_result, BaseException):
            raise primary_result

        return primary_result  # type: ignore[return-value]

    async def _annotate_one(
        self,
        frame_id: str,
        model_name: str,
        provider: BaseProvider,
        tracker: RateTracker,
        image_path: str,
        prompt: PromptPair,
        prompt_hash: str,
        is_primary: bool,
    ) -> str:
        """Annotate a single frame with a single model.

        Args:
            frame_id: Frame identifier.
            model_name: Model name.
            provider: Provider instance.
            tracker: Rate tracker for this model.
            image_path: Full path to the image.
            prompt: Prompt pair.
            prompt_hash: Cache key hash.
            is_primary: Whether this is the primary model.

        Returns:
            'processed' or 'skipped'.
        """
        async with self._semaphore:
            # Cache check
            if is_primary:
                hit = await self._run_in_executor(
                    self._store.cache_hit, frame_id, model_name, prompt_hash
                )
            else:
                hit = await self._run_in_executor(
                    self._store.cache_hit_comparison, frame_id, model_name, prompt_hash
                )
            if hit:
                return "skipped"

            # Get API key
            key = await tracker.acquire(estimated_tokens=2000)

            # Call API
            result = await self._call_provider(provider, image_path, prompt, key)
            tracker.record(key, result.total_tokens)

            # Cost tracking
            self._cost_tracker.check_and_record(result.total_tokens, model_name)

            # Validate
            validation = validate_output(
                result.raw_response, self._config.annotation.schema_version
            )

            # Extract confidence_overall from parsed output
            confidence = None
            parsed = None
            if validation.valid:
                try:
                    data = json.loads(result.raw_response)
                    confidence = data.get("scene", {}).get("confidence_overall")
                    parsed = result.raw_response
                except json.JSONDecodeError:
                    pass

            errors_json = json.dumps(validation.errors) if validation.errors else None

            # Write to DB
            if is_primary:
                rec = AnnotationRecord(
                    annotation_id=str(uuid.uuid4()),
                    frame_id=frame_id,
                    model_name=model_name,
                    prompt_hash=prompt_hash,
                    raw_response=result.raw_response,
                    parsed_output=parsed,
                    schema_valid=validation.valid,
                    validation_errors=errors_json,
                    confidence_overall=confidence,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    api_latency_ms=result.latency_ms,
                )
                await self._run_in_executor(self._store.insert_annotation, rec)
            else:
                cmp_rec = ComparisonRecord(
                    comparison_id=str(uuid.uuid4()),
                    frame_id=frame_id,
                    model_name=model_name,
                    prompt_hash=prompt_hash,
                    raw_response=result.raw_response,
                    parsed_output=parsed,
                    schema_valid=validation.valid,
                    validation_errors=errors_json,
                    confidence_overall=confidence,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    api_latency_ms=result.latency_ms,
                )
                await self._run_in_executor(self._store.insert_comparison, cmp_rec)

            logger.info(
                '{"event": "annotated", "frame_id": "%s", "model": "%s", '
                '"valid": %s, "tokens": %d, "latency_ms": %d}',
                frame_id, model_name, str(validation.valid).lower(),
                result.total_tokens, result.latency_ms,
            )
            return "processed"

    async def _call_provider(
        self, provider: BaseProvider, image_path: str, prompt: PromptPair, api_key: str
    ) -> ProviderResult:
        """Call provider's annotate method. Separated for easy mocking in tests.

        Args:
            provider: The provider instance.
            image_path: Path to the image.
            prompt: Prompt pair.
            api_key: API key.

        Returns:
            ProviderResult.
        """
        return await provider.annotate(image_path, prompt, api_key)

    async def _run_in_executor(self, fn, *args):
        """Run a sync function in the DB thread pool.

        Args:
            fn: Synchronous function to call.
            *args: Arguments to pass.

        Returns:
            The function's return value.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._db_executor, fn, *args)

    def _setup_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                pass  # Windows

    def _handle_shutdown(self) -> None:
        """Set shutdown flag to stop after current batch."""
        logger.info('{"event": "shutdown_requested"}')
        self._shutdown = True
