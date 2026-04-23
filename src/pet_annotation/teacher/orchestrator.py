"""Annotation orchestrator — async batch dispatch for 1..N annotators (LLM/classifier/rule).

Phase 4 wire: LLM (Subagent B) + Classifier + Rule (Subagent C). Human in Subagent D.

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
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from pet_schema import ClassifierAnnotation, LLMAnnotation, RuleAnnotation
from pet_schema.renderer import render_prompt
from pet_schema.validator import validate_output

from pet_annotation.classifiers.base import BaseClassifierAnnotator
from pet_annotation.config import (
    AnnotationConfig,
    ClassifierAnnotatorConfig,
    LLMAnnotatorConfig,
    RuleAnnotatorConfig,
)
from pet_annotation.rules.base import BaseRuleAnnotator
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
        # Classifier/rule plugin dicts initialised empty; tests may inject via attribute.
        # Production callers should populate via _build_classifier_plugins() /
        # _build_rule_plugins() after construction (or override in subclass).
        self._classifier_plugins: dict[str, BaseClassifierAnnotator] = {}
        self._rule_plugins: dict[str, BaseRuleAnnotator] = {}
        self._semaphore = asyncio.Semaphore(config.llm.max_concurrent)
        # Lock serializes sqlite3 writes across asyncio tasks sharing one connection.
        # Python's sqlite3.Connection is not safe for interleaved execute/commit from
        # concurrent coroutines; without this lock, coroutine A's insert can be committed
        # inside coroutine B's implicit transaction, losing rollback boundaries.
        self._write_lock = asyncio.Lock()
        self._shutdown = False

    async def run(self, paradigms: list[str] | None = None) -> dict[str, int]:
        """Main dispatch loop for selected paradigms (LLM, classifier, rule).

        Dispatches each enabled paradigm in sequence: LLM → classifier → rule.
        Each paradigm returns its own stats; totals are accumulated.

        Args:
            paradigms: Optional list restricting which paradigms to dispatch
                (subset of {"llm","classifier","rule"}). None (default) = all
                configured paradigms. CLI passes [annotator] when --annotator
                is specified so a single-paradigm run doesn't trigger others.

        Returns:
            Stats dict: {"processed": N, "skipped": M, "failed": K}
        """
        self._setup_signal_handlers()

        stats: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}
        enabled = set(paradigms) if paradigms is not None else {"llm", "classifier", "rule"}

        if not (
            (self._config.llm.annotators and "llm" in enabled)
            or (self._config.classifier.annotators and "classifier" in enabled)
            or (self._config.rule.annotators and "rule" in enabled)
        ):
            logger.info('{"event": "no_annotators_configured"}')
            return stats

        # LLM paradigm
        if "llm" in enabled and self._config.llm.annotators:
            llm_stats = await self._run_llm_paradigm()
            for k in stats:
                stats[k] += llm_stats[k]

        # Classifier paradigm
        if (
            "classifier" in enabled
            and self._config.classifier.annotators
            and not self._shutdown
        ):
            cls_stats = await self._run_classifier_paradigm()
            for k in stats:
                stats[k] += cls_stats[k]

        # Rule paradigm
        if (
            "rule" in enabled
            and self._config.rule.annotators
            and not self._shutdown
        ):
            rule_stats = await self._run_rule_paradigm()
            for k in stats:
                stats[k] += rule_stats[k]

        logger.info(f'{{"event": "orchestrator_done", "stats": {json.dumps(stats)}}}')
        return stats

    async def _run_llm_paradigm(self) -> dict[str, int]:
        """Dispatch all configured LLM annotators.

        Returns:
            Stats dict: {"processed": N, "skipped": M, "failed": K}
        """
        # Ingest new pending targets from pet-data
        annotator_ids = [a.id for a in self._config.llm.annotators]
        new_count = self._store.ingest_pending_from_petdata(
            self._pet_data_db,
            annotator_ids,
            annotator_type="llm",
            modality=None,
        )
        logger.info(f'{{"event": "llm_ingested_pending", "count": {new_count}}}')

        # Render prompt once (shared by all annotators per run)
        system_prompt, user_prompt = render_prompt(
            version=self._config.annotation.schema_version
        )
        prompt_hash = compute_prompt_hash(
            system_prompt, user_prompt, self._config.annotation.schema_version
        )

        stats: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}

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
                        async with self._write_lock:
                            self._store.mark_target_failed(tid, llm_cfg.id, str(res))
                        logger.error(
                            f'{{"event": "target_failed", "target_id": "{tid}", '
                            f'"annotator_id": "{llm_cfg.id}", "error": "{res}"}}'
                        )
                    elif res == "done":
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

        return stats

    async def _run_classifier_paradigm(self) -> dict[str, int]:
        """Dispatch all configured classifier annotators.

        Uses asyncio.to_thread() for synchronous plugin inference to keep
        dispatch uniformly async (consistent with LLM paradigm framework).

        Returns:
            Stats dict: {"processed": N, "skipped": M, "failed": K}
        """
        annotator_ids = [a.id for a in self._config.classifier.annotators]
        new_count = self._store.ingest_pending_from_petdata(
            self._pet_data_db,
            annotator_ids,
            annotator_type="classifier",
            modality=None,
        )
        logger.info(f'{{"event": "classifier_ingested_pending", "count": {new_count}}}')

        semaphore = asyncio.Semaphore(self._config.classifier.max_concurrent)
        stats: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}

        for cls_cfg in self._config.classifier.annotators:
            if self._shutdown:
                break
            plugin = self._classifier_plugins.get(cls_cfg.id)
            if plugin is None:
                logger.warning(
                    f'{{"event": "classifier_plugin_missing", "annotator_id": "{cls_cfg.id}"}}'
                )
                continue

            while not self._shutdown:
                target_ids = self._store.claim_pending_targets(
                    cls_cfg.id, self._config.classifier.batch_size
                )
                if not target_ids:
                    break

                tasks = [
                    self._process_one_classifier(tid, cls_cfg, plugin, semaphore)
                    for tid in target_ids
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for tid, res in zip(target_ids, results):
                    if isinstance(res, Exception):
                        stats["failed"] += 1
                        async with self._write_lock:
                            self._store.mark_target_failed(tid, cls_cfg.id, str(res))
                        logger.error(
                            f'{{"event": "classifier_target_failed", "target_id": "{tid}", '
                            f'"annotator_id": "{cls_cfg.id}", "error": "{res}"}}'
                        )
                    elif res == "done":
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

        return stats

    async def _run_rule_paradigm(self) -> dict[str, int]:
        """Dispatch all configured rule annotators.

        Fetches metadata from petdata frames table (read-only) for each target, then
        applies rule via asyncio.to_thread() for uniform async dispatch.

        Returns:
            Stats dict: {"processed": N, "skipped": M, "failed": K}
        """
        annotator_ids = [a.id for a in self._config.rule.annotators]
        new_count = self._store.ingest_pending_from_petdata(
            self._pet_data_db,
            annotator_ids,
            annotator_type="rule",
            modality=None,
        )
        logger.info(f'{{"event": "rule_ingested_pending", "count": {new_count}}}')

        semaphore = asyncio.Semaphore(self._config.rule.max_concurrent)
        stats: dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0}

        for rule_cfg in self._config.rule.annotators:
            if self._shutdown:
                break
            plugin = self._rule_plugins.get(rule_cfg.id)
            if plugin is None:
                logger.warning(
                    f'{{"event": "rule_plugin_missing", "annotator_id": "{rule_cfg.id}"}}'
                )
                continue

            while not self._shutdown:
                target_ids = self._store.claim_pending_targets(
                    rule_cfg.id, self._config.rule.batch_size
                )
                if not target_ids:
                    break

                tasks = [
                    self._process_one_rule(tid, rule_cfg, plugin, semaphore)
                    for tid in target_ids
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for tid, res in zip(target_ids, results):
                    if isinstance(res, Exception):
                        stats["failed"] += 1
                        async with self._write_lock:
                            self._store.mark_target_failed(tid, rule_cfg.id, str(res))
                        logger.error(
                            f'{{"event": "rule_target_failed", "target_id": "{tid}", '
                            f'"annotator_id": "{rule_cfg.id}", "error": "{res}"}}'
                        )
                    elif res == "done":
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

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
                modality=self._config.annotation.modality_default,
                schema_version=self._config.annotation.schema_version,
                created_at=datetime.now(UTC),
                storage_uri=None,
                prompt_hash=prompt_hash,
                raw_response=raw_response,
                parsed_output=parsed,
            )
            # Serialize sqlite writes across concurrent tasks — see __init__ comment.
            async with self._write_lock:
                self._store.insert_llm(ann)
                self._store.mark_target_done(target_id, llm_cfg.id)

            logger.info(
                f'{{"event": "annotated", "target_id": "{target_id}", '
                f'"annotator_id": "{llm_cfg.id}", "schema_valid": {str(validation.valid).lower()}, '
                f'"tokens": {prompt_tokens + completion_tokens}}}'
            )
            return "done"

    async def _process_one_classifier(
        self,
        target_id: str,
        cls_cfg: ClassifierAnnotatorConfig,
        plugin: BaseClassifierAnnotator,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Annotate one target × one classifier annotator.

        Runs synchronous plugin.annotate() via asyncio.to_thread() for uniform
        async framework. Serializes store writes with write_lock.

        Args:
            target_id: The target (frame) identifier. Used as target_data path.
            cls_cfg: The classifier annotator configuration.
            plugin: Loaded classifier plugin instance.
            semaphore: Concurrency limiter.

        Returns:
            'done' on success.

        Raises:
            Exception: On inference error or store write failure.
        """
        async with semaphore:
            predicted_class, class_probs, logits = await asyncio.to_thread(
                plugin.annotate,
                target_id,
                **cls_cfg.extra_params,
            )

            annotation_id = f"{target_id}:{cls_cfg.id}:{uuid.uuid4().hex[:8]}"
            ann = ClassifierAnnotation(
                annotation_id=annotation_id,
                target_id=target_id,
                annotator_id=cls_cfg.id,
                annotator_type="classifier",
                modality=self._config.annotation.modality_default,
                schema_version=self._config.annotation.schema_version,
                created_at=datetime.now(UTC),
                storage_uri=None,
                predicted_class=predicted_class,
                class_probs=class_probs,
                logits=logits,
            )
            async with self._write_lock:
                self._store.insert_classifier(ann)
                self._store.mark_target_done(target_id, cls_cfg.id)

            logger.info(
                f'{{"event": "classifier_annotated", "target_id": "{target_id}", '
                f'"annotator_id": "{cls_cfg.id}", "predicted_class": "{predicted_class}"}}'
            )
            return "done"

    async def _process_one_rule(
        self,
        target_id: str,
        rule_cfg: RuleAnnotatorConfig,
        plugin: BaseRuleAnnotator,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Annotate one target × one rule annotator.

        Fetches frame metadata from petdata (read-only), then runs synchronous
        plugin.apply() via asyncio.to_thread(). Serializes store writes with write_lock.

        Args:
            target_id: The target (frame) identifier.
            rule_cfg: The rule annotator configuration.
            plugin: Loaded rule plugin instance.
            semaphore: Concurrency limiter.

        Returns:
            'done' on success.

        Raises:
            Exception: On metadata fetch error or store write failure.
        """
        async with semaphore:
            target_metadata = await asyncio.to_thread(
                self._fetch_frame_metadata, target_id
            )
            rule_output = await asyncio.to_thread(
                plugin.apply,
                target_metadata,
                **rule_cfg.extra_params,
            )

            annotation_id = f"{target_id}:{rule_cfg.id}:{uuid.uuid4().hex[:8]}"
            ann = RuleAnnotation(
                annotation_id=annotation_id,
                target_id=target_id,
                annotator_id=rule_cfg.id,
                annotator_type="rule",
                modality=self._config.annotation.modality_default,
                schema_version=self._config.annotation.schema_version,
                created_at=datetime.now(UTC),
                storage_uri=None,
                rule_id=rule_cfg.rule_id,
                rule_output=rule_output,
            )
            async with self._write_lock:
                self._store.insert_rule(ann)
                self._store.mark_target_done(target_id, rule_cfg.id)

            logger.info(
                f'{{"event": "rule_annotated", "target_id": "{target_id}", '
                f'"annotator_id": "{rule_cfg.id}", "rule_id": "{rule_cfg.rule_id}", '
                f'"rule_output_empty": {str(not rule_output).lower()}}}'
            )
            return "done"

    def _fetch_frame_metadata(self, frame_id: str) -> dict[str, Any]:
        """Fetch all columns for a frame from pet-data (read-only).

        Opens a short-lived read-only connection to petdata SQLite, returns
        all columns as a dict for rule plugin consumption.

        Args:
            frame_id: The frame identifier to look up.

        Returns:
            Dict of column_name → value from the frames table row.
            Returns empty dict if frame_id is not found.
        """
        conn = sqlite3.connect(
            f"file:{self._pet_data_db}?mode=ro", uri=True, timeout=10
        )
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM frames WHERE frame_id = ?", (frame_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()

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
