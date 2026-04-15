"""Tests for AnnotationOrchestrator."""
from __future__ import annotations

import sqlite3
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from pet_annotation.config import (
    AccountConfig,
    AnnotationConfig,
    AnnotationParams,
    DatabaseConfig,
    DpoParams,
    ModelConfig,
)
from pet_annotation.store import AnnotationRecord, AnnotationStore
from pet_annotation.teacher.orchestrator import AnnotationOrchestrator
from pet_annotation.teacher.provider import ProviderResult


def _make_config(db_path: str = ":memory:") -> AnnotationConfig:
    """Build a minimal config for orchestrator tests."""
    return AnnotationConfig(
        database=DatabaseConfig(path=db_path, data_root="/data"),
        annotation=AnnotationParams(
            batch_size=2, max_concurrent=5, max_daily_tokens=100000,
            review_sampling_rate=0.0, low_confidence_threshold=0.70,
            primary_model="primary", schema_version="1.0",
        ),
        models={
            "primary": ModelConfig(
                provider="openai_compat", base_url="http://a/v1", model_name="a",
                accounts=[AccountConfig(key_env="K1", rpm=100, tpm=999999)],
            ),
        },
        dpo=DpoParams(min_pairs_per_release=100),
    )


def _insert_frames(conn: sqlite3.Connection, count: int) -> list[str]:
    """Insert test frames into the in-memory DB."""
    ids = []
    for i in range(count):
        fid = f"frame_{i:03d}"
        conn.execute(
            "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) "
            "VALUES (?, ?, ?, ?, ?)",
            (fid, "v1", "selfshot", f"frames/{fid}.jpg", "/data"),
        )
        ids.append(fid)
    conn.commit()
    return ids


MOCK_RAW_RESPONSE = (
    '{"schema_version":"1.0","pet_present":false,"pet_count":0,'
    '"pet":null,"bowl":{"food_fill_ratio":0.5,"water_fill_ratio":null,'
    '"food_type_visible":"dry"},"scene":{"lighting":"bright",'
    '"image_quality":"clear","confidence_overall":0.9},"narrative":"test"}'
)


class TestOrchestrator:
    async def test_processes_pending_frames(self, db_conn):
        """Orchestrator processes all pending frames and marks them auto_checked."""
        config = _make_config()
        store = AnnotationStore(conn=db_conn)
        _insert_frames(db_conn, 3)

        mock_result = ProviderResult(
            raw_response=MOCK_RAW_RESPONSE,
            prompt_tokens=100, completion_tokens=50, latency_ms=500,
        )

        # Mock validate_output to return valid
        mock_validation = MagicMock()
        mock_validation.valid = True
        mock_validation.errors = []

        orch = AnnotationOrchestrator(config=config, store=store)

        with patch.object(orch, "_call_provider", new_callable=AsyncMock, return_value=mock_result):
            with patch(
                "pet_annotation.teacher.orchestrator.render_prompt",
                return_value=("sys", "usr"),
            ):
                with patch(
                    "pet_annotation.teacher.orchestrator.validate_output",
                    return_value=mock_validation,
                ):
                    await orch.run()

        rows = db_conn.execute(
            "SELECT annotation_status FROM frames ORDER BY frame_id"
        ).fetchall()
        assert all(r[0] == "auto_checked" for r in rows)

    async def test_skips_cached_frames(self, db_conn):
        """Orchestrator skips frames that already have cached annotations."""
        config = _make_config()
        store = AnnotationStore(conn=db_conn)
        _insert_frames(db_conn, 1)

        # Pre-insert a cached annotation
        store.insert_annotation(AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            frame_id="frame_000",
            model_name="primary",
            prompt_hash="test_hash",
            raw_response="{}",
            schema_valid=True,
        ))

        call_count = 0

        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return ProviderResult("{}", 10, 5, 100)

        orch = AnnotationOrchestrator(config=config, store=store)
        with patch.object(orch, "_call_provider", side_effect=mock_call):
            with patch(
                "pet_annotation.teacher.orchestrator.render_prompt",
                return_value=("sys", "usr"),
            ):
                with patch(
                    "pet_annotation.teacher.orchestrator.compute_prompt_hash",
                    return_value="test_hash",
                ):
                    await orch.run()

        assert call_count == 0

    async def test_failed_frame_reverts_to_pending(self, db_conn):
        """Frames that fail annotation revert to pending status."""
        config = _make_config()
        store = AnnotationStore(conn=db_conn)
        _insert_frames(db_conn, 1)

        async def mock_fail(*args, **kwargs):
            raise RuntimeError("API error")

        orch = AnnotationOrchestrator(config=config, store=store)
        with patch.object(orch, "_call_provider", side_effect=mock_fail):
            with patch(
                "pet_annotation.teacher.orchestrator.render_prompt",
                return_value=("sys", "usr"),
            ):
                await orch.run()

        row = db_conn.execute(
            "SELECT annotation_status FROM frames WHERE frame_id='frame_000'"
        ).fetchone()
        assert row[0] == "pending"
