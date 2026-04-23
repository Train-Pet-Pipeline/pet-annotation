"""Tests for MVP storage_uri resolution — LLM + classifier paradigms.

Verifies that _fetch_storage_uri is called and that the resolved URI is:
- Passed to provider.annotate (LLM) and plugin.annotate (classifier) as image_path/target_data
- Stored in the annotation row's storage_uri column
- Falls back to target_id when storage_uri column is absent or row not found
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from pet_annotation.config import (
    ClassifierAnnotatorConfig,
    ClassifierParadigmConfig,
    LLMAnnotatorConfig,
    LLMParadigmConfig,
)
from pet_annotation.store import AnnotationStore
from pet_annotation.teacher.orchestrator import AnnotationOrchestrator


_VALID_RESPONSE = json.dumps({
    "scene": {"confidence_overall": 0.9, "pets": [{"species": "cat", "activity": "eating"}]}
})


def _make_petdata_db_with_storage_uri(path: Path, frame_ids: list[str]) -> str:
    """Create pet-data DB where each frame has a storage_uri."""
    db_path = str(path / "petdata.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE frames "
        "(frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT, storage_uri TEXT)"
    )
    for fid in frame_ids:
        conn.execute(
            "INSERT INTO frames VALUES (?, 'pending', 'vision', ?)",
            (fid, f"s3://frames/{fid}.jpg"),
        )
    conn.commit()
    conn.close()
    return db_path


def _make_petdata_db_without_storage_uri(path: Path, frame_ids: list[str]) -> str:
    """Create pet-data DB where frames table has NO storage_uri column."""
    db_path = str(path / "petdata.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE frames "
        "(frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT)"
    )
    for fid in frame_ids:
        conn.execute("INSERT INTO frames VALUES (?, 'pending', 'vision')", (fid,))
    conn.commit()
    conn.close()
    return db_path


def _make_llm_config(tmp_path: Path, batch_size: int = 10):
    """Build a minimal AnnotationConfig with one LLM annotator."""
    params = {
        "database": {"path": str(tmp_path / "ann.db"), "data_root": str(tmp_path)},
        "annotation": {
            "primary_model": "test-model",
            "schema_version": "1.0",
            "pet_data_db_path": str(tmp_path / "petdata.db"),
        },
        "models": {
            "test-model": {
                "provider": "openai_compat",
                "base_url": "http://x",
                "model_name": "m",
                "accounts": [{"key_env": "", "rpm": 999, "tpm": 999999}],
            }
        },
        "dpo": {"min_pairs_per_release": 500},
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump(params))

    from pet_annotation.config import load_config

    cfg = load_config(params_path)
    llm_cfg = LLMParadigmConfig(
        annotators=[
            LLMAnnotatorConfig(
                id="llm-1", provider="vllm", base_url="http://x", model_name="m"
            )
        ],
        batch_size=batch_size,
    )
    object.__setattr__(cfg, "llm", llm_cfg)
    return cfg


def _make_cls_config(tmp_path: Path, batch_size: int = 10):
    """Build a minimal AnnotationConfig with one classifier annotator."""
    params = {
        "database": {"path": str(tmp_path / "ann.db"), "data_root": str(tmp_path)},
        "annotation": {
            "primary_model": "test-model",
            "schema_version": "1.0",
            "pet_data_db_path": str(tmp_path / "petdata.db"),
        },
        "models": {
            "test-model": {
                "provider": "openai_compat",
                "base_url": "http://x",
                "model_name": "m",
                "accounts": [{"key_env": "", "rpm": 999, "tpm": 999999}],
            }
        },
        "dpo": {"min_pairs_per_release": 500},
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump(params))

    from pet_annotation.config import load_config

    cfg = load_config(params_path)
    cls_cfg = ClassifierParadigmConfig(
        annotators=[
            ClassifierAnnotatorConfig(
                id="cls-1", plugin="noop_classifier", model_path="/m.pt"
            )
        ],
        batch_size=batch_size,
    )
    object.__setattr__(cfg, "cls", cls_cfg)
    object.__setattr__(cfg, "classifier", cls_cfg)
    return cfg


def _mock_provider_result(raw: str = _VALID_RESPONSE):
    """Return a ProviderResult mock."""
    from pet_annotation.teacher.provider import ProviderResult

    return ProviderResult(
        raw_response=raw, prompt_tokens=10, completion_tokens=5, latency_ms=50
    )


# ---------------------------------------------------------------------------
# Test 1: LLM provider receives storage_uri, annotation row stores it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_provider_receives_storage_uri(tmp_path: Path) -> None:
    """When storage_uri column exists, provider.annotate called with resolved URI."""
    cfg = _make_llm_config(tmp_path)
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db_with_storage_uri(tmp_path, ["frame-001"])

    received_image_paths: list[str] = []

    async def capturing_annotate(image_path, prompt, api_key):
        received_image_paths.append(image_path)
        return _mock_provider_result()

    mock_provider = MagicMock()
    mock_provider.annotate = capturing_annotate

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {"llm-1": mock_provider}

    stats = await orch.run(paradigms=["llm"])

    assert stats["processed"] == 1
    # Provider should receive the resolved storage_uri, not the raw target_id
    assert len(received_image_paths) == 1
    assert received_image_paths[0] == "s3://frames/frame-001.jpg"

    # storage_uri should be persisted in the annotation row
    row = store._conn.execute(
        "SELECT storage_uri FROM llm_annotations WHERE target_id = 'frame-001'"
    ).fetchone()
    assert row is not None
    assert row[0] == "s3://frames/frame-001.jpg"


# ---------------------------------------------------------------------------
# Test 2: LLM falls back to target_id when storage_uri column absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_provider_fallback_to_target_id(tmp_path: Path) -> None:
    """When storage_uri column is absent, provider.annotate called with target_id."""
    cfg = _make_llm_config(tmp_path)
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    # DB without storage_uri column
    pet_db = _make_petdata_db_without_storage_uri(tmp_path, ["frame-002"])

    received_image_paths: list[str] = []

    async def capturing_annotate(image_path, prompt, api_key):
        received_image_paths.append(image_path)
        return _mock_provider_result()

    mock_provider = MagicMock()
    mock_provider.annotate = capturing_annotate

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {"llm-1": mock_provider}

    stats = await orch.run(paradigms=["llm"])

    assert stats["processed"] == 1
    # Fallback: provider receives target_id when storage_uri unavailable
    assert received_image_paths[0] == "frame-002"

    # storage_uri stored as None in annotation row
    row = store._conn.execute(
        "SELECT storage_uri FROM llm_annotations WHERE target_id = 'frame-002'"
    ).fetchone()
    assert row is not None
    assert row[0] is None


# ---------------------------------------------------------------------------
# Test 3: Classifier plugin receives resolved storage_uri
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classifier_plugin_receives_storage_uri(tmp_path: Path) -> None:
    """When storage_uri exists, plugin.annotate called with resolved URI."""
    from pet_annotation.classifiers.base import BaseClassifierAnnotator
    from typing import ClassVar

    received_target_data: list[str] = []

    class CapturingClassifier(BaseClassifierAnnotator):
        plugin_name: ClassVar[str] = "capturing_classifier"

        def annotate(self, target_data, **kwargs):
            received_target_data.append(str(target_data))
            return "cat", {"cat": 1.0}, None

    cfg = _make_cls_config(tmp_path)
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db_with_storage_uri(tmp_path, ["frame-003"])

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._classifier_plugins = {"cls-1": CapturingClassifier()}

    stats = await orch.run(paradigms=["classifier"])

    assert stats["processed"] == 1
    assert len(received_target_data) == 1
    assert received_target_data[0] == "s3://frames/frame-003.jpg"

    # storage_uri stored in classifier_annotations
    row = store._conn.execute(
        "SELECT storage_uri FROM classifier_annotations WHERE target_id = 'frame-003'"
    ).fetchone()
    assert row is not None
    assert row[0] == "s3://frames/frame-003.jpg"
