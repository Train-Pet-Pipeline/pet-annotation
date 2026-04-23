"""Tests for AnnotationOrchestrator human paradigm — submit + pull paths.

Covers:
  - Submit path: mock LSClient assert correct tasks submitted with storage_uri
  - Pull path: 3 completed annotations → 3 human_annotations rows, targets → done
  - Auth failure: LS auth error marks targets failed
  - No completed tasks: graceful skip (skipped += 0, no rows)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pet_annotation.config import (
    HumanAnnotatorConfig,
    HumanParadigmConfig,
    LLMParadigmConfig,
)
from pet_annotation.store import AnnotationStore
from pet_annotation.teacher.orchestrator import AnnotationOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_petdata_db(path: Path, frame_ids: list[str], with_storage_uri: bool = True) -> str:
    """Create a minimal pet-data SQLite with pending frames.

    Args:
        path: Directory to create petdata.db in.
        frame_ids: List of frame_id values to insert as pending.
        with_storage_uri: If True, include a storage_uri column with a fake value.
    """
    db_path = str(path / "petdata.db")
    conn = sqlite3.connect(db_path)
    if with_storage_uri:
        conn.execute(
            "CREATE TABLE frames "
            "(frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT, "
            "storage_uri TEXT)"
        )
        for fid in frame_ids:
            conn.execute(
                "INSERT INTO frames VALUES (?, 'pending', 'vision', ?)",
                (fid, f"s3://bucket/{fid}.jpg"),
            )
    else:
        conn.execute(
            "CREATE TABLE frames "
            "(frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT)"
        )
        for fid in frame_ids:
            conn.execute(
                "INSERT INTO frames VALUES (?, 'pending', 'vision')", (fid,)
            )
    conn.commit()
    conn.close()
    return db_path


def _make_config_with_human(
    tmp_path: Path,
    human_dicts: list[dict],
    batch_size: int = 10,
):
    """Build a minimal AnnotationConfig with only human annotators configured."""
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
    # Override llm to empty (no LLM annotators)
    object.__setattr__(cfg, "llm", LLMParadigmConfig(annotators=[], batch_size=batch_size))

    human_annotators = [HumanAnnotatorConfig(**h) for h in human_dicts]
    human_cfg = HumanParadigmConfig(annotators=human_annotators, batch_size=batch_size)
    object.__setattr__(cfg, "human", human_cfg)
    return cfg


def _make_ls_completed_task(target_id: str, decision: str = "accept") -> dict:
    """Build a minimal LS completed task dict.

    Args:
        target_id: The target_id embedded in task meta.
        decision: The decision choice label.
    """
    return {
        "id": hash(target_id) % 10000,
        "meta": {"target_id": target_id, "annotator_id": "ls-project-1"},
        "annotations": [
            {
                "completed_by": {"email": "reviewer@example.com"},
                "result": [
                    {
                        "type": "choices",
                        "value": {"choices": [decision]},
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: Submit path — mock LSClient, assert tasks submitted with storage_uri
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_submit_path_sends_tasks_with_storage_uri(tmp_path: Path) -> None:
    """Human submit: 3 pending frames → 3 tasks submitted to LS with storage_uri in data."""
    human_dict = {
        "id": "ls-project-1",
        "ls_base_url": "http://localhost:8080",
        "ls_project_id": 1,
    }
    cfg = _make_config_with_human(tmp_path, [human_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2", "f3"], with_storage_uri=True)

    submitted_tasks: list[dict] = []

    def fake_submit_tasks(tasks):
        """Capture submitted tasks and return fake LS IDs."""
        submitted_tasks.extend(tasks)
        return list(range(100, 100 + len(tasks)))

    mock_ls_client = MagicMock()
    mock_ls_client.submit_tasks = fake_submit_tasks
    mock_ls_client.fetch_completed_annotations = MagicMock(return_value=[])

    with (
        patch(
            "pet_annotation.teacher.orchestrator.get_ls_session",
            return_value=MagicMock(),
        ),
        patch(
            "pet_annotation.teacher.orchestrator.LSClient",
            return_value=mock_ls_client,
        ),
    ):
        orch = AnnotationOrchestrator(cfg, store, pet_db)
        await orch.run(paradigms=["human"])

    # All 3 tasks should have been submitted
    assert len(submitted_tasks) == 3
    # Each task should have storage_uri in data.image
    for task in submitted_tasks:
        assert "data" in task
        assert "image" in task["data"]
        assert "s3://bucket/" in task["data"]["image"]
        # target_id should be in meta for round-trip
        assert "meta" in task
        assert "target_id" in task["meta"]

    # Targets should be in_progress after submit (not done — waiting for human)
    states = store._conn.execute(
        "SELECT state FROM annotation_targets"
    ).fetchall()
    assert all(s[0] == "in_progress" for s in states)

    # No human_annotations rows yet (pull returned nothing)
    human_count = store._conn.execute(
        "SELECT COUNT(*) FROM human_annotations"
    ).fetchone()[0]
    assert human_count == 0


# ---------------------------------------------------------------------------
# Test 2: Pull path — 3 completed tasks → 3 human_annotations rows, targets done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_pull_path_inserts_annotations_marks_done(tmp_path: Path) -> None:
    """Human pull: fetch 3 completed LS annotations → 3 rows in human_annotations, all done."""
    human_dict = {
        "id": "ls-project-1",
        "ls_base_url": "http://localhost:8080",
        "ls_project_id": 1,
    }
    cfg = _make_config_with_human(tmp_path, [human_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2", "f3"], with_storage_uri=True)

    completed = [
        _make_ls_completed_task("f1", "accept"),
        _make_ls_completed_task("f2", "reject"),
        _make_ls_completed_task("f3", "accept"),
    ]

    mock_ls_client = MagicMock()
    mock_ls_client.submit_tasks = MagicMock(return_value=[101, 102, 103])
    mock_ls_client.fetch_completed_annotations = MagicMock(return_value=completed)

    with (
        patch(
            "pet_annotation.teacher.orchestrator.get_ls_session",
            return_value=MagicMock(),
        ),
        patch(
            "pet_annotation.teacher.orchestrator.LSClient",
            return_value=mock_ls_client,
        ),
    ):
        orch = AnnotationOrchestrator(cfg, store, pet_db)
        stats = await orch.run(paradigms=["human"])

    # 3 human_annotations rows inserted
    human_count = store._conn.execute(
        "SELECT COUNT(*) FROM human_annotations"
    ).fetchone()[0]
    assert human_count == 3

    # All targets done
    done_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='done' AND annotator_type='human'"
    ).fetchone()[0]
    assert done_count == 3

    # Processed count = 3
    assert stats["processed"] == 3
    assert stats["failed"] == 0

    # Verify decisions stored correctly
    decisions = {
        r[0]: r[1]
        for r in store._conn.execute(
            "SELECT target_id, decision FROM human_annotations"
        ).fetchall()
    }
    assert decisions["f1"] == "accept"
    assert decisions["f2"] == "reject"
    assert decisions["f3"] == "accept"


# ---------------------------------------------------------------------------
# Test 3: LS auth failure → targets failed, graceful error log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_ls_auth_failure_marks_failed(tmp_path: Path) -> None:
    """When LS auth fails, orchestrator logs error and increments stats['failed']."""
    human_dict = {
        "id": "ls-project-1",
        "ls_base_url": "http://localhost:8080",
        "ls_project_id": 1,
    }
    cfg = _make_config_with_human(tmp_path, [human_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1"], with_storage_uri=True)

    with patch(
        "pet_annotation.teacher.orchestrator.get_ls_session",
        side_effect=RuntimeError("auth error"),
    ):
        orch = AnnotationOrchestrator(cfg, store, pet_db)
        stats = await orch.run(paradigms=["human"])

    assert stats["failed"] == 1
    assert stats["processed"] == 0


# ---------------------------------------------------------------------------
# Test 4: No completed tasks → no rows inserted, no error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_no_completed_tasks_graceful(tmp_path: Path) -> None:
    """When LS returns no completed tasks, no rows inserted and no error raised."""
    human_dict = {
        "id": "ls-project-1",
        "ls_base_url": "http://localhost:8080",
        "ls_project_id": 1,
    }
    cfg = _make_config_with_human(tmp_path, [human_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2"], with_storage_uri=True)

    mock_ls_client = MagicMock()
    mock_ls_client.submit_tasks = MagicMock(return_value=[101, 102])
    mock_ls_client.fetch_completed_annotations = MagicMock(return_value=[])

    with (
        patch(
            "pet_annotation.teacher.orchestrator.get_ls_session",
            return_value=MagicMock(),
        ),
        patch(
            "pet_annotation.teacher.orchestrator.LSClient",
            return_value=mock_ls_client,
        ),
    ):
        orch = AnnotationOrchestrator(cfg, store, pet_db)
        stats = await orch.run(paradigms=["human"])

    assert stats["processed"] == 0
    assert stats["failed"] == 0

    human_count = store._conn.execute(
        "SELECT COUNT(*) FROM human_annotations"
    ).fetchone()[0]
    assert human_count == 0

    # Targets are in_progress (submitted but not completed yet)
    in_progress = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='in_progress'"
    ).fetchone()[0]
    assert in_progress == 2


# ---------------------------------------------------------------------------
# Test 5: storage_uri fallback when column absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_submit_storage_uri_fallback_to_target_id(tmp_path: Path) -> None:
    """When storage_uri column is absent, task data.image falls back to target_id."""
    human_dict = {
        "id": "ls-project-1",
        "ls_base_url": "http://localhost:8080",
        "ls_project_id": 1,
    }
    cfg = _make_config_with_human(tmp_path, [human_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    # Pet-data DB without storage_uri column
    pet_db = _make_petdata_db(tmp_path, ["f1"], with_storage_uri=False)

    submitted_tasks: list[dict] = []

    def fake_submit(tasks):
        submitted_tasks.extend(tasks)
        return [200]

    mock_ls_client = MagicMock()
    mock_ls_client.submit_tasks = fake_submit
    mock_ls_client.fetch_completed_annotations = MagicMock(return_value=[])

    with (
        patch(
            "pet_annotation.teacher.orchestrator.get_ls_session",
            return_value=MagicMock(),
        ),
        patch(
            "pet_annotation.teacher.orchestrator.LSClient",
            return_value=mock_ls_client,
        ),
    ):
        orch = AnnotationOrchestrator(cfg, store, pet_db)
        await orch.run(paradigms=["human"])

    assert len(submitted_tasks) == 1
    # Fallback: data.image = target_id when storage_uri is None
    assert submitted_tasks[0]["data"]["image"] == "f1"
