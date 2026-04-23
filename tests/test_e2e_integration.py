"""E2E integration test — full pipeline: pet-data fixture → orchestrator → 4 paradigms → export.

This is the definitive Phase 4 definition-of-done test. It exercises:

1. Create a tmp pet-data DB with 5 pending frames (including storage_uri).
2. Configure 1 LLM + 1 classifier + 1 rule + 1 human annotator.
3. Run orchestrator.run() (all paradigms; LS submit+pull mocked).
4. Verify:
   - 5 llm_annotations rows
   - 5 classifier_annotations rows
   - 5 rule_annotations rows
   - 5 annotation_targets done per LLM/classifier/rule annotator (15 done)
   - LS submit called with 5 tasks (mocked)
5. Simulate LS pull — 5 human_annotations rows, 5 targets done.
6. Export --format sft --annotator llm → JSONL with 5 valid entries.
7. Export --format dpo --annotator llm → 0 pairs (single annotator, expected).
8. Export --format sft --annotator classifier → 5 valid entries.

Total: 20 annotation_targets (5 × 4 annotators), all 20 done after human pull.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pet_annotation.classifiers.base import NoopClassifier
from pet_annotation.config import (
    ClassifierAnnotatorConfig,
    ClassifierParadigmConfig,
    HumanAnnotatorConfig,
    HumanParadigmConfig,
    LLMAnnotatorConfig,
    LLMParadigmConfig,
    RuleAnnotatorConfig,
    RuleParadigmConfig,
)
from pet_annotation.rules.base import BrightnessRule
from pet_annotation.store import AnnotationStore
from pet_annotation.teacher.orchestrator import AnnotationOrchestrator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FRAME_IDS = [f"frame-{i:03d}" for i in range(5)]
_VALID_RESPONSE = json.dumps({
    "scene": {"confidence_overall": 0.9, "pets": [{"species": "cat", "activity": "eating"}]}
})


def _make_petdata_db(path: Path) -> str:
    """Create a pet-data SQLite with 5 pending frames, each with a storage_uri."""
    db_path = str(path / "petdata.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE frames "
        "(frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT, "
        "brightness_score REAL, storage_uri TEXT)"
    )
    for fid in _FRAME_IDS:
        conn.execute(
            "INSERT INTO frames VALUES (?, 'pending', 'vision', 0.15, ?)",
            (fid, f"s3://pet-frames/{fid}.jpg"),
        )
    conn.commit()
    conn.close()
    return db_path


def _make_full_config(tmp_path: Path):
    """Build AnnotationConfig with 1 LLM + 1 classifier + 1 rule + 1 human annotator."""
    params = {
        "database": {"path": str(tmp_path / "ann.db"), "data_root": str(tmp_path)},
        "annotation": {
            "primary_model": "test-model",
            "schema_version": "1.0",
            "modality_default": "vision",
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
                id="llm-qwen", provider="vllm", base_url="http://localhost:8000",
                model_name="Qwen2.5-VL-72B"
            )
        ],
        batch_size=10,
    )
    cls_cfg = ClassifierParadigmConfig(
        annotators=[
            ClassifierAnnotatorConfig(
                id="noop-cls", plugin="noop_classifier", model_path="/models/noop.pt"
            )
        ],
        batch_size=10,
    )
    rule_cfg = RuleParadigmConfig(
        annotators=[
            RuleAnnotatorConfig(
                id="brightness-rule", plugin="brightness_rule",
                rule_id="brightness_lt_0.3"
            )
        ],
        batch_size=50,
    )
    human_cfg = HumanParadigmConfig(
        annotators=[
            HumanAnnotatorConfig(
                id="ls-project-1", ls_base_url="http://localhost:8080",
                ls_project_id=1
            )
        ],
        batch_size=50,
    )

    object.__setattr__(cfg, "llm", llm_cfg)
    object.__setattr__(cfg, "classifier", cls_cfg)
    object.__setattr__(cfg, "rule", rule_cfg)
    object.__setattr__(cfg, "human", human_cfg)
    return cfg


def _mock_ls_completed(frame_ids: list[str]) -> list[dict]:
    """Build LS completed task responses for all frames."""
    return [
        {
            "id": i + 1000,
            "meta": {"target_id": fid, "annotator_id": "ls-project-1"},
            "annotations": [
                {
                    "completed_by": {"email": "reviewer@example.com"},
                    "result": [
                        {"type": "choices", "value": {"choices": ["accept"]}}
                    ],
                }
            ],
        }
        for i, fid in enumerate(frame_ids)
    ]


# ---------------------------------------------------------------------------
# E2E Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_full_pipeline(tmp_path: Path) -> None:
    """Full E2E: 5 frames × 4 paradigms → 20 annotation_targets, all done; export valid."""
    from pet_annotation.teacher.provider import ProviderResult

    cfg = _make_full_config(tmp_path)
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path)

    # ------------------------------------------------------------------
    # Mock: LLM provider returns valid JSON for all frames
    # ------------------------------------------------------------------
    mock_provider = MagicMock()
    mock_provider.annotate = MagicMock(
        return_value=ProviderResult(
            raw_response=_VALID_RESPONSE,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=200,
        )
    )
    # Wrap to async
    from unittest.mock import AsyncMock
    mock_provider.annotate = AsyncMock(
        return_value=ProviderResult(
            raw_response=_VALID_RESPONSE,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=200,
        )
    )

    # ------------------------------------------------------------------
    # Mock: LS submit returns task IDs; LS pull returns 5 completed tasks
    # ------------------------------------------------------------------
    submitted_tasks: list[dict] = []

    def fake_submit_tasks(tasks: list[dict]) -> list[int]:
        submitted_tasks.extend(tasks)
        return list(range(1000, 1000 + len(tasks)))

    mock_ls_client = MagicMock()
    mock_ls_client.submit_tasks = fake_submit_tasks
    mock_ls_client.fetch_completed_annotations = MagicMock(
        return_value=_mock_ls_completed(_FRAME_IDS)
    )

    # ------------------------------------------------------------------
    # Build orchestrator with all plugins
    # ------------------------------------------------------------------
    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {"llm-qwen": mock_provider}
    orch._classifier_plugins = {"noop-cls": NoopClassifier()}
    orch._rule_plugins = {"brightness-rule": BrightnessRule(threshold=0.3)}

    # ------------------------------------------------------------------
    # Run all paradigms
    # ------------------------------------------------------------------
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
        stats = await orch.run(paradigms=["llm", "classifier", "rule", "human"])

    # ------------------------------------------------------------------
    # Verify annotation row counts
    # ------------------------------------------------------------------
    llm_count = store._conn.execute("SELECT COUNT(*) FROM llm_annotations").fetchone()[0]
    cls_count = store._conn.execute(
        "SELECT COUNT(*) FROM classifier_annotations"
    ).fetchone()[0]
    rule_count = store._conn.execute("SELECT COUNT(*) FROM rule_annotations").fetchone()[0]
    human_count = store._conn.execute(
        "SELECT COUNT(*) FROM human_annotations"
    ).fetchone()[0]

    assert llm_count == 5, f"Expected 5 LLM annotations, got {llm_count}"
    assert cls_count == 5, f"Expected 5 classifier annotations, got {cls_count}"
    assert rule_count == 5, f"Expected 5 rule annotations, got {rule_count}"
    assert human_count == 5, f"Expected 5 human annotations, got {human_count}"

    # ------------------------------------------------------------------
    # Verify annotation_targets: 20 total (5 frames × 4 annotators), all done
    # ------------------------------------------------------------------
    total_targets = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets"
    ).fetchone()[0]
    assert total_targets == 20, f"Expected 20 annotation_targets, got {total_targets}"

    done_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='done'"
    ).fetchone()[0]
    assert done_count == 20, f"Expected 20 done targets, got {done_count}"

    # ------------------------------------------------------------------
    # Verify LS submit was called with all 5 frames
    # ------------------------------------------------------------------
    assert len(submitted_tasks) == 5, f"Expected 5 LS tasks submitted, got {len(submitted_tasks)}"
    submitted_target_ids = {t["meta"]["target_id"] for t in submitted_tasks}
    assert submitted_target_ids == set(_FRAME_IDS)

    # Each submitted task has a resolvable storage_uri in data.image
    for task in submitted_tasks:
        assert task["data"]["image"].startswith("s3://pet-frames/")

    # ------------------------------------------------------------------
    # Verify LLM storage_uri was resolved and stored
    # ------------------------------------------------------------------
    uris = store._conn.execute(
        "SELECT storage_uri FROM llm_annotations"
    ).fetchall()
    assert all(r[0] is not None and r[0].startswith("s3://") for r in uris), \
        "All LLM annotations should have resolved storage_uri"

    # ------------------------------------------------------------------
    # Stats: 5 LLM + 5 cls + 5 rule + 5 human = 20 processed
    # ------------------------------------------------------------------
    assert stats["processed"] == 20
    assert stats["failed"] == 0

    # ------------------------------------------------------------------
    # Export SFT: llm annotator → 5 valid JSONL entries
    # ------------------------------------------------------------------
    from pet_annotation.export.sft_dpo import to_dpo_pairs, to_sft_samples

    sft_samples = to_sft_samples(store, annotator_type="llm")
    assert len(sft_samples) == 5

    for sample in sft_samples:
        assert sample["annotator_type"] == "llm"
        assert sample["input"].startswith("s3://pet-frames/")
        # Output should be parseable JSON
        output_data = json.loads(sample["output"])
        assert "scene" in output_data

    # ------------------------------------------------------------------
    # Export DPO: single LLM annotator → 0 pairs (nothing to pair against)
    # ------------------------------------------------------------------
    dpo_pairs = to_dpo_pairs(store, annotator_type="llm")
    assert len(dpo_pairs) == 0, "Single LLM annotator → no DPO pairs expected"

    # ------------------------------------------------------------------
    # Export SFT classifier → 5 valid entries
    # ------------------------------------------------------------------
    cls_samples = to_sft_samples(store, annotator_type="classifier")
    assert len(cls_samples) == 5
    for s in cls_samples:
        assert s["annotator_type"] == "classifier"
        output_data = json.loads(s["output"])
        assert "predicted_class" in output_data
        assert output_data["predicted_class"] == "unknown"  # NoopClassifier

    # ------------------------------------------------------------------
    # Export SFT human → 5 valid entries with decision='accept'
    # ------------------------------------------------------------------
    human_samples = to_sft_samples(store, annotator_type="human")
    assert len(human_samples) == 5
    for s in human_samples:
        assert s["annotator_type"] == "human"
        output_data = json.loads(s["output"])
        assert output_data["decision"] == "accept"
        assert output_data["reviewer"] == "reviewer@example.com"

    # ------------------------------------------------------------------
    # Final: all 4 paradigm tables populated → downstream-consumable output
    # ------------------------------------------------------------------
    # Rule: all 5 should have brightness label since all have score=0.15
    rule_outputs = store._conn.execute(
        "SELECT rule_output FROM rule_annotations ORDER BY target_id"
    ).fetchall()
    rule_data = [json.loads(r[0]) for r in rule_outputs]
    non_empty_rule = [r for r in rule_data if r]
    assert len(non_empty_rule) == 5, "All 5 frames have brightness_score → non-empty rule_output"
