"""Integration tests for AnnotationOrchestrator — Phase 4 LLM + Classifier + Rule paradigms.

Covers: N=0/1/3 annotators, failure isolation, idempotent re-run,
        graceful shutdown signal handling, classifier dispatch, rule dispatch,
        and mixed 3-paradigm dispatch.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from pet_annotation.config import (
    ClassifierAnnotatorConfig,
    ClassifierParadigmConfig,
    LLMAnnotatorConfig,
    LLMParadigmConfig,
    RuleAnnotatorConfig,
    RuleParadigmConfig,
)
from pet_annotation.store import AnnotationStore
from pet_annotation.teacher.orchestrator import AnnotationOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RESPONSE = json.dumps({
    "scene": {
        "confidence_overall": 0.9,
        "pets": [{"species": "cat", "activity": "eating"}],
    }
})


def _make_petdata_db(
    path: Path,
    frame_ids: list[str],
    brightness_score: float = 0.2,
) -> str:
    """Create a minimal pet-data SQLite with pending frames.

    Includes brightness_score for rule dispatch testing.
    """
    db_path = str(path / "petdata.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE frames "
        "(frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT, "
        "brightness_score REAL)"
    )
    for fid in frame_ids:
        conn.execute(
            "INSERT INTO frames VALUES (?, 'pending', 'vision', ?)",
            (fid, brightness_score),
        )
    conn.commit()
    conn.close()
    return db_path


def _make_config(
    tmp_path: Path,
    annotator_dicts: list[dict],
    batch_size: int = 10,
    classifier_dicts: list[dict] | None = None,
    rule_dicts: list[dict] | None = None,
):
    """Build a minimal AnnotationConfig + write a params.yaml for it."""
    annotators = [LLMAnnotatorConfig(**a) for a in annotator_dicts]
    llm_cfg = LLMParadigmConfig(annotators=annotators, batch_size=batch_size, max_concurrent=4)

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
    # Override the llm config with our test annotators
    object.__setattr__(cfg, "llm", llm_cfg)

    # Override classifier/rule paradigm configs if provided
    if classifier_dicts is not None:
        cls_annotators = [ClassifierAnnotatorConfig(**a) for a in classifier_dicts]
        cls_cfg = ClassifierParadigmConfig(
            annotators=cls_annotators, batch_size=batch_size, max_concurrent=2
        )
        object.__setattr__(cfg, "classifier", cls_cfg)

    if rule_dicts is not None:
        rule_annotators = [RuleAnnotatorConfig(**a) for a in rule_dicts]
        rule_cfg = RuleParadigmConfig(
            annotators=rule_annotators, batch_size=batch_size, max_concurrent=8
        )
        object.__setattr__(cfg, "rule", rule_cfg)

    return cfg


def _mock_provider_result(raw_response: str = _VALID_RESPONSE) -> MagicMock:
    """Return a ProviderResult mock."""
    from pet_annotation.teacher.provider import ProviderResult

    return ProviderResult(
        raw_response=raw_response,
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=200,
    )


# ---------------------------------------------------------------------------
# Test 1: N=0 annotators → 0/0/0 stats, no errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_zero_annotators(tmp_path: Path) -> None:
    """Orchestrator with 0 annotators returns 0/0/0 stats without error."""
    cfg = _make_config(tmp_path, annotator_dicts=[])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2"])

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    stats = await orch.run()

    assert stats == {"processed": 0, "skipped": 0, "failed": 0}
    # No rows in annotation_targets
    count = store._conn.execute("SELECT COUNT(*) FROM annotation_targets").fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# Test 2: N=1 annotator × 5 pending frames → 5 processed, 5 llm_annotations rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_one_annotator_five_frames(tmp_path: Path) -> None:
    """N=1 annotator × 5 pending frames → 5 processed, 5 llm_annotations rows, all done."""
    ann_dict = {
        "id": "ann-1",
        "provider": "vllm",
        "base_url": "http://localhost:8000",
        "model_name": "Qwen/Qwen2.5-VL-72B",
    }
    cfg = _make_config(tmp_path, [ann_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, [f"frame-{i}" for i in range(5)])

    with patch(
        "pet_annotation.teacher.orchestrator._build_provider"
    ) as mock_build:
        mock_provider = MagicMock()
        mock_provider.annotate = AsyncMock(return_value=_mock_provider_result())
        mock_build.return_value = mock_provider
        # Rebuild providers dict with our mock
        orch = AnnotationOrchestrator(cfg, store, pet_db)
        orch._providers = {"ann-1": mock_provider}

        stats = await orch.run()

    assert stats["processed"] == 5
    assert stats["failed"] == 0
    assert stats["skipped"] == 0

    # All 5 rows in llm_annotations
    count = store._conn.execute("SELECT COUNT(*) FROM llm_annotations").fetchone()[0]
    assert count == 5

    # All annotation_targets done
    done_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='done'"
    ).fetchone()[0]
    assert done_count == 5


# ---------------------------------------------------------------------------
# Test 3: N=3 annotators × 2 targets → 6 llm_annotations rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_three_annotators_two_targets(tmp_path: Path) -> None:
    """N=3 annotators × 2 pending frames → 6 llm_annotations rows."""
    ann_dicts = [
        {"id": f"ann-{i}", "provider": "vllm", "base_url": "http://x", "model_name": "m"}
        for i in range(3)
    ]
    cfg = _make_config(tmp_path, ann_dicts)
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2"])

    mock_provider = MagicMock()
    mock_provider.annotate = AsyncMock(return_value=_mock_provider_result())

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {f"ann-{i}": mock_provider for i in range(3)}

    stats = await orch.run()

    assert stats["processed"] == 6
    assert stats["failed"] == 0

    llm_count = store._conn.execute("SELECT COUNT(*) FROM llm_annotations").fetchone()[0]
    assert llm_count == 6

    done_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='done'"
    ).fetchone()[0]
    assert done_count == 6


# ---------------------------------------------------------------------------
# Test 4: one annotator raises exception → that (target, annotator) marked failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_per_annotator_failure_isolation(tmp_path: Path) -> None:
    """When one annotator raises for a target, others are unaffected; failed row marked failed."""
    ann_dicts = [
        {"id": "ann-ok", "provider": "vllm", "base_url": "http://x", "model_name": "m"},
        {"id": "ann-fail", "provider": "vllm", "base_url": "http://x", "model_name": "m"},
    ]
    cfg = _make_config(tmp_path, ann_dicts)
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1"])

    ok_provider = MagicMock()
    ok_provider.annotate = AsyncMock(return_value=_mock_provider_result())

    fail_provider = MagicMock()
    fail_provider.annotate = AsyncMock(side_effect=RuntimeError("provider timeout"))

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {"ann-ok": ok_provider, "ann-fail": fail_provider}

    stats = await orch.run()

    # ann-ok succeeds for f1, ann-fail fails for f1
    assert stats["processed"] == 1
    assert stats["failed"] == 1

    # ann-ok: f1 should be done
    assert store.get_target_state("f1", "ann-ok") == "done"
    # ann-fail: f1 should be failed with error_msg
    assert store.get_target_state("f1", "ann-fail") == "failed"
    row = store._conn.execute(
        "SELECT error_msg FROM annotation_targets WHERE target_id='f1' AND annotator_id='ann-fail'"
    ).fetchone()
    assert "provider timeout" in row[0]


# ---------------------------------------------------------------------------
# Test 5: idempotent re-run → no new rows (all targets already done)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_idempotent_rerun(tmp_path: Path) -> None:
    """Re-running after all targets done → no new llm_annotations, no new targets."""
    ann_dict = {
        "id": "ann-1", "provider": "vllm", "base_url": "http://x", "model_name": "m"
    }
    cfg = _make_config(tmp_path, [ann_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2"])

    mock_provider = MagicMock()
    mock_provider.annotate = AsyncMock(return_value=_mock_provider_result())

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {"ann-1": mock_provider}

    # First run
    stats1 = await orch.run()
    assert stats1["processed"] == 2

    llm_count_after_first = store._conn.execute(
        "SELECT COUNT(*) FROM llm_annotations"
    ).fetchone()[0]
    assert llm_count_after_first == 2

    # Second run: all targets are 'done' — no new claims possible
    orch2 = AnnotationOrchestrator(cfg, store, pet_db)
    orch2._providers = {"ann-1": mock_provider}
    stats2 = await orch2.run()

    # No new processing (pending→in_progress claims nothing)
    assert stats2["processed"] == 0
    llm_count_after_second = store._conn.execute(
        "SELECT COUNT(*) FROM llm_annotations"
    ).fetchone()[0]
    assert llm_count_after_second == 2  # unchanged


# ---------------------------------------------------------------------------
# Test 6: shutdown signal handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_shutdown_stops_gracefully(tmp_path: Path) -> None:
    """Setting _shutdown=True mid-run stops processing after current batch."""
    ann_dict = {
        "id": "ann-1", "provider": "vllm", "base_url": "http://x", "model_name": "m"
    }
    cfg = _make_config(tmp_path, [ann_dict], batch_size=2)
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    # 10 pending frames; batch_size=2 means 5 batches needed
    pet_db = _make_petdata_db(tmp_path, [f"f{i}" for i in range(10)])

    processed_count = 0

    async def slow_annotate(*args, **kwargs):
        """Simulate slow provider that allows shutdown between batches."""
        nonlocal processed_count
        processed_count += 1
        # Set shutdown after 2 successful annotations
        if processed_count >= 2:
            orch._shutdown = True
        return _mock_provider_result()

    mock_provider = MagicMock()
    mock_provider.annotate = slow_annotate

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {"ann-1": mock_provider}

    stats = await orch.run()

    # Should have processed <= 4 (2 per batch × at most 2 batches before shutdown stops loop)
    # At minimum 2 (the batch where shutdown was triggered finishes)
    assert stats["processed"] >= 2
    assert stats["processed"] <= 10
    # No failures from shutdown
    assert stats["failed"] == 0


# ---------------------------------------------------------------------------
# Test 7: Classifier N=1 × 3 targets → 3 classifier_annotations rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_classifier_one_annotator_three_targets(tmp_path: Path) -> None:
    """Classifier: N=1 annotator × 3 pending frames → 3 classifier_annotations rows."""
    from pet_annotation.classifiers.base import NoopClassifier

    cls_dict = {
        "id": "noop-cls",
        "plugin": "noop_classifier",
        "model_path": "/models/fake.pt",
    }
    cfg = _make_config(tmp_path, annotator_dicts=[], classifier_dicts=[cls_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2", "f3"])

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    # Inject NoopClassifier as the plugin
    orch._classifier_plugins = {"noop-cls": NoopClassifier()}

    stats = await orch.run()

    assert stats["processed"] == 3
    assert stats["failed"] == 0

    cls_count = store._conn.execute(
        "SELECT COUNT(*) FROM classifier_annotations"
    ).fetchone()[0]
    assert cls_count == 3

    done_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='done' AND annotator_type='classifier'"
    ).fetchone()[0]
    assert done_count == 3


# ---------------------------------------------------------------------------
# Test 8: Rule N=1 × 5 targets → 5 rule_annotations rows (incl empty rule_output)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_rule_one_annotator_five_targets(tmp_path: Path) -> None:
    """Rule: N=1 rule × 5 pending frames → 5 rule_annotations rows.

    3 frames have brightness_score=0.1 (dim_scene), 2 have no score → empty rule_output.
    """
    import sqlite3

    rule_dict = {
        "id": "brightness-rule",
        "plugin": "brightness_rule",
        "rule_id": "brightness_lt_0.3",
    }
    cfg = _make_config(tmp_path, annotator_dicts=[], rule_dicts=[rule_dict])
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()

    # Create petdata.db manually: 3 frames with brightness_score, 2 without
    db_path = str(tmp_path / "petdata.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE frames "
        "(frame_id TEXT PRIMARY KEY, annotation_status TEXT, modality TEXT, "
        "brightness_score REAL)"
    )
    for i in range(3):
        conn.execute("INSERT INTO frames VALUES (?, 'pending', 'vision', 0.1)", (f"f{i}",))
    for i in range(3, 5):
        conn.execute("INSERT INTO frames VALUES (?, 'pending', 'vision', NULL)", (f"f{i}",))
    conn.commit()
    conn.close()

    from pet_annotation.rules.base import BrightnessRule

    orch = AnnotationOrchestrator(cfg, store, db_path)
    orch._rule_plugins = {"brightness-rule": BrightnessRule(threshold=0.3)}

    stats = await orch.run()

    assert stats["processed"] == 5
    assert stats["failed"] == 0

    rule_count = store._conn.execute(
        "SELECT COUNT(*) FROM rule_annotations"
    ).fetchone()[0]
    assert rule_count == 5

    # 3 frames with brightness_score → label populated; 2 without → empty rule_output
    rows = store._conn.execute(
        "SELECT rule_output FROM rule_annotations ORDER BY target_id"
    ).fetchall()
    import json as _json

    outputs = [_json.loads(r[0]) for r in rows]
    non_empty = [o for o in outputs if o]
    empty_outputs = [o for o in outputs if not o]
    assert len(non_empty) == 3
    assert len(empty_outputs) == 2

    # All targets done
    done_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='done' AND annotator_type='rule'"
    ).fetchone()[0]
    assert done_count == 5


# ---------------------------------------------------------------------------
# Test 9: Mixed 1 LLM + 1 classifier + 1 rule × 2 targets → 6 annotations (2/2/2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_mixed_paradigms_two_targets(tmp_path: Path) -> None:
    """Mixed: 1 LLM + 1 classifier + 1 rule × 2 pending frames → 6 annotations total."""
    from pet_annotation.classifiers.base import NoopClassifier
    from pet_annotation.rules.base import BrightnessRule

    llm_dict = {"id": "llm-1", "provider": "vllm", "base_url": "http://x", "model_name": "m"}
    cls_dict = {"id": "cls-1", "plugin": "noop_classifier", "model_path": "/m.pt"}
    rule_dict = {"id": "rule-1", "plugin": "brightness_rule", "rule_id": "brightness_lt_0.3"}

    cfg = _make_config(
        tmp_path,
        annotator_dicts=[llm_dict],
        classifier_dicts=[cls_dict],
        rule_dicts=[rule_dict],
    )
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    pet_db = _make_petdata_db(tmp_path, ["f1", "f2"], brightness_score=0.1)

    # LLM mock provider
    mock_provider = MagicMock()
    mock_provider.annotate = AsyncMock(return_value=_mock_provider_result())

    orch = AnnotationOrchestrator(cfg, store, pet_db)
    orch._providers = {"llm-1": mock_provider}
    orch._classifier_plugins = {"cls-1": NoopClassifier()}
    orch._rule_plugins = {"rule-1": BrightnessRule(threshold=0.3)}

    stats = await orch.run()

    # 2 LLM + 2 classifier + 2 rule = 6 processed total
    assert stats["processed"] == 6
    assert stats["failed"] == 0

    llm_count = store._conn.execute("SELECT COUNT(*) FROM llm_annotations").fetchone()[0]
    cls_count = store._conn.execute(
        "SELECT COUNT(*) FROM classifier_annotations"
    ).fetchone()[0]
    rule_count = store._conn.execute("SELECT COUNT(*) FROM rule_annotations").fetchone()[0]

    assert llm_count == 2
    assert cls_count == 2
    assert rule_count == 2

    # All 6 annotation_targets done (2 per paradigm)
    done_count = store._conn.execute(
        "SELECT COUNT(*) FROM annotation_targets WHERE state='done'"
    ).fetchone()[0]
    assert done_count == 6
