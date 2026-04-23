"""Tests for SFT/DPO export — to_sft_samples() / to_dpo_pairs() + CLI export command.

TDD:
  - 3 done LLM annotations → export sft → 3 valid JSONL entries
  - Export to file → file contains 3 lines
  - 2 annotators × 1 target → DPO pair formed (chosen/rejected)
  - Single annotator per target → DPO pairs skipped for llm
  - CLI export sft → echoes correct count
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime

import pytest
import yaml
from click.testing import CliRunner

from pet_annotation.cli import cli
from pet_annotation.store import AnnotationStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_store(tmp_path: Path) -> AnnotationStore:
    """Create an initialised AnnotationStore in tmp_path."""
    store = AnnotationStore(str(tmp_path / "ann.db"))
    store.init_schema()
    return store


def _seed_llm_annotations(
    store: AnnotationStore,
    target_ids: list[str],
    annotator_id: str = "llm-1",
    confidence: float = 0.9,
) -> None:
    """Insert done llm_annotations + annotation_targets rows."""
    from pet_schema import LLMAnnotation

    raw = json.dumps({"scene": {"confidence_overall": confidence, "pets": []}})
    for tid in target_ids:
        ann = LLMAnnotation(
            annotation_id=f"{tid}:{annotator_id}:aa",
            target_id=tid,
            annotator_id=annotator_id,
            annotator_type="llm",
            modality="vision",
            schema_version="1.0",
            created_at=datetime(2026, 4, 23),
            storage_uri=f"s3://bucket/{tid}.jpg",
            prompt_hash="ph",
            raw_response=raw,
            parsed_output=json.loads(raw),
        )
        store.insert_llm(ann)
        # Insert into annotation_targets as done
        store._conn.execute(
            "INSERT INTO annotation_targets "
            "(target_id, annotator_id, annotator_type, state, claimed_at, finished_at) "
            "VALUES (?, ?, 'llm', 'done', '2026-04-23', '2026-04-23')",
            (tid, annotator_id),
        )
    store._conn.commit()


def _seed_classifier_annotations(
    store: AnnotationStore, target_ids: list[str], annotator_id: str = "cls-1"
) -> None:
    """Insert done classifier_annotations + annotation_targets rows."""
    from pet_schema import ClassifierAnnotation

    for tid in target_ids:
        ann = ClassifierAnnotation(
            annotation_id=f"{tid}:{annotator_id}:cc",
            target_id=tid,
            annotator_id=annotator_id,
            annotator_type="classifier",
            modality="vision",
            schema_version="1.0",
            created_at=datetime(2026, 4, 23),
            storage_uri=None,
            predicted_class="cat",
            class_probs={"cat": 0.9, "dog": 0.1},
            logits=None,
        )
        store.insert_classifier(ann)
        store._conn.execute(
            "INSERT INTO annotation_targets "
            "(target_id, annotator_id, annotator_type, state, claimed_at, finished_at) "
            "VALUES (?, ?, 'classifier', 'done', '2026-04-23', '2026-04-23')",
            (tid, annotator_id),
        )
    store._conn.commit()


def _make_minimal_params(tmp_path: Path) -> Path:
    """Write a minimal params.yaml so CLI can load config."""
    params = {
        "database": {"path": str(tmp_path / "ann.db"), "data_root": str(tmp_path)},
        "annotation": {
            "primary_model": "test-model",
            "schema_version": "1.0",
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
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(params))
    return p


# ---------------------------------------------------------------------------
# Test 1: 3 done LLM annotations → to_sft_samples returns 3 valid entries
# ---------------------------------------------------------------------------


def test_to_sft_samples_llm_three_annotations(tmp_path: Path) -> None:
    """3 done LLM annotations → to_sft_samples returns 3 samples with expected schema."""
    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1", "f2", "f3"])

    samples = to_sft_samples(store, annotator_type="llm")

    assert len(samples) == 3
    for s in samples:
        assert "sample_id" in s
        assert "annotator_id" in s
        assert s["annotator_type"] == "llm"
        assert "input" in s
        assert "output" in s
        # input should be storage_uri (seeded as s3://bucket/<id>.jpg)
        assert s["input"].startswith("s3://bucket/")
        # output should be valid JSON
        json.loads(s["output"])


# ---------------------------------------------------------------------------
# Test 2: Export to file → file exists and has 3 JSONL lines
# ---------------------------------------------------------------------------


def test_to_sft_samples_writes_jsonl_file(tmp_path: Path) -> None:
    """to_sft_samples with output_path writes 3-line JSONL file."""
    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1", "f2", "f3"])

    out = tmp_path / "output" / "sft.jsonl"
    samples = to_sft_samples(store, annotator_type="llm", output_path=out)

    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        data = json.loads(line)
        assert data["annotator_type"] == "llm"


# ---------------------------------------------------------------------------
# Test 3: 2 annotators × 1 target → DPO pair formed
# ---------------------------------------------------------------------------


def test_to_dpo_pairs_two_annotators_one_target(tmp_path: Path) -> None:
    """2 LLM annotators for same target → 1 DPO pair with chosen/rejected."""
    from pet_annotation.export.sft_dpo import to_dpo_pairs

    store = _init_store(tmp_path)
    # High-confidence annotation from ann-1
    _seed_llm_annotations(store, ["f1"], annotator_id="ann-1", confidence=0.95)
    # Low-confidence annotation from ann-2
    _seed_llm_annotations(store, ["f1"], annotator_id="ann-2", confidence=0.3)

    pairs = to_dpo_pairs(store, annotator_type="llm")

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["sample_id"] == "f1"
    assert pair["chosen_annotator_id"] == "ann-1"  # higher confidence
    assert pair["rejected_annotator_id"] == "ann-2"
    assert pair["chosen"] != "" and pair["rejected"] != ""


# ---------------------------------------------------------------------------
# Test 4: Single annotator per target → DPO pairs skipped for llm
# ---------------------------------------------------------------------------


def test_to_dpo_pairs_single_annotator_skipped(tmp_path: Path) -> None:
    """Single LLM annotation per target → no DPO pairs produced."""
    from pet_annotation.export.sft_dpo import to_dpo_pairs

    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1", "f2", "f3"])

    pairs = to_dpo_pairs(store, annotator_type="llm")

    # Each target has only 1 annotation → no pairs possible
    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Test 5: Classifier export SFT → 3 valid entries with annotator_type='classifier'
# ---------------------------------------------------------------------------


def test_to_sft_samples_classifier_three_annotations(tmp_path: Path) -> None:
    """3 done classifier annotations → to_sft_samples returns 3 samples."""
    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)
    _seed_classifier_annotations(store, ["f1", "f2", "f3"])

    samples = to_sft_samples(store, annotator_type="classifier")

    assert len(samples) == 3
    for s in samples:
        assert s["annotator_type"] == "classifier"
        output = json.loads(s["output"])
        assert "predicted_class" in output
        assert "class_probs" in output


# ---------------------------------------------------------------------------
# Test 6: CLI export sft → exit 0, stderr shows "exported 3 sft samples"
# ---------------------------------------------------------------------------


def test_cli_export_sft_command(tmp_path: Path) -> None:
    """CLI export --format sft --annotator llm outputs JSONL and echoes count."""
    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1", "f2", "f3"])
    params = _make_minimal_params(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "export",
            "--format", "sft",
            "--annotator", "llm",
            "--params", str(params),
            "--db", str(tmp_path / "ann.db"),
        ],
    )

    assert result.exit_code == 0, result.output
    # Should echo export count message
    assert "exported 3 sft" in result.output


# ---------------------------------------------------------------------------
# Test 7: CLI export dpo → exit 0
# ---------------------------------------------------------------------------


def test_cli_export_dpo_command(tmp_path: Path) -> None:
    """CLI export --format dpo --annotator llm with 2 annotators → 1 pair."""
    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1"], annotator_id="ann-1", confidence=0.9)
    _seed_llm_annotations(store, ["f1"], annotator_id="ann-2", confidence=0.4)
    params = _make_minimal_params(tmp_path)
    out = tmp_path / "out.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "export",
            "--format", "dpo",
            "--annotator", "llm",
            "--params", str(params),
            "--db", str(tmp_path / "ann.db"),
            "--output", str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "exported 1 dpo" in result.output
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1
    pair = json.loads(lines[0])
    assert "chosen" in pair and "rejected" in pair
