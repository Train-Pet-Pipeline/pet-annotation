"""Tests for SFT/DPO export — to_sft_samples() / to_dpo_pairs() + CLI export command.

Phase 5 rewrite: exporter now emits LLaMA-Factory-ready formats.
  - to_sft_samples(llm)    → ShareGPTSFTSample conversations format
  - to_sft_samples(human)  → ShareGPTSFTSample conversations format
  - to_sft_samples(classifier|rule) → empty list + UserWarning
  - to_dpo_pairs(llm)      → DPOSample {sample_id, chosen, rejected, ...} + prompt
  - to_dpo_pairs(non-llm)  → empty list + UserWarning
  - Producer-side validator: each sample passes model_validate() before write

TDD:
  1. 3 done LLM annotations → to_sft_samples returns 3 ShareGPTSFTSample dicts
  2. Each sample has conversations list with human + gpt turns; gpt turn = raw_response
  3. Each sample passes ShareGPTSFTSample.model_validate() (producer validator)
  4. Export to file → file contains 3 lines, each valid JSON with "conversations" key
  5. 2 annotators × 1 target → 1 DPO pair with prompt/chosen/rejected
  6. DPO pair passes DPOSample.model_validate() (producer validator)
  7. Single annotator per target → DPO pairs skipped for llm
  8. classifier type → empty list + UserWarning
  9. Malformed exporter output → validator raises (regression)
  10. CLI export sft → echoes correct count
  11. CLI export dpo → 1 pair, JSONL has prompt/chosen/rejected
  12. Human paradigm → ShareGPTSFTSample with gpt turn = serialised decision JSON
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

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


def _seed_human_annotations(
    store: AnnotationStore,
    target_ids: list[str],
    annotator_id: str = "human-1",
    decision: str = "accept",
    reviewer: str = "reviewer@example.com",
) -> None:
    """Insert done human_annotations + annotation_targets rows."""
    from pet_schema import HumanAnnotation

    for tid in target_ids:
        ann = HumanAnnotation(
            annotation_id=f"{tid}:{annotator_id}:hh",
            target_id=tid,
            annotator_id=annotator_id,
            annotator_type="human",
            modality="vision",
            schema_version="1.0",
            created_at=datetime(2026, 4, 23),
            storage_uri=None,
            reviewer=reviewer,
            decision=decision,
            notes="looks good",
        )
        store.insert_human(ann)
        store._conn.execute(
            "INSERT INTO annotation_targets "
            "(target_id, annotator_id, annotator_type, state, claimed_at, finished_at) "
            "VALUES (?, ?, 'human', 'done', '2026-04-23', '2026-04-23')",
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
# Test 1 + 2 + 3: LLM → ShareGPT conversations shape + producer validator
# ---------------------------------------------------------------------------


def test_to_sft_samples_llm_sharegpt_format(tmp_path: Path) -> None:
    """3 done LLM annotations → 3 ShareGPTSFTSample dicts with conversations list."""
    from pet_schema import ShareGPTSFTSample

    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1", "f2", "f3"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        samples = to_sft_samples(store, annotator_type="llm")

    assert len(samples) == 3

    for s in samples:
        # Must have conversations list, not old flat keys
        assert "conversations" in s, "Must have conversations key (ShareGPT format)"
        assert "input" not in s, "Old flat 'input' key must not be present"
        assert "output" not in s, "Old flat 'output' key must not be present"

        convs = s["conversations"]
        assert len(convs) >= 2
        # Last turn must be gpt turn with annotation content
        gpt_turns = [c for c in convs if c["from"] == "gpt"]
        assert len(gpt_turns) == 1
        gpt_value = gpt_turns[0]["value"]
        # gpt turn should contain the raw LLM response (valid JSON with scene)
        parsed = json.loads(gpt_value)
        assert "scene" in parsed

        # Lineage fields present
        assert s.get("sample_id") is not None
        assert s.get("annotator_id") == "llm-1"

        # Producer-side validator: must not raise
        ShareGPTSFTSample.model_validate(s)


# ---------------------------------------------------------------------------
# Test 4: Export to file → 3 lines, each has conversations key
# ---------------------------------------------------------------------------


def test_to_sft_samples_writes_jsonl_file(tmp_path: Path) -> None:
    """to_sft_samples with output_path writes 3-line JSONL file in ShareGPT format."""
    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1", "f2", "f3"])

    out = tmp_path / "output" / "sft.jsonl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        to_sft_samples(store, annotator_type="llm", output_path=out)

    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        data = json.loads(line)
        assert "conversations" in data
        assert data.get("annotator_id") is not None


# ---------------------------------------------------------------------------
# Test 5 + 6: DPO pair → prompt/chosen/rejected + producer validator
# ---------------------------------------------------------------------------


def test_to_dpo_pairs_two_annotators_one_target(tmp_path: Path) -> None:
    """2 LLM annotators for same target → 1 DPO pair with prompt/chosen/rejected."""
    from pet_schema import DPOSample

    from pet_annotation.export.sft_dpo import to_dpo_pairs

    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1"], annotator_id="ann-1", confidence=0.95)
    _seed_llm_annotations(store, ["f1"], annotator_id="ann-2", confidence=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pairs = to_dpo_pairs(store, annotator_type="llm")

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["sample_id"] == "f1"
    assert pair["chosen_annotator_id"] == "ann-1"  # higher confidence
    assert pair["rejected_annotator_id"] == "ann-2"
    assert pair["chosen"] != "" and pair["rejected"] != ""
    # prompt field must be present (LLaMA-Factory Alpaca DPO requirement)
    assert "prompt" in pair
    assert pair["prompt"] != ""

    # Producer-side validator: validate DPOSample fields (sans prompt which is injected)
    pair_no_prompt = {k: v for k, v in pair.items() if k != "prompt"}
    DPOSample.model_validate(pair_no_prompt)


# ---------------------------------------------------------------------------
# Test 7: Single annotator per target → DPO pairs skipped
# ---------------------------------------------------------------------------


def test_to_dpo_pairs_single_annotator_skipped(tmp_path: Path) -> None:
    """Single LLM annotation per target → no DPO pairs produced."""
    from pet_annotation.export.sft_dpo import to_dpo_pairs

    store = _init_store(tmp_path)
    _seed_llm_annotations(store, ["f1", "f2", "f3"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pairs = to_dpo_pairs(store, annotator_type="llm")

    assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Test 8: Classifier type → empty list + UserWarning
# ---------------------------------------------------------------------------


def test_to_sft_samples_classifier_returns_empty_with_warning(tmp_path: Path) -> None:
    """classifier annotator_type → empty list + UserWarning (not natural SFT shape)."""
    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)
    _seed_classifier_annotations(store, ["f1", "f2", "f3"])

    with pytest.warns(UserWarning, match="classifier"):
        samples = to_sft_samples(store, annotator_type="classifier")

    assert samples == []


def test_to_sft_samples_rule_returns_empty_with_warning(tmp_path: Path) -> None:
    """rule annotator_type → empty list + UserWarning."""
    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)

    with pytest.warns(UserWarning, match="rule"):
        samples = to_sft_samples(store, annotator_type="rule")

    assert samples == []


def test_to_dpo_pairs_non_llm_returns_empty_with_warning(tmp_path: Path) -> None:
    """non-llm DPO → empty list + UserWarning."""
    from pet_annotation.export.sft_dpo import to_dpo_pairs

    store = _init_store(tmp_path)

    with pytest.warns(UserWarning, match="DPO pair"):
        pairs = to_dpo_pairs(store, annotator_type="classifier")

    assert pairs == []


# ---------------------------------------------------------------------------
# Test 9: Malformed sample → validator raises (regression)
# ---------------------------------------------------------------------------


def test_producer_validator_raises_on_malformed_sample() -> None:
    """ShareGPTSFTSample.model_validate raises ValidationError on malformed dict."""
    from pet_schema import ShareGPTSFTSample
    from pydantic import ValidationError

    bad_dict = {"conversations": [{"from": "unknown_role", "value": "x"}]}
    with pytest.raises(ValidationError):
        ShareGPTSFTSample.model_validate(bad_dict)


def test_producer_validator_raises_on_malformed_dpo_pair() -> None:
    """DPOSample.model_validate raises ValidationError when required fields missing."""
    from pet_schema import DPOSample
    from pydantic import ValidationError

    bad_dict = {"chosen": "a"}  # missing required fields
    with pytest.raises(ValidationError):
        DPOSample.model_validate(bad_dict)


# ---------------------------------------------------------------------------
# Test 10: CLI export sft → exit 0, output shows count
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
    assert "exported 3 sft" in result.output


# ---------------------------------------------------------------------------
# Test 11: CLI export dpo → 1 pair, JSONL has prompt
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
    assert "prompt" in pair


# ---------------------------------------------------------------------------
# Test 12: Human paradigm → ShareGPTSFTSample with gpt turn = serialised decision
# ---------------------------------------------------------------------------


def test_to_sft_samples_human_sharegpt_format(tmp_path: Path) -> None:
    """Human annotations → ShareGPTSFTSample with gpt turn containing decision JSON."""
    from pet_schema import ShareGPTSFTSample

    from pet_annotation.export.sft_dpo import to_sft_samples

    store = _init_store(tmp_path)
    _seed_human_annotations(store, ["h1", "h2"], decision="accept", reviewer="rev@x.com")

    samples = to_sft_samples(store, annotator_type="human")

    assert len(samples) == 2
    for s in samples:
        assert "conversations" in s
        ShareGPTSFTSample.model_validate(s)

        gpt_turns = [c for c in s["conversations"] if c["from"] == "gpt"]
        assert len(gpt_turns) == 1
        decision_data = json.loads(gpt_turns[0]["value"])
        assert decision_data["decision"] == "accept"
        assert decision_data["reviewer"] == "rev@x.com"
