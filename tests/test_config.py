"""Tests for config module (TDD)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pet_annotation.config import (
    AccountConfig,
    AnnotationConfig,
    LLMAnnotatorConfig,
    LLMParadigmConfig,
    load_config,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_params(tmp_path: Path) -> Path:
    """Create a minimal valid params.yaml for testing."""
    config = {
        "database": {
            "path": "/data/pet-data/pet_data.db",
            "data_root": "/data/pet-data",
        },
        "annotation": {
            "batch_size": 16,
            "max_concurrent": 50,
            "max_daily_tokens": 10_000_000,
            "review_sampling_rate": 0.15,
            "low_confidence_threshold": 0.70,
            "primary_model": "test-model",
            "schema_version": "1.0",
        },
        "models": {
            "test-model": {
                "provider": "openai_compat",
                "base_url": "https://example.com/v1",
                "model_name": "test-model-instruct",
                "accounts": [
                    {"key_env": "TEST_API_KEY", "rpm": 60, "tpm": 100000},
                ],
                "timeout": 60,
                "max_retries": 3,
            }
        },
        "dpo": {
            "min_pairs_per_release": 500,
        },
        "dvc": {
            "remote": "local",
            "remote_path": "/data/dvc-cache",
        },
    }
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.dump(config))
    return params_file


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_config(minimal_params: Path) -> None:
    """load_config parses a minimal params.yaml and exposes all fields."""
    cfg = load_config(minimal_params)

    assert isinstance(cfg, AnnotationConfig)
    assert cfg.database.path == "/data/pet-data/pet_data.db"
    assert cfg.database.data_root == "/data/pet-data"
    assert cfg.annotation.batch_size == 16
    assert cfg.annotation.primary_model == "test-model"
    assert cfg.annotation.schema_version == "1.0"
    assert "test-model" in cfg.models
    model = cfg.models["test-model"]
    assert model.provider == "openai_compat"
    assert model.timeout == 60
    assert len(model.accounts) == 1
    assert cfg.dpo.min_pairs_per_release == 500


def test_primary_model_must_exist_in_models(tmp_path: Path) -> None:
    """load_config raises ValueError when primary_model is not in models dict."""
    config = {
        "database": {"path": "/data/db", "data_root": "/data"},
        "annotation": {
            "primary_model": "nonexistent-model",
            "schema_version": "1.0",
        },
        "models": {
            "other-model": {
                "provider": "openai_compat",
                "base_url": "https://example.com/v1",
                "model_name": "other-model-instruct",
                "accounts": [{"key_env": "KEY", "rpm": 60, "tpm": 100000}],
            }
        },
        "dpo": {"min_pairs_per_release": 500},
    }
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.dump(config))

    with pytest.raises(ValueError, match="nonexistent-model"):
        load_config(params_file)


def test_account_key_resolution(minimal_params: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_key() returns the env var value when key_env is set."""
    monkeypatch.setenv("TEST_API_KEY", "sk-test-secret-value")
    cfg = load_config(minimal_params)
    account = cfg.models["test-model"].accounts[0]
    assert account.resolve_key() == "sk-test-secret-value"


def test_empty_key_env_returns_empty(tmp_path: Path) -> None:
    """resolve_key() returns empty string when key_env is empty."""
    account = AccountConfig(key_env="", rpm=999, tpm=999999)
    assert account.resolve_key() == ""


def test_setup_logging_runs_without_error() -> None:
    """setup_logging() configures root logger without raising."""
    setup_logging()


def test_params_has_modality_default() -> None:
    """Verify that the actual params.yaml includes modality_default."""
    params_path = Path(__file__).parent.parent / "params.yaml"
    params = yaml.safe_load(params_path.read_text())
    assert params["annotation"]["modality_default"] == "vision"


# ---------------------------------------------------------------------------
# LLMAnnotatorConfig / LLMParadigmConfig tests (Phase 4)
# ---------------------------------------------------------------------------


def test_llm_paradigm_config_empty_annotators_default() -> None:
    """LLMParadigmConfig default has empty annotators list (N=0 valid)."""
    cfg = LLMParadigmConfig()
    assert cfg.annotators == []
    assert cfg.batch_size == 10
    assert cfg.max_concurrent == 4


def test_llm_annotator_config_valid() -> None:
    """LLMAnnotatorConfig validates all fields correctly."""
    cfg = LLMAnnotatorConfig(
        id="qwen25-vl-72b",
        provider="vllm",
        base_url="http://localhost:8000",
        model_name="Qwen/Qwen2.5-VL-72B-Instruct",
        temperature=0.1,
        max_tokens=2048,
        api_key="",
    )
    assert cfg.id == "qwen25-vl-72b"
    assert cfg.provider == "vllm"
    assert cfg.api_key == ""


def test_llm_annotator_config_extra_forbid() -> None:
    """LLMAnnotatorConfig must reject unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LLMAnnotatorConfig(
            id="a",
            provider="vllm",
            base_url="http://x",
            model_name="m",
            unknown_field="oops",
        )


def test_llm_paradigm_config_extra_forbid() -> None:
    """LLMParadigmConfig must reject unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LLMParadigmConfig(annotators=[], unknown_field="oops")


def test_llm_paradigm_with_one_annotator() -> None:
    """LLMParadigmConfig with 1 annotator parses correctly."""
    cfg = LLMParadigmConfig(
        annotators=[
            {
                "id": "local-qwen",
                "provider": "vllm",
                "base_url": "http://localhost:8000",
                "model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
            }
        ]
    )
    assert len(cfg.annotators) == 1
    assert cfg.annotators[0].id == "local-qwen"


def test_llm_paradigm_with_three_annotators() -> None:
    """LLMParadigmConfig with 3 annotators parses correctly."""
    annotators = [
        {"id": f"ann-{i}", "provider": "vllm", "base_url": "http://x", "model_name": "m"}
        for i in range(3)
    ]
    cfg = LLMParadigmConfig(annotators=annotators)
    assert len(cfg.annotators) == 3
    assert [a.id for a in cfg.annotators] == ["ann-0", "ann-1", "ann-2"]


def test_annotation_config_has_llm_field_with_default(minimal_params: Path) -> None:
    """AnnotationConfig parsed from minimal params has llm field with default."""
    cfg = load_config(minimal_params)
    assert hasattr(cfg, "llm")
    assert isinstance(cfg.llm, LLMParadigmConfig)
    assert cfg.llm.annotators == []


def test_annotation_params_pet_data_db_path_default() -> None:
    """AnnotationParams.pet_data_db_path has a default value."""
    from pet_annotation.config import AnnotationParams

    params = AnnotationParams(primary_model="x")
    assert params.pet_data_db_path == "/data/pet-data/pet_data.db"


# ---------------------------------------------------------------------------
# ClassifierAnnotatorConfig / ClassifierParadigmConfig tests (Phase 4 Subagent C)
# ---------------------------------------------------------------------------


def test_classifier_paradigm_config_empty_annotators_default() -> None:
    """ClassifierParadigmConfig default has empty annotators list (N=0 valid)."""
    from pet_annotation.config import ClassifierParadigmConfig

    cfg = ClassifierParadigmConfig()
    assert cfg.annotators == []
    assert cfg.batch_size == 16
    assert cfg.max_concurrent == 2


def test_classifier_annotator_config_valid() -> None:
    """ClassifierAnnotatorConfig validates all required and optional fields."""
    from pet_annotation.config import ClassifierAnnotatorConfig

    cfg = ClassifierAnnotatorConfig(
        id="feeding-activity-v1",
        plugin="audio_cnn_classifier",
        model_path="/models/feeding_cnn_v1.pt",
        device="cpu",
    )
    assert cfg.id == "feeding-activity-v1"
    assert cfg.plugin == "audio_cnn_classifier"
    assert cfg.model_path == "/models/feeding_cnn_v1.pt"
    assert cfg.device == "cpu"
    assert cfg.extra_params == {}


def test_classifier_annotator_config_extra_params() -> None:
    """ClassifierAnnotatorConfig stores plugin-specific extra_params."""
    from pet_annotation.config import ClassifierAnnotatorConfig

    cfg = ClassifierAnnotatorConfig(
        id="cnn-v2",
        plugin="audio_cnn_classifier",
        model_path="/m.pt",
        extra_params={"top_k": 3, "threshold": 0.5},
    )
    assert cfg.extra_params["top_k"] == 3
    assert cfg.extra_params["threshold"] == 0.5


def test_classifier_annotator_config_extra_forbid() -> None:
    """ClassifierAnnotatorConfig rejects unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    from pet_annotation.config import ClassifierAnnotatorConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ClassifierAnnotatorConfig(
            id="x",
            plugin="y",
            model_path="/z",
            unknown_field="oops",
        )


def test_classifier_paradigm_config_extra_forbid() -> None:
    """ClassifierParadigmConfig rejects unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    from pet_annotation.config import ClassifierParadigmConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ClassifierParadigmConfig(annotators=[], unknown_field="oops")


def test_classifier_paradigm_with_one_annotator() -> None:
    """ClassifierParadigmConfig with 1 annotator parses correctly."""
    from pet_annotation.config import ClassifierParadigmConfig

    cfg = ClassifierParadigmConfig(
        annotators=[
            {
                "id": "feeding-v1",
                "plugin": "audio_cnn_classifier",
                "model_path": "/models/feeding.pt",
            }
        ]
    )
    assert len(cfg.annotators) == 1
    assert cfg.annotators[0].id == "feeding-v1"


def test_classifier_paradigm_with_multiple_annotators() -> None:
    """ClassifierParadigmConfig with 3 annotators parses all correctly."""
    from pet_annotation.config import ClassifierParadigmConfig

    annotators = [
        {"id": f"cls-{i}", "plugin": "audio_cnn_classifier", "model_path": f"/m{i}.pt"}
        for i in range(3)
    ]
    cfg = ClassifierParadigmConfig(annotators=annotators)
    assert len(cfg.annotators) == 3
    assert [a.id for a in cfg.annotators] == ["cls-0", "cls-1", "cls-2"]


def test_annotation_config_has_classifier_field_with_default(minimal_params: Path) -> None:
    """AnnotationConfig parsed from minimal params has classifier field with default."""
    from pet_annotation.config import ClassifierParadigmConfig

    cfg = load_config(minimal_params)
    assert hasattr(cfg, "classifier")
    assert isinstance(cfg.classifier, ClassifierParadigmConfig)
    assert cfg.classifier.annotators == []


# ---------------------------------------------------------------------------
# RuleAnnotatorConfig / RuleParadigmConfig tests (Phase 4 Subagent C)
# ---------------------------------------------------------------------------


def test_rule_paradigm_config_empty_annotators_default() -> None:
    """RuleParadigmConfig default has empty annotators list (N=0 valid)."""
    from pet_annotation.config import RuleParadigmConfig

    cfg = RuleParadigmConfig()
    assert cfg.annotators == []
    assert cfg.batch_size == 50
    assert cfg.max_concurrent == 8


def test_rule_annotator_config_valid() -> None:
    """RuleAnnotatorConfig validates all required and optional fields."""
    from pet_annotation.config import RuleAnnotatorConfig

    cfg = RuleAnnotatorConfig(
        id="night-scene-rule",
        plugin="brightness_rule",
        rule_id="brightness_lt_0.3",
    )
    assert cfg.id == "night-scene-rule"
    assert cfg.plugin == "brightness_rule"
    assert cfg.rule_id == "brightness_lt_0.3"
    assert cfg.extra_params == {}


def test_rule_annotator_config_extra_params() -> None:
    """RuleAnnotatorConfig stores plugin-specific extra_params."""
    from pet_annotation.config import RuleAnnotatorConfig

    cfg = RuleAnnotatorConfig(
        id="bright-rule",
        plugin="brightness_rule",
        rule_id="brightness_lt_0.3",
        extra_params={"threshold": 0.3},
    )
    assert cfg.extra_params["threshold"] == 0.3


def test_rule_annotator_config_extra_forbid() -> None:
    """RuleAnnotatorConfig rejects unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    from pet_annotation.config import RuleAnnotatorConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RuleAnnotatorConfig(
            id="r",
            plugin="p",
            rule_id="rid",
            unknown_field="oops",
        )


def test_rule_paradigm_config_extra_forbid() -> None:
    """RuleParadigmConfig rejects unknown fields (extra='forbid')."""
    from pydantic import ValidationError

    from pet_annotation.config import RuleParadigmConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RuleParadigmConfig(annotators=[], unknown_field="oops")


def test_rule_paradigm_with_one_annotator() -> None:
    """RuleParadigmConfig with 1 annotator parses correctly."""
    from pet_annotation.config import RuleParadigmConfig

    cfg = RuleParadigmConfig(
        annotators=[
            {
                "id": "night-scene",
                "plugin": "brightness_rule",
                "rule_id": "brightness_lt_0.3",
                "extra_params": {"threshold": 0.3},
            }
        ]
    )
    assert len(cfg.annotators) == 1
    assert cfg.annotators[0].rule_id == "brightness_lt_0.3"


def test_rule_paradigm_with_multiple_annotators() -> None:
    """RuleParadigmConfig with 3 annotators parses all correctly."""
    from pet_annotation.config import RuleParadigmConfig

    annotators = [
        {"id": f"rule-{i}", "plugin": "some_rule", "rule_id": f"rid-{i}"}
        for i in range(3)
    ]
    cfg = RuleParadigmConfig(annotators=annotators)
    assert len(cfg.annotators) == 3
    assert [a.id for a in cfg.annotators] == ["rule-0", "rule-1", "rule-2"]


def test_annotation_config_has_rule_field_with_default(minimal_params: Path) -> None:
    """AnnotationConfig parsed from minimal params has rule field with default."""
    from pet_annotation.config import RuleParadigmConfig

    cfg = load_config(minimal_params)
    assert hasattr(cfg, "rule")
    assert isinstance(cfg.rule, RuleParadigmConfig)
    assert cfg.rule.annotators == []
