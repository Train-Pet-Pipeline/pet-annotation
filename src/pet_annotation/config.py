"""Configuration module for pet-annotation pipeline.

Loads params.yaml into validated Pydantic models and configures structured logging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pet_infra.logging import setup_logging as _infra_setup_logging
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AccountConfig(BaseModel):
    """Configuration for a single API account (key + rate limits)."""

    key_env: str
    rpm: int
    tpm: int

    def resolve_key(self) -> str:
        """Return the API key by reading the env var named by key_env.

        Returns an empty string when key_env is empty (e.g. local vLLM).
        """
        if not self.key_env:
            return ""
        return os.environ.get(self.key_env, "")


class ModelConfig(BaseModel):
    """Configuration for a single VLM provider/model."""

    provider: str
    base_url: str
    model_name: str
    accounts: list[AccountConfig]
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.1
    max_tokens: int = 2048


class DatabaseConfig(BaseModel):
    """Database connection parameters."""

    path: str
    data_root: str
    busy_timeout_ms: int = 10000


class LLMAnnotatorConfig(BaseModel):
    """Configuration for a single LLM annotator in the Phase 4 paradigm orchestrator.

    Attributes:
        id: Annotator identifier stored as annotator_id in annotation_targets.
        provider: Provider type; must match a registered provider class.
        base_url: Base URL for the provider API.
        model_name: Model identifier for the API.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        api_key: API key string; empty string means no auth header.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    provider: Literal["openai_compat", "vllm"]
    base_url: str
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 2048
    api_key: str = ""


class LLMParadigmConfig(BaseModel):
    """Configuration for the LLM annotator paradigm (Phase 4 orchestrator).

    Attributes:
        annotators: List of LLM annotator configs. N=0 is valid (no LLM annotation).
        batch_size: Number of targets to claim per batch.
        max_concurrent: Maximum number of concurrent provider requests.
    """

    model_config = ConfigDict(extra="forbid")

    annotators: list[LLMAnnotatorConfig] = []
    batch_size: int = 10
    max_concurrent: int = 4


class AnnotationParams(BaseModel):
    """Annotation pipeline tuning parameters."""

    batch_size: int = 16
    max_concurrent: int = 50
    max_daily_tokens: int = 10_000_000
    review_sampling_rate: float = 0.15
    low_confidence_threshold: float = 0.70
    primary_model: str
    schema_version: str = "1.0"
    modality_default: str = "vision"
    pet_data_db_path: str = "/data/pet-data/pet_data.db"


class QualityParams(BaseModel):
    """Quality-check tuning parameters."""

    anomaly_threshold: float = 0.3


class DpoParams(BaseModel):
    """DPO pair generation parameters."""

    min_pairs_per_release: int = 500


class ClassifierAnnotatorConfig(BaseModel):
    """Configuration for a single classifier annotator in the Phase 4 paradigm orchestrator.

    Attributes:
        id: Annotator identifier stored as annotator_id in annotation_targets.
        plugin: Entry-point name for the classifier plugin, e.g. 'audio_cnn_classifier'.
        model_path: Local filesystem path to classifier weights.
        device: Inference device, e.g. 'cpu' or 'cuda:0'.
        extra_params: Plugin-specific kwargs passed through at inference time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    plugin: str
    model_path: str
    device: str = "cpu"
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ClassifierParadigmConfig(BaseModel):
    """Configuration for the classifier annotator paradigm (Phase 4 orchestrator).

    Attributes:
        annotators: List of classifier annotator configs. N=0 is valid (no classifier annotation).
        batch_size: Number of targets to claim per batch.
        max_concurrent: Maximum number of concurrent inference threads.
    """

    model_config = ConfigDict(extra="forbid")

    annotators: list[ClassifierAnnotatorConfig] = []
    batch_size: int = 16
    max_concurrent: int = 2


class RuleAnnotatorConfig(BaseModel):
    """Configuration for a single rule annotator in the Phase 4 paradigm orchestrator.

    Attributes:
        id: Annotator identifier stored as annotator_id in annotation_targets.
        plugin: Entry-point name for the rule plugin, e.g. 'brightness_rule'.
        rule_id: Rule identifier stored in rule_annotations.rule_id.
        extra_params: Plugin-specific kwargs passed through at apply time.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    plugin: str
    rule_id: str
    extra_params: dict[str, Any] = Field(default_factory=dict)


class RuleParadigmConfig(BaseModel):
    """Configuration for the rule annotator paradigm (Phase 4 orchestrator).

    Attributes:
        annotators: List of rule annotator configs. N=0 is valid (no rule annotation).
        batch_size: Number of targets to claim per batch.
        max_concurrent: Maximum number of concurrent rule threads.
    """

    model_config = ConfigDict(extra="forbid")

    annotators: list[RuleAnnotatorConfig] = []
    batch_size: int = 50
    max_concurrent: int = 8


class HumanAnnotatorConfig(BaseModel):
    """Configuration for a single human annotator in the Phase 4 paradigm orchestrator.

    Attributes:
        id: Annotator identifier stored as annotator_id in annotation_targets.
        ls_base_url: Label Studio instance URL (e.g. http://localhost:8080).
        ls_project_id: Label Studio project ID to submit tasks to.
        ls_api_token_env: Name of the env var holding the LS API token (not the token itself).
        annotation_template: Template name in ls_templates/ for task rendering.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    ls_base_url: str
    ls_project_id: int
    ls_api_token_env: str = "LABEL_STUDIO_TOKEN"
    annotation_template: str = "default"


class HumanParadigmConfig(BaseModel):
    """Configuration for the human annotator paradigm (Phase 4 orchestrator).

    Attributes:
        annotators: List of human annotator configs. N=0 is valid (no human annotation).
        batch_size: Number of tasks per submit/pull batch.
    """

    model_config = ConfigDict(extra="forbid")

    annotators: list[HumanAnnotatorConfig] = []
    batch_size: int = 50


class AnnotationConfig(BaseModel):
    """Top-level configuration object for the pet-annotation pipeline."""

    database: DatabaseConfig
    annotation: AnnotationParams
    models: dict[str, ModelConfig]
    quality: QualityParams = QualityParams()
    dpo: DpoParams
    llm: LLMParadigmConfig = LLMParadigmConfig()
    classifier: ClassifierParadigmConfig = ClassifierParadigmConfig()
    rule: RuleParadigmConfig = RuleParadigmConfig()
    human: HumanParadigmConfig = HumanParadigmConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = Path(__file__).parent.parent.parent.parent / "params.yaml"


def load_config(params_path: Path | None = None) -> AnnotationConfig:
    """Load and validate configuration from params.yaml.

    Args:
        params_path: Path to params.yaml. Defaults to the project-root params.yaml.

    Returns:
        Validated AnnotationConfig instance.

    Raises:
        ValueError: If the primary_model is not present in the models dict.
    """
    path = params_path if params_path is not None else _DEFAULT_PARAMS
    raw: dict[str, Any] = yaml.safe_load(path.read_text())

    cfg = AnnotationConfig.model_validate(raw)

    if cfg.annotation.primary_model not in cfg.models:
        raise ValueError(
            f"primary_model '{cfg.annotation.primary_model}' not found in models dict. "
            f"Available models: {list(cfg.models.keys())}"
        )

    return cfg


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    """Configure structured JSON logging."""
    _infra_setup_logging("pet-annotation")
