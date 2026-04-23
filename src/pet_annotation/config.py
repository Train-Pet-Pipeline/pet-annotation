"""Configuration module for pet-annotation pipeline.

Loads params.yaml into validated Pydantic models and configures structured logging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pet_infra.logging import setup_logging as _infra_setup_logging
from pydantic import BaseModel

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


class AnnotationParams(BaseModel):
    """Annotation pipeline tuning parameters."""

    batch_size: int = 16
    max_concurrent: int = 50
    max_daily_tokens: int = 10_000_000
    review_sampling_rate: float = 0.15
    low_confidence_threshold: float = 0.70
    primary_model: str
    schema_version: str = "1.0"


class QualityParams(BaseModel):
    """Quality-check tuning parameters."""

    anomaly_threshold: float = 0.3


class DpoParams(BaseModel):
    """DPO pair generation parameters."""

    min_pairs_per_release: int = 500


class AnnotationConfig(BaseModel):
    """Top-level configuration object for the pet-annotation pipeline."""

    database: DatabaseConfig
    annotation: AnnotationParams
    models: dict[str, ModelConfig]
    quality: QualityParams = QualityParams()
    dpo: DpoParams


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
