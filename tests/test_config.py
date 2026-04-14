"""Tests for config module (TDD)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pet_annotation.config import AccountConfig, AnnotationConfig, load_config, setup_logging

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
