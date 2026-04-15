"""Tests for ProviderRegistry."""
from __future__ import annotations

import pytest

from pet_annotation.config import (
    AccountConfig,
    AnnotationConfig,
    AnnotationParams,
    DatabaseConfig,
    DpoParams,
    ModelConfig,
)
from pet_annotation.teacher.provider import ProviderRegistry
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


@pytest.fixture
def config() -> AnnotationConfig:
    """Build a config with three provider types."""
    return AnnotationConfig(
        database=DatabaseConfig(path=":memory:", data_root="/tmp"),
        annotation=AnnotationParams(
            batch_size=8, max_concurrent=10, max_daily_tokens=1000,
            review_sampling_rate=0.15, low_confidence_threshold=0.70,
            primary_model="model-a", schema_version="1.0",
        ),
        models={
            "model-a": ModelConfig(
                provider="openai_compat", base_url="http://a/v1", model_name="a",
                accounts=[AccountConfig(key_env="K1", rpm=10, tpm=1000)],
            ),
            "model-b": ModelConfig(
                provider="doubao", base_url="http://b/v1", model_name="b",
                accounts=[AccountConfig(key_env="K2", rpm=10, tpm=1000)],
            ),
            "model-c": ModelConfig(
                provider="vllm", base_url="http://c/v1", model_name="c",
                accounts=[AccountConfig(key_env="", rpm=999, tpm=999999)],
            ),
        },
        dpo=DpoParams(min_pairs_per_release=100),
    )


class TestProviderRegistry:
    def test_get_primary(self, config):
        """Primary model returns the correct provider type."""
        reg = ProviderRegistry(config)
        name, provider, tracker = reg.get_primary()
        assert name == "model-a"
        assert isinstance(provider, OpenAICompatProvider)

    def test_get_all(self, config):
        """All configured models are available."""
        reg = ProviderRegistry(config)
        all_providers = reg.get_all()
        assert len(all_providers) == 3
        names = {n for n, _, _ in all_providers}
        assert names == {"model-a", "model-b", "model-c"}

    def test_provider_types(self, config):
        """Each model gets the correct provider class."""
        reg = ProviderRegistry(config)
        types = {n: type(p).__name__ for n, p, _ in reg.get_all()}
        assert types["model-a"] == "OpenAICompatProvider"
        assert types["model-b"] == "DoubaoProvider"
        assert types["model-c"] == "VLLMProvider"

    def test_unknown_provider_raises(self):
        """Unknown provider type raises ValueError."""
        config = AnnotationConfig(
            database=DatabaseConfig(path=":memory:", data_root="/tmp"),
            annotation=AnnotationParams(
                primary_model="bad", schema_version="1.0",
            ),
            models={
                "bad": ModelConfig(
                    provider="unknown_type", base_url="http://x", model_name="x",
                    accounts=[AccountConfig(key_env="K", rpm=1, tpm=1)],
                ),
            },
            dpo=DpoParams(),
        )
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderRegistry(config)
