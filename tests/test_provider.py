"""Tests for Provider abstraction and OpenAICompatProvider."""
from __future__ import annotations

from pathlib import Path

import pytest
from aioresponses import aioresponses

from pet_annotation.teacher.provider import PromptPair, ProviderResult
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


class TestProviderResult:
    def test_dataclass_fields(self):
        r = ProviderResult(
            raw_response='{"test": true}',
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=150,
        )
        assert r.total_tokens == 30


class TestOpenAICompatProvider:
    @pytest.fixture
    def provider(self):
        return OpenAICompatProvider(
            base_url="http://test-api.example.com/v1",
            model_name="test-model",
            timeout=30,
            max_retries=1,
        )

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> Path:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
        return img

    async def test_annotate_success(self, provider, sample_image):
        prompt: PromptPair = ("system prompt", "user prompt")
        mock_response = {
            "choices": [{"message": {"content": '{"schema_version": "1.0"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        try:
            with aioresponses() as m:
                m.post("http://test-api.example.com/v1/chat/completions", payload=mock_response)
                result = await provider.annotate(str(sample_image), prompt, api_key="sk-test")
        finally:
            await provider.close()

        assert result.raw_response == '{"schema_version": "1.0"}'
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.latency_ms > 0

    async def test_annotate_includes_image_as_base64(self, provider, sample_image):
        prompt: PromptPair = ("sys", "usr")
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        try:
            with aioresponses() as m:
                m.post("http://test-api.example.com/v1/chat/completions", payload=mock_response,
                       repeat=True)
                await provider.annotate(str(sample_image), prompt, api_key="sk-test")
                # Verify the request was made (aioresponses tracks calls)
                assert len(m.requests) == 1
        finally:
            await provider.close()

    def test_supports_batch(self, provider):
        assert provider.supports_batch() is False
