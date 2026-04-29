"""v2.2.0 — LLMAnnotatorConfig.extra_payload threads opaque kwargs into VLM request body.

Used to disable Doubao's reasoning mode (thinking={"type":"disabled"}) at runtime,
which gives ~6× per-request speedup on doubao-seed-2-0-mini per probe v2 (2026-04-29).
"""
from pet_annotation.config import LLMAnnotatorConfig
from pet_annotation.teacher.orchestrator import _build_provider
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


def test_llm_annotator_config_accepts_extra_payload():
    cfg = LLMAnnotatorConfig(
        id="doubao-seed-2-0-mini",
        provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model_name="doubao-seed-2-0-mini-260215",
        extra_payload={"thinking": {"type": "disabled"}},
    )
    assert cfg.extra_payload == {"thinking": {"type": "disabled"}}


def test_extra_payload_default_is_none():
    cfg = LLMAnnotatorConfig(
        id="x",
        provider="openai_compat",
        base_url="https://example.com",
        model_name="m",
    )
    assert cfg.extra_payload is None


def test_provider_build_payload_merges_extra(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    p = OpenAICompatProvider(
        base_url="https://example.com/v1",
        model_name="m",
        extra_payload={"thinking": {"type": "disabled"}},
    )
    payload = p._build_payload(str(img), ("sys", "user"))
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == "m"
    assert payload["temperature"] == 0.1


def test_provider_no_extra_payload_no_thinking_key(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    p = OpenAICompatProvider(base_url="https://example.com/v1", model_name="m")
    payload = p._build_payload(str(img), ("sys", "user"))
    assert "thinking" not in payload


def test_build_provider_propagates_extra_payload():
    cfg = LLMAnnotatorConfig(
        id="d",
        provider="doubao",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model_name="doubao-seed-2-0-mini-260215",
        extra_payload={"thinking": {"type": "disabled"}},
    )
    provider = _build_provider(cfg)
    assert provider._extra_payload == {"thinking": {"type": "disabled"}}
