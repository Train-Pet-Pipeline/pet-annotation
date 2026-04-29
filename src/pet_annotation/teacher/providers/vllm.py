"""Self-hosted vLLM provider.

Uses OpenAI-compatible API served by vLLM. Skips auth when key is empty.
VLLMProvider inherits OpenAICompatProvider directly; the parent annotate()
already skips Authorization when api_key is empty, so no override needed.
"""

from __future__ import annotations

from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


class VLLMProvider(OpenAICompatProvider):
    """Provider for self-hosted vLLM instances.

    Inherits OpenAI-compatible protocol. Skips Authorization header
    when api_key is empty (local deployment). Uses higher default timeout
    suitable for on-premise GPU inference.

    Args:
        base_url: vLLM server URL (e.g. http://localhost:8000/v1).
        model_name: Model name as served by vLLM.
        timeout: Request timeout in seconds (default 120 for local).
        max_retries: Maximum retry attempts (default 2 for stability).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 120,
        max_retries: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        extra_payload: dict | None = None,
    ) -> None:
        """Initialize VLLMProvider with custom defaults for local deployment."""
        super().__init__(
            base_url, model_name, timeout, max_retries,
            temperature, max_tokens, extra_payload,
        )
