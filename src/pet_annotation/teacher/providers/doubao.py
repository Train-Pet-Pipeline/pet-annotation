"""Doubao (Volcengine) VLM provider.

Uses the Volcengine ARK API which is OpenAI-compatible with minor differences.
"""

from __future__ import annotations

from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


class DoubaoProvider(OpenAICompatProvider):
    """Provider for Doubao/Volcengine vision models.

    Inherits from OpenAICompatProvider since the Volcengine ARK API
    is OpenAI-compatible for chat completions.

    Args:
        base_url: Volcengine ARK API base URL.
        model_name: Doubao model identifier (endpoint ID).
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
    """

    pass  # Protocol is compatible; override only if Volcengine diverges
