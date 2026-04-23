"""OpenAI-compatible API provider for VLM annotation.

Covers: DashScope (Qwen), any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import aiohttp
from pet_infra.retry import standard_retry_async

from pet_annotation.teacher.provider import BaseProvider, PromptPair, ProviderResult

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseProvider):
    """Provider for OpenAI-compatible chat completion endpoints.

    Sends images as base64-encoded content parts in the user message.

    Args:
        base_url: Base URL for the API (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1).
        model_name: Model identifier for the API.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on transient errors.
        temperature: Sampling temperature (default 0.1 for deterministic annotation).
        max_tokens: Maximum tokens to generate in the response.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 60,
        max_retries: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._session: aiohttp.ClientSession | None = None
        self._call_api_with_retry = standard_retry_async(self._call_api, max_attempts=max_retries)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared aiohttp session.

        Returns:
            Active ClientSession.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """Close the shared aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_payload(self, image_path: str, prompt: PromptPair) -> dict:
        """Build the chat completions request payload (shared between providers).

        Args:
            image_path: Path to the image file.
            prompt: (system_prompt, user_prompt) tuple.

        Returns:
            Request payload dict with model, messages, temperature, and max_tokens.
        """
        system_prompt, user_prompt = prompt
        image_b64 = self._encode_image(image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        return {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

    async def annotate(self, image_path: str, prompt: PromptPair, api_key: str) -> ProviderResult:
        """Send a single frame for annotation via chat completions endpoint.

        Args:
            image_path: Path to the image file.
            prompt: (system_prompt, user_prompt) tuple.
            api_key: API key for Authorization header.

        Returns:
            ProviderResult with parsed response.
        """
        payload = self._build_payload(image_path, prompt)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self._base_url}/chat/completions"

        return await self._call_api_with_retry(url, payload, headers)

    async def _call_api(self, url: str, payload: dict, headers: dict) -> ProviderResult:
        """Make the API call (wrapped by tenacity retry).

        Args:
            url: Full endpoint URL.
            payload: Request body.
            headers: HTTP headers.

        Returns:
            ProviderResult.
        """
        start = time.monotonic()
        session = await self._get_session()
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        latency_ms = int((time.monotonic() - start) * 1000)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return ProviderResult(
            raw_response=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read and base64-encode an image file.

        Args:
            image_path: Path to the image.

        Returns:
            Base64-encoded string.
        """
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    def supports_batch(self) -> bool:
        """OpenAI-compat endpoints don't support native batch."""
        return False
