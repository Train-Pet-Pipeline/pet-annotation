"""Self-hosted vLLM provider.

Uses OpenAI-compatible API served by vLLM. Skips auth when key is empty.
"""

from __future__ import annotations

from pet_annotation.teacher.provider import PromptPair, ProviderResult
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


class VLLMProvider(OpenAICompatProvider):
    """Provider for self-hosted vLLM instances.

    Inherits OpenAI-compatible protocol. Skips Authorization header
    when api_key is empty (local deployment).

    Args:
        base_url: vLLM server URL (e.g. http://localhost:8000/v1).
        model_name: Model name as served by vLLM.
        timeout: Request timeout in seconds (default higher for local).
        max_retries: Maximum retry attempts.
    """

    def __init__(
        self, base_url: str, model_name: str, timeout: int = 120, max_retries: int = 2
    ) -> None:
        """Initialize VLLMProvider with custom defaults for local deployment.

        Args:
            base_url: vLLM server URL.
            model_name: Model name.
            timeout: Request timeout in seconds (default 120 for local).
            max_retries: Maximum retry attempts (default 2 for stability).
        """
        super().__init__(base_url, model_name, timeout, max_retries)

    async def annotate(self, image_path: str, prompt: PromptPair, api_key: str) -> ProviderResult:
        """Send a single frame for annotation via vLLM endpoint.

        Skips Authorization header when api_key is empty (local deployment).

        Args:
            image_path: Path to the image file.
            prompt: (system_prompt, user_prompt) tuple.
            api_key: API key for Authorization header (empty for local).

        Returns:
            ProviderResult with parsed response.
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

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        headers = {"Content-Type": "application/json"}
        # Skip Authorization header when api_key is empty (local vLLM deployment)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self._base_url}/chat/completions"

        return await self._call_api_with_retry(url, payload, headers)
