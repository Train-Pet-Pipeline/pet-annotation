"""Provider abstraction for VLM API annotation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Type alias matching pet_schema.render_prompt() return type
PromptPair = tuple[str, str]  # (system_prompt, user_prompt)


@dataclass
class ProviderResult:
    """Result from a single annotation API call.

    Attributes:
        raw_response: Raw JSON string from the model's response content.
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens generated.
        latency_ms: Request round-trip time in milliseconds.
    """

    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens


class BaseProvider(ABC):
    """Abstract base class for VLM annotation providers.

    All providers must implement annotate() for single-frame inference.
    """

    @abstractmethod
    async def annotate(
        self, image_path: str, prompt: PromptPair, api_key: str
    ) -> ProviderResult:
        """Send a single frame for annotation.

        Args:
            image_path: Absolute path to the frame image.
            prompt: Tuple of (system_prompt, user_prompt).
            api_key: API key for authentication.

        Returns:
            ProviderResult with the model's response and usage stats.

        Raises:
            aiohttp.ClientError: On network errors (retriable).
            asyncio.TimeoutError: On request timeout (retriable).
        """

    @abstractmethod
    def supports_batch(self) -> bool:
        """Whether this provider supports native batch API.

        Returns:
            True if batch endpoint is available.
        """
