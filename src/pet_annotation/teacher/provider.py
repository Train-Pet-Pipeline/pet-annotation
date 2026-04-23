"""Provider abstraction for VLM API annotation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pet_annotation.config import AnnotationConfig
from pet_annotation.teacher.rate_tracker import RateTracker

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
    async def annotate(self, image_path: str, prompt: PromptPair, api_key: str) -> ProviderResult:
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


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a provider class by name.

    Args:
        name: Provider type string (matches params.yaml 'provider' field).
        cls: The provider class.
    """
    _PROVIDER_CLASSES[name] = cls


class ProviderRegistry:
    """Instantiates and manages providers + rate trackers from config.

    Args:
        config: The loaded AnnotationConfig.
    """

    def __init__(self, config: AnnotationConfig) -> None:
        """Build providers and rate trackers from configuration.

        Args:
            config: Validated annotation configuration.

        Raises:
            ValueError: If an unknown provider type is encountered.
        """
        from pet_annotation.teacher.providers.doubao import DoubaoProvider
        from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider
        from pet_annotation.teacher.providers.vllm import VLLMProvider

        register_provider("openai_compat", OpenAICompatProvider)
        register_provider("doubao", DoubaoProvider)
        register_provider("vllm", VLLMProvider)

        self._primary_name = config.annotation.primary_model
        self._entries: dict[str, tuple[BaseProvider, RateTracker]] = {}

        for model_name, model_cfg in config.models.items():
            cls = _PROVIDER_CLASSES.get(model_cfg.provider)
            if cls is None:
                msg = f"Unknown provider type: '{model_cfg.provider}' for model '{model_name}'"
                raise ValueError(msg)

            provider = cls(  # type: ignore[call-arg]
                base_url=model_cfg.base_url,
                model_name=model_cfg.model_name,
                timeout=model_cfg.timeout,
                max_retries=model_cfg.max_retries,
                temperature=model_cfg.temperature,
                max_tokens=model_cfg.max_tokens,
            )
            tracker = RateTracker(model_cfg.accounts)
            self._entries[model_name] = (provider, tracker)

    def get_primary(self) -> tuple[str, BaseProvider, RateTracker]:
        """Return the primary model's (name, provider, tracker).

        Returns:
            Tuple of (model_name, provider_instance, rate_tracker).
        """
        provider, tracker = self._entries[self._primary_name]
        return self._primary_name, provider, tracker

    def get_all(self) -> list[tuple[str, BaseProvider, RateTracker]]:
        """Return all configured models.

        Returns:
            List of (model_name, provider_instance, rate_tracker) tuples.
        """
        return [(name, p, t) for name, (p, t) in self._entries.items()]

    async def close(self) -> None:
        """Close all provider sessions."""
        for _, (provider, _) in self._entries.items():
            if hasattr(provider, "close"):
                await provider.close()
