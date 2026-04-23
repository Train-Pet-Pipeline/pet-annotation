"""Classifier annotator plugin interface.

Each plugin implements annotate(target_data, **kwargs) -> (predicted_class, class_probs, logits).
Plugins are identified by plugin_name class variable and loaded by the orchestrator based on
ClassifierAnnotatorConfig.plugin.

Implementations are typically synchronous (local model inference). The orchestrator wraps
calls in asyncio.to_thread() for uniform async dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class BaseClassifierAnnotator(ABC):
    """Abstract classifier annotator plugin.

    Subclasses must declare:
        plugin_name: ClassVar[str]  # e.g. "audio_cnn_classifier"

    Implementations are typically synchronous (local inference). The orchestrator
    wraps calls with asyncio.to_thread() to keep dispatch uniformly async.
    """

    plugin_name: ClassVar[str]

    @abstractmethod
    def annotate(
        self,
        target_data: bytes | str,
        **kwargs: object,
    ) -> tuple[str, dict[str, float], list[float] | None]:
        """Run classification on target_data.

        Args:
            target_data: Raw bytes of the sample OR a file path string.
            **kwargs: Plugin-specific runtime parameters from extra_params.

        Returns:
            Tuple of (predicted_class, class_probs, logits) where:
                predicted_class: The top predicted class label.
                class_probs: Dict mapping class label -> probability (sum ~= 1.0).
                logits: Optional raw logit values before softmax; None if not available.
        """


class NoopClassifier(BaseClassifierAnnotator):
    """No-op classifier for testing; always predicts 'unknown' with uniform probability.

    Used in unit tests and dry-run scenarios where no real model is available.
    """

    plugin_name: ClassVar[str] = "noop_classifier"

    def annotate(
        self,
        target_data: bytes | str,
        **kwargs: object,
    ) -> tuple[str, dict[str, float], list[float] | None]:
        """Return fixed 'unknown' prediction regardless of input.

        Args:
            target_data: Ignored; accepted for interface compatibility.
            **kwargs: Ignored; accepted for interface compatibility.

        Returns:
            ('unknown', {'unknown': 1.0}, None)
        """
        return "unknown", {"unknown": 1.0}, None
