"""Rule annotator plugin interface.

Each rule is a deterministic function on target metadata that produces a rule_output dict.
Rules have no network IO and no model weights — they are pure functions over metadata fields.

Implementations are synchronous. The orchestrator wraps calls with asyncio.to_thread()
for uniform async dispatch framework and future scalability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class BaseRuleAnnotator(ABC):
    """Abstract rule annotator plugin.

    Subclasses must declare:
        rule_id: ClassVar[str]  # unique rule identifier; stored in rule_annotations.rule_id

    Implementations are pure functions (deterministic, no side effects). The orchestrator
    wraps calls with asyncio.to_thread() to keep dispatch uniformly async.
    """

    rule_id: ClassVar[str]

    @abstractmethod
    def apply(self, target_metadata: dict, **kwargs: object) -> dict:
        """Apply rule to target metadata.

        Args:
            target_metadata: Dict of metadata columns from the pet-data frames row.
            **kwargs: Plugin-specific runtime params from RuleAnnotatorConfig.extra_params.

        Returns:
            rule_output dict (must be JSON-serializable). Return an empty dict {}
            when the rule condition is not met (rule did not trigger).
        """


class BrightnessRule(BaseRuleAnnotator):
    """Threshold rule on brightness_score metadata field.

    Labels a frame as 'dim_scene' when brightness_score < threshold, or
    'normal_scene' otherwise. Returns {} if brightness_score is absent.

    Args:
        threshold: Brightness threshold below which to label as dim_scene (default 0.3).
    """

    rule_id: ClassVar[str] = "brightness_threshold"

    def __init__(self, threshold: float = 0.3) -> None:
        """Initialise with brightness threshold.

        Args:
            threshold: Brightness score boundary. Score < threshold → dim_scene.
        """
        self.threshold = threshold

    def apply(self, target_metadata: dict, **kwargs: object) -> dict:
        """Apply brightness threshold rule to target metadata.

        Args:
            target_metadata: Dict that may contain 'brightness_score' float field.
            **kwargs: Accepted for interface compatibility; not used by this rule.

        Returns:
            Dict with 'label', 'score', 'threshold' keys, or {} if field absent.
        """
        score = target_metadata.get("brightness_score")
        if score is None:
            return {}
        label = "dim_scene" if score < self.threshold else "normal_scene"
        return {"label": label, "score": score, "threshold": self.threshold}
