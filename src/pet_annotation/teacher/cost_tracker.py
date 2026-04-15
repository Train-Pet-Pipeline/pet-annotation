"""Daily token budget tracker — stops annotation when limit is reached."""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks daily token usage across all models.

    Args:
        max_daily_tokens: Maximum tokens allowed per day.
    """

    def __init__(self, max_daily_tokens: int) -> None:
        self._max = max_daily_tokens
        self._total = 0
        self._per_model: dict[str, int] = defaultdict(int)

    @property
    def remaining(self) -> int:
        """Tokens remaining in the daily budget."""
        return max(0, self._max - self._total)

    def check_and_record(self, tokens: int, model_name: str = "_all") -> bool:
        """Record token usage and check if still within budget.

        Args:
            tokens: Number of tokens consumed.
            model_name: Which model used them.

        Returns:
            True if within budget after recording, False if limit reached.
        """
        if self._total + tokens >= self._max:
            logger.warning(
                '{"event": "daily_token_limit", "total": %d, "max": %d, "model": "%s"}',
                self._total + tokens,
                self._max,
                model_name,
            )
            return False
        self._total += tokens
        self._per_model[model_name] += tokens
        return True

    def get_stats(self) -> dict[str, int]:
        """Return per-model and total token usage.

        Returns:
            Dict with model names as keys and token counts as values,
            plus a 'total' key.
        """
        stats = dict(self._per_model)
        stats["total"] = self._total
        return stats
