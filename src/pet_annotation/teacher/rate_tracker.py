"""Rate-aware API key scheduler with sliding window tracking."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable

from pet_annotation.config import AccountConfig

logger = logging.getLogger(__name__)

# Sliding window duration in seconds
WINDOW_SECONDS = 60.0


class _KeyWindow:
    """Sliding window counters for a single API key."""

    def __init__(self, key: str, rpm: int, tpm: int) -> None:
        self.key = key
        self.rpm = rpm
        self.tpm = tpm
        self._request_times: deque[float] = deque()
        self._token_entries: deque[tuple[float, int]] = deque()

    def _prune(self) -> None:
        """Remove entries older than the window."""
        cutoff = time.monotonic() - WINDOW_SECONDS
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        while self._token_entries and self._token_entries[0][0] < cutoff:
            self._token_entries.popleft()

    @property
    def current_rpm(self) -> int:
        """Current requests in the window."""
        self._prune()
        return len(self._request_times)

    @property
    def current_tpm(self) -> int:
        """Current tokens in the window."""
        self._prune()
        return sum(t for _, t in self._token_entries)

    def rpm_headroom(self) -> float:
        """Fraction of RPM remaining (1.0 = fully available)."""
        return max(0.0, 1.0 - self.current_rpm / self.rpm)

    def tpm_headroom(self, estimated_tokens: int = 0) -> float:
        """Fraction of TPM remaining after estimated usage."""
        used = self.current_tpm + estimated_tokens
        return max(0.0, 1.0 - used / self.tpm)

    def can_acquire(self, estimated_tokens: int = 0) -> bool:
        """Check if this key can accept another request."""
        return self.current_rpm < self.rpm and (self.current_tpm + estimated_tokens) < self.tpm

    def record(self, tokens: int) -> None:
        """Record a completed request."""
        now = time.monotonic()
        self._request_times.append(now)
        self._token_entries.append((now, tokens))


class RateTracker:
    """Manages multiple API keys with rate-aware scheduling.

    Selects the key with the most headroom. When all keys are saturated,
    awaits until one becomes available.

    Args:
        accounts: List of account configurations with rate limits.
        key_resolver: Function to extract the actual API key string from an AccountConfig.
    """

    def __init__(
        self,
        accounts: list[AccountConfig],
        key_resolver: Callable[[AccountConfig], str] | None = None,
    ) -> None:
        resolver = key_resolver or (lambda a: a.resolve_key())
        self._windows: dict[str, _KeyWindow] = {}
        for acc in accounts:
            key = resolver(acc)
            self._windows[key] = _KeyWindow(key, acc.rpm, acc.tpm)
        self._event = asyncio.Event()
        self._event.set()

    async def acquire(self, estimated_tokens: int = 0) -> str:
        """Select the API key with the most headroom.

        Blocks until a key is available if all are saturated.

        Args:
            estimated_tokens: Expected token usage for this request.

        Returns:
            The selected API key string.
        """
        while True:
            best_key = None
            best_score = -1.0

            for key, window in self._windows.items():
                if not window.can_acquire(estimated_tokens):
                    continue
                # Score = min(rpm_headroom, tpm_headroom) — bottleneck-aware
                score = min(window.rpm_headroom(), window.tpm_headroom(estimated_tokens))
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key is not None:
                return best_key

            # All keys saturated — wait and retry
            logger.info('{"event": "rate_tracker_waiting", "reason": "all_keys_saturated"}')
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=1.0)
            except TimeoutError:
                pass  # Retry after timeout

    def record(self, key: str, tokens: int) -> None:
        """Record a completed request's token usage.

        Args:
            key: The API key that was used.
            tokens: Number of tokens consumed.
        """
        if key in self._windows:
            self._windows[key].record(tokens)
        self._event.set()
