"""Tests for RateTracker."""
from __future__ import annotations

import asyncio

import pytest

from pet_annotation.config import AccountConfig
from pet_annotation.teacher.rate_tracker import RateTracker


@pytest.fixture
def two_keys() -> list[AccountConfig]:
    return [
        AccountConfig(key_env="K1", rpm=2, tpm=10000),
        AccountConfig(key_env="K2", rpm=2, tpm=10000),
    ]


class TestRateTracker:
    async def test_acquire_returns_key(self, two_keys):
        tracker = RateTracker(two_keys, key_resolver=lambda k: k.key_env)
        key = await tracker.acquire()
        assert key in ("K1", "K2")

    async def test_round_robin_when_equal(self, two_keys):
        tracker = RateTracker(two_keys, key_resolver=lambda k: k.key_env)
        k1 = await tracker.acquire()
        tracker.record(k1, tokens=100)
        k2 = await tracker.acquire()
        # After recording on k1, k2 should have more headroom
        assert k2 != k1

    async def test_respects_rpm_limit(self):
        accounts = [AccountConfig(key_env="K1", rpm=1, tpm=999999)]
        tracker = RateTracker(accounts, key_resolver=lambda k: k.key_env)
        k = await tracker.acquire()
        tracker.record(k, tokens=10)
        # Next acquire should wait since RPM=1 and we just used it
        # Use a short timeout to verify it blocks
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(tracker.acquire(), timeout=0.2)

    async def test_prefers_least_loaded_key(self, two_keys):
        tracker = RateTracker(two_keys, key_resolver=lambda k: k.key_env)
        # Load up K1
        tracker.record("K1", tokens=5000)
        # K2 should be preferred
        k = await tracker.acquire()
        assert k == "K2"

    async def test_tpm_awareness(self):
        accounts = [
            AccountConfig(key_env="K1", rpm=100, tpm=100),
            AccountConfig(key_env="K2", rpm=100, tpm=100),
        ]
        tracker = RateTracker(accounts, key_resolver=lambda k: k.key_env)
        # Exhaust K1's TPM
        tracker.record("K1", tokens=95)
        k = await tracker.acquire(estimated_tokens=10)
        assert k == "K2"
