"""Peer-dependency fail-fast guards for pet-annotation.

Verifies Mode B (DEV_GUIDE §11.3): register_all() raises RuntimeError when
pet-schema is not installed.
"""

from __future__ import annotations

import sys

import pytest


def test_register_all_raises_when_pet_schema_missing(monkeypatch):
    """register_all() must raise RuntimeError with diagnostic message when pet-schema absent."""
    monkeypatch.setitem(sys.modules, "pet_schema.version", None)
    # Reload _register to re-evaluate module-level code with patched modules
    if "pet_annotation._register" in sys.modules:
        del sys.modules["pet_annotation._register"]
    import pet_annotation._register as reg  # noqa: PLC0415

    with pytest.raises(RuntimeError) as excinfo:
        reg.register_all()
    msg = str(excinfo.value)
    assert "pet-schema" in msg
    assert "peer" in msg.lower() or "peer dependency" in msg.lower() or "peer-dep" in msg.lower()
