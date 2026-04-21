"""Fail-fast guard: pet-annotation must raise ImportError if pet-infra is missing."""

import importlib
import sys

import pytest


def test_register_raises_friendly_error_if_pet_infra_missing(monkeypatch):
    """_register.py must raise ImportError with diagnostic message when pet-infra absent."""
    monkeypatch.setitem(sys.modules, "pet_infra", None)
    if "pet_annotation._register" in sys.modules:
        del sys.modules["pet_annotation._register"]
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("pet_annotation._register")
    msg = str(excinfo.value)
    assert "pet-infra" in msg
    assert "compatibility_matrix" in msg
