"""Ensure pet-annotation ships on Phase 2 foundation pins."""

import importlib.metadata

import pet_annotation
import pet_annotation._version_pins as pins


def test_version():
    assert pet_annotation.__version__ == "2.1.1"


def test_version_parity():
    """__version__ must match the installed package metadata."""
    installed = importlib.metadata.version("pet-annotation")
    assert pet_annotation.__version__ == installed


def test_foundation_pins():
    assert pins.PET_SCHEMA_PIN == "v3.2.1"
    assert pins.PET_INFRA_PIN == "v2.6.0"
