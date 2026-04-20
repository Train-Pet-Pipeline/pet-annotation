"""Ensure pet-annotation ships on Phase 2 foundation pins."""

import pet_annotation
import pet_annotation._version_pins as pins


def test_version():
    assert pet_annotation.__version__ == "1.1.0"


def test_foundation_pins():
    assert pins.PET_SCHEMA_PIN == "v2.0.0"
    assert pins.PET_INFRA_PIN == "v2.0.0"
