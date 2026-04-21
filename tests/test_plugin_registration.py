"""Tests for pet-annotation entry-point and plugin registration."""

import importlib
import importlib.metadata as md

from pet_infra.registry import DATASETS


def test_pet_annotation_entry_point_discoverable():
    """pet_annotation entry-point must be discoverable in pet_infra.plugins group."""
    eps = md.entry_points(group="pet_infra.plugins")
    names = {ep.name for ep in eps}
    assert "pet_annotation" in names


def test_register_all_registers_four_paradigm_datasets():
    """Simulate pet-infra discovery invoking register_all() — 4 new keys must be present."""
    from pet_annotation.datasets import (
        classifier_annotations,
        human_annotations,
        llm_annotations,
        rule_annotations,
    )

    # Force re-registration by reloading plugin modules
    for mod in (llm_annotations, classifier_annotations, rule_annotations, human_annotations):
        importlib.reload(mod)

    for key in (
        "pet_annotation.llm",
        "pet_annotation.classifier",
        "pet_annotation.rule",
        "pet_annotation.human",
    ):
        assert key in DATASETS.module_dict, f"missing key: {key}"


def test_old_keys_absent_after_register_all():
    """Stale vision/audio_annotations plugin keys must not be registered."""
    from pet_annotation import _register

    _register.register_all()
    for stale in ("pet_annotation.vision_annotations", "pet_annotation.audio_annotations"):
        assert stale not in DATASETS.module_dict, f"stale key still present: {stale}"
