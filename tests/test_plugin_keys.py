"""Tests that 4 paradigm dataset plugins are registered under the correct keys."""

from pet_infra.registry import DATASETS

from pet_annotation._register import register_all


def test_four_paradigm_keys_discoverable():
    """All 4 annotator paradigm keys must be registered in DATASETS."""
    register_all()
    expected = {
        "pet_annotation.llm",
        "pet_annotation.classifier",
        "pet_annotation.rule",
        "pet_annotation.human",
    }
    got = set(DATASETS.module_dict.keys())
    assert expected <= got, f"missing: {expected - got}"


def test_old_keys_not_registered():
    """Stale vision/audio_annotations plugin keys must not be present."""
    register_all()
    forbidden = {"pet_annotation.vision_annotations", "pet_annotation.audio_annotations"}
    got = set(DATASETS.module_dict.keys())
    assert forbidden.isdisjoint(got), f"stale keys still present: {forbidden & got}"
