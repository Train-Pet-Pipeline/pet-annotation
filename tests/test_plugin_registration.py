import importlib.metadata as md

from pet_infra.registry import DATASETS


def test_pet_annotation_entry_point_discoverable():
    eps = md.entry_points(group="pet_infra.plugins")
    names = {ep.name for ep in eps}
    assert "pet_annotation" in names


def test_register_all_registers_both_datasets():
    """Simulate pet-infra discovery invoking register_all() on a fresh registry state."""
    DATASETS.module_dict.pop("pet_annotation.vision_annotations", None)
    DATASETS.module_dict.pop("pet_annotation.audio_annotations", None)
    from pet_annotation import _register

    _register.register_all()
    assert "pet_annotation.vision_annotations" in DATASETS.module_dict
    assert "pet_annotation.audio_annotations" in DATASETS.module_dict
