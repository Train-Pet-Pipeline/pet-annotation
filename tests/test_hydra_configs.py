"""Hydra config composition tests for pet-annotation config group."""
from pathlib import Path

from hydra import compose, initialize_config_dir

CFG_DIR = str((Path(__file__).parent.parent / "src" / "pet_annotation" / "configs").resolve())


def test_compose_dataset_vision_annotations():
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_annotation_vision",
            overrides=["dataset=vision_annotations"],
        )
    assert cfg.dataset.type == "pet_annotation.vision_annotations"
    assert cfg.dataset.modality == "vision"


def test_compose_override_audio():
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_annotation_vision",
            overrides=["dataset=audio_annotations"],
        )
    assert cfg.dataset.type == "pet_annotation.audio_annotations"
    assert cfg.dataset.modality == "audio"
