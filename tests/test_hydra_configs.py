"""Hydra config composition tests for pet-annotation 4-paradigm dataset configs."""

from pathlib import Path

from hydra import compose, initialize_config_dir

CFG_DIR = str((Path(__file__).parent.parent / "src" / "pet_annotation" / "configs").resolve())


def test_compose_dataset_llm_annotations():
    """Dataset config for llm_annotations must resolve correct type."""
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_annotation_vision",
            overrides=["dataset=llm_annotations"],
        )
    assert cfg.dataset.type == "pet_annotation.llm"
    assert cfg.dataset.modality == "multimodal"


def test_compose_override_classifier():
    """Dataset config for classifier_annotations must resolve correct type."""
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_annotation_vision",
            overrides=["dataset=classifier_annotations"],
        )
    assert cfg.dataset.type == "pet_annotation.classifier"
    assert cfg.dataset.modality == "multimodal"


def test_compose_override_rule():
    """Dataset config for rule_annotations must resolve correct type."""
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_annotation_vision",
            overrides=["dataset=rule_annotations"],
        )
    assert cfg.dataset.type == "pet_annotation.rule"


def test_compose_override_human():
    """Dataset config for human_annotations must resolve correct type."""
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_annotation_vision",
            overrides=["dataset=human_annotations"],
        )
    assert cfg.dataset.type == "pet_annotation.human"
