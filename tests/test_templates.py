"""Tests for human_review.templates module."""
import pytest

from pet_annotation.human_review.templates import template_for


def test_template_for_vision_contains_image_tag():
    result = template_for("vision")
    assert "<Image name=" in result


def test_template_for_audio_contains_audio_tag():
    result = template_for("audio")
    assert "<Audio name=" in result


def test_template_for_unknown_modality_raises():
    with pytest.raises(ValueError, match="No LS template"):
        template_for("sensor")


def test_vision_template_has_corrected_output_textarea():
    # Matches current LABELING_CONFIG in import_to_ls.py
    result = template_for("vision")
    assert "corrected_output" in result
    assert "review_decision" in result


def test_audio_template_has_correction_choices():
    result = template_for("audio")
    # From plan spec
    assert 'value="bark"' in result
    assert 'value="silence"' in result
