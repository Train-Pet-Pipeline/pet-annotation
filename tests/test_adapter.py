"""Tests for pet_annotation.adapter — VisionAnnotationRow / AudioAnnotationRow round-trips."""
from __future__ import annotations

import json

import pet_schema
import pytest

from pet_annotation.adapter import audio_row_to_annotation, vision_row_to_annotation
from pet_annotation.store import AudioAnnotationRow, VisionAnnotationRow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PET_FEEDER_EVENT = {
    "schema_version": "1.0",
    "pet_present": True,
    "pet_count": 1,
    "pet": {
        "species": "cat",
        "breed_estimate": "shorthair",
        "id_tag": "cat-001",
        "id_confidence": 0.9,
        "action": {
            "primary": "eating",
            "distribution": {
                "eating": 0.8,
                "drinking": 0.0,
                "sniffing_only": 0.1,
                "leaving_bowl": 0.0,
                "sitting_idle": 0.05,
                "other": 0.05,
            },
        },
        "eating_metrics": {
            "speed": {"fast": 0.3, "normal": 0.5, "slow": 0.2},
            "engagement": 0.85,
            "abandoned_midway": 0.1,
        },
        "mood": {"alertness": 0.7, "anxiety": 0.1, "engagement": 0.8},
        "body_signals": {"posture": "relaxed", "ear_position": "forward"},
        "anomaly_signals": {
            "vomit_gesture": 0.0,
            "food_rejection": 0.05,
            "excessive_sniffing": 0.0,
            "lethargy": 0.0,
            "aggression": 0.0,
        },
    },
    "bowl": {"food_type_visible": "dry", "food_fill_ratio": 0.5, "water_fill_ratio": 0.3},
    "scene": {"lighting": "bright", "image_quality": "clear", "confidence_overall": 0.9},
    "narrative": "Cat eating dry food from bowl.",
}


def _make_vision_row(**kwargs: object) -> VisionAnnotationRow:
    """Build a minimal VisionAnnotationRow for tests."""
    defaults: dict[str, object] = {
        "annotation_id": "ann-v1",
        "frame_id": "frame-001",
        "model_name": "gpt-4o",
        "prompt_hash": "abc123",
        "raw_response": json.dumps(_VALID_PET_FEEDER_EVENT),
        "schema_valid": 1,
        "parsed_output": json.dumps(_VALID_PET_FEEDER_EVENT),
        "validation_errors": None,
        "confidence_overall": 0.9,
        "review_status": "pending",
        "reviewer": None,
        "review_notes": None,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "api_latency_ms": 300,
        "modality": "vision",
        "storage_uri": "gs://bucket/frames/frame-001.jpg",
    }
    defaults.update(kwargs)
    return VisionAnnotationRow(**defaults)  # type: ignore[arg-type]


def _make_audio_row(**kwargs: object) -> AudioAnnotationRow:
    """Build a minimal AudioAnnotationRow for tests."""
    defaults: dict[str, object] = {
        "annotation_id": "ann-a1",
        "sample_id": "sample-001",
        "annotator_type": "cnn",
        "annotator_id": "cnn-v2",
        "predicted_class": "bark",
        "class_probs": json.dumps({"bark": 0.9, "silence": 0.1}),
        "modality": "audio",
        "schema_version": "2.0.0",
        "logits": json.dumps([0.1, 0.2, 0.7]),
    }
    defaults.update(kwargs)
    return AudioAnnotationRow(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests — vision
# ---------------------------------------------------------------------------


def test_vision_row_round_trip() -> None:
    """vision_row_to_annotation returns a pet_schema.VisionAnnotation."""
    row = _make_vision_row()
    result = vision_row_to_annotation(row)
    assert isinstance(result, pet_schema.VisionAnnotation)
    assert result.annotation_id == "ann-v1"
    assert result.sample_id == "frame-001"
    assert result.annotator_id == "gpt-4o"
    assert result.prompt_hash == "abc123"
    assert result.raw_response == row.raw_response
    assert result.modality == "vision"


def test_vision_row_round_trip_annotator_type() -> None:
    """annotator_type defaults to 'vlm' for vision rows."""
    row = _make_vision_row()
    result = vision_row_to_annotation(row)
    assert result.annotator_type == "vlm"


def test_vision_row_round_trip_parsed_output() -> None:
    """parsed field is populated from parsed_output JSON when valid."""
    row = _make_vision_row()
    result = vision_row_to_annotation(row)
    assert result.parsed is not None
    assert result.parsed.pet_present is True


def test_vision_row_round_trip_no_parsed_output() -> None:
    """parsed falls back to raw_response when parsed_output is None."""
    row = _make_vision_row(parsed_output=None)
    result = vision_row_to_annotation(row)
    assert result.parsed is not None


def test_vision_row_round_trip_schema_version() -> None:
    """schema_version is carried through correctly."""
    row = _make_vision_row()
    result = vision_row_to_annotation(row)
    assert isinstance(result.schema_version, str)
    assert len(result.schema_version) > 0


# ---------------------------------------------------------------------------
# Tests — audio
# ---------------------------------------------------------------------------


def test_audio_row_round_trip() -> None:
    """audio_row_to_annotation returns a pet_schema.AudioAnnotation."""
    row = _make_audio_row()
    result = audio_row_to_annotation(row)
    assert isinstance(result, pet_schema.AudioAnnotation)
    assert result.annotation_id == "ann-a1"
    assert result.sample_id == "sample-001"
    assert result.annotator_type == "cnn"
    assert result.annotator_id == "cnn-v2"
    assert result.predicted_class == "bark"
    assert result.modality == "audio"


def test_audio_row_round_trip_class_probs() -> None:
    """class_probs JSON string is parsed to a dict."""
    row = _make_audio_row()
    result = audio_row_to_annotation(row)
    assert isinstance(result.class_probs, dict)
    assert result.class_probs["bark"] == pytest.approx(0.9)


def test_audio_row_round_trip_logits() -> None:
    """logits JSON string is parsed to a list of floats."""
    row = _make_audio_row()
    result = audio_row_to_annotation(row)
    assert isinstance(result.logits, list)
    assert len(result.logits) == 3


def test_audio_row_round_trip_logits_none() -> None:
    """logits=None is passed through as None."""
    row = _make_audio_row(logits=None)
    result = audio_row_to_annotation(row)
    assert result.logits is None


def test_audio_row_round_trip_schema_version() -> None:
    """schema_version is carried through."""
    row = _make_audio_row()
    result = audio_row_to_annotation(row)
    assert result.schema_version == "2.0.0"
