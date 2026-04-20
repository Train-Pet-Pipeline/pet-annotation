"""Adapters that convert store rows to pet_schema annotation objects.

These functions perform the final mapping between the SQLite row dataclasses
(VisionAnnotationRow, AudioAnnotationRow) and the canonical pet_schema models
(VisionAnnotation, AudioAnnotation).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import pet_schema
from pet_schema.models import PetFeederEvent

from pet_annotation.store import AudioAnnotationRow, VisionAnnotationRow


def vision_row_to_annotation(row: VisionAnnotationRow) -> pet_schema.VisionAnnotation:
    """Convert a VisionAnnotationRow to a pet_schema.VisionAnnotation.

    Fields are mapped as follows:
    - ``annotation_id`` → ``annotation_id``
    - ``frame_id``      → ``sample_id``
    - ``model_name``    → ``annotator_id``
    - ``prompt_hash``   → ``prompt_hash``
    - ``raw_response``  → ``raw_response``
    - ``parsed_output`` (or ``raw_response`` fallback) → ``parsed``
    - ``modality``      → ``"vision"`` (Literal)
    - ``annotator_type`` is always ``"vlm"`` for VLM-generated rows.
    - ``created_at`` defaults to the current UTC time (not stored in the row).
    - ``schema_version`` defaults to ``"2.0.0"``.

    Args:
        row: A VisionAnnotationRow read from the annotations table.

    Returns:
        A pet_schema.VisionAnnotation populated from the row.
    """
    source = row.parsed_output if row.parsed_output is not None else row.raw_response
    parsed_dict = json.loads(source)
    parsed = PetFeederEvent(**parsed_dict)

    return pet_schema.VisionAnnotation(
        annotation_id=row.annotation_id,
        sample_id=row.frame_id,
        annotator_type="vlm",
        annotator_id=row.model_name,
        modality="vision",
        created_at=datetime.now(UTC),
        schema_version="2.0.0",
        raw_response=row.raw_response,
        parsed=parsed,
        prompt_hash=row.prompt_hash,
    )


def audio_row_to_annotation(row: AudioAnnotationRow) -> pet_schema.AudioAnnotation:
    """Convert an AudioAnnotationRow to a pet_schema.AudioAnnotation.

    Fields are mapped directly; ``class_probs`` is JSON-decoded from the stored
    TEXT column and ``logits`` is JSON-decoded when not None.

    Args:
        row: An AudioAnnotationRow read from the audio_annotations table.

    Returns:
        A pet_schema.AudioAnnotation populated from the row.
    """
    class_probs: dict[str, float] = json.loads(row.class_probs)
    logits: list[float] | None = json.loads(row.logits) if row.logits is not None else None

    return pet_schema.AudioAnnotation(
        annotation_id=row.annotation_id,
        sample_id=row.sample_id,
        annotator_type=row.annotator_type,  # type: ignore[arg-type]
        annotator_id=row.annotator_id,
        modality="audio",
        created_at=datetime.now(UTC),
        schema_version=row.schema_version,
        predicted_class=row.predicted_class,
        class_probs=class_probs,
        logits=logits,
    )
