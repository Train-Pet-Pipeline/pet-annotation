"""Tests for VisionAnnotationsDataset plugin registration and behaviour."""
from __future__ import annotations

import json

import pet_schema
from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS

import pet_annotation.datasets.vision_annotations  # noqa: F401  (trigger registration)
from pet_annotation.store import AnnotationStore, VisionAnnotationRow

# ---------------------------------------------------------------------------
# Shared valid PetFeederEvent JSON (matches adapter test pattern)
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

_VALID_JSON = json.dumps(_VALID_PET_FEEDER_EVENT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vision_row(
    annotation_id: str = "ann-v1", frame_id: str = "frame-001"
) -> VisionAnnotationRow:
    """Build a minimal VisionAnnotationRow with valid parsed_output."""
    return VisionAnnotationRow(
        annotation_id=annotation_id,
        frame_id=frame_id,
        model_name="gpt-4o",
        prompt_hash="abc123",
        raw_response=_VALID_JSON,
        schema_valid=1,
        parsed_output=_VALID_JSON,
        modality="vision",
    )


def _make_audio_row(annotation_id: str = "ann-a1") -> dict:
    """Return kwargs for a raw-SQL audio_annotations insert."""
    return {
        "annotation_id": annotation_id,
        "sample_id": "sample-001",
        "annotator_type": "cnn",
        "annotator_id": "audio-cnn-v1",
        "modality": "audio",
        "schema_version": pet_schema.SCHEMA_VERSION,
        "predicted_class": "eating",
        "class_probs": json.dumps({"eating": 0.9, "drinking": 0.1}),
        "logits": None,
    }


_FRAMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL,
    source          TEXT NOT NULL,
    frame_path      TEXT NOT NULL,
    data_root       TEXT NOT NULL,
    timestamp_ms    INTEGER,
    species         TEXT,
    breed           TEXT,
    lighting        TEXT,
    bowl_type       TEXT,
    quality_flag    TEXT NOT NULL DEFAULT 'normal',
    blur_score      REAL,
    annotation_status TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _bootstrap_db(tmp_path):
    """Create a fresh AnnotationStore-initialised DB and return (store, db_path).

    The frames table (owned by pet-data) must exist before AnnotationStore._recover_stuck_frames
    runs, so we pre-create it via a raw connection before handing off to AnnotationStore.
    """
    import sqlite3 as _sqlite3

    db_path = tmp_path / "db.sqlite"
    # Pre-create the frames table that pet-data owns
    _conn = _sqlite3.connect(str(db_path))
    _conn.executescript(_FRAMES_SCHEMA)
    _conn.commit()
    _conn.close()
    store = AnnotationStore(db_path=db_path)
    return store, db_path


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_vision_annotations_registered():
    """Plugin must appear in DATASETS.module_dict under flat dotted key."""
    assert "pet_annotation.vision_annotations" in DATASETS.module_dict


# ---------------------------------------------------------------------------
# Class-level tests
# ---------------------------------------------------------------------------


def test_vision_annotations_is_base_dataset():
    """Plugin class must be a subclass of BaseDataset."""
    cls = DATASETS.module_dict["pet_annotation.vision_annotations"]
    ds = cls()
    assert isinstance(ds, BaseDataset)


def test_vision_annotations_modality():
    """modality() must return the string 'vision'."""
    cls = DATASETS.module_dict["pet_annotation.vision_annotations"]
    ds = cls()
    assert ds.modality() == "vision"


# ---------------------------------------------------------------------------
# build() tests
# ---------------------------------------------------------------------------


def _insert_frame(db_path, frame_id: str) -> None:
    """Insert a minimal frames row to satisfy FK constraint."""
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        INSERT OR IGNORE INTO frames (
            frame_id, video_id, source, frame_path, data_root, quality_flag, annotation_status
        ) VALUES (?,?,?,?,?,?,?)
        """,
        (frame_id, "vid-001", "youtube", f"/data/{frame_id}.jpg", "/data", "normal", "pending"),
    )
    conn.commit()
    conn.close()


def test_vision_annotations_build_yields_vision_annotation(tmp_path):
    """build() must yield pet_schema.VisionAnnotation objects."""
    store, db_path = _bootstrap_db(tmp_path)
    store.close()
    _insert_frame(db_path, "frame-001")
    store2 = AnnotationStore(db_path=db_path)
    store2.insert_annotation(_make_vision_row())
    store2.close()

    cls = DATASETS.module_dict["pet_annotation.vision_annotations"]
    ds = cls()
    result = next(iter(ds.build({"db_path": str(db_path)})))
    assert isinstance(result, pet_schema.VisionAnnotation)


def test_vision_annotations_filters_audio_rows(tmp_path):
    """build() must return only vision rows; audio rows in annotations table are excluded."""
    store, db_path = _bootstrap_db(tmp_path)
    store.close()
    _insert_frame(db_path, "frame-v1")
    store2 = AnnotationStore(db_path=db_path)
    store2.insert_annotation(_make_vision_row(annotation_id="ann-v1", frame_id="frame-v1"))
    store2.close()

    # Insert a second row into annotations directly with modality='audio' to test filtering.
    # We also need a frame row for it, but since FK is enforced we insert frame-a1 first.
    _insert_frame(db_path, "frame-a1")
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        INSERT INTO annotations (
            annotation_id, frame_id, model_name, prompt_hash,
            raw_response, parsed_output, schema_valid, modality
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        ("ann-a1", "frame-a1", "audio-model", "hash-a", _VALID_JSON, _VALID_JSON, 1, "audio"),
    )
    conn.commit()
    conn.close()

    cls = DATASETS.module_dict["pet_annotation.vision_annotations"]
    ds = cls()
    results = list(ds.build({"db_path": str(db_path)}))
    assert len(results) == 1
    assert results[0].annotation_id == "ann-v1"


def test_vision_annotations_build_empty_db(tmp_path):
    """build() over an empty annotations table must return an empty iterator."""
    store, db_path = _bootstrap_db(tmp_path)
    store.close()

    cls = DATASETS.module_dict["pet_annotation.vision_annotations"]
    ds = cls()
    assert list(ds.build({"db_path": str(db_path)})) == []
