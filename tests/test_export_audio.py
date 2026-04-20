"""Tests for export_audio_labels (B12).

Schema decision notes:
- audio_annotations has NO review_status column → export ALL rows.
- audio_annotations has NO storage_uri column → derive as ``local://<sample_id>``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pet_annotation.export.to_audio_labels import export_audio_labels
from pet_annotation.store import AnnotationStore, AudioAnnotationRow


def _make_store(db_conn: sqlite3.Connection) -> AnnotationStore:
    """Return an AnnotationStore wrapping the provided in-memory connection."""
    return AnnotationStore(conn=db_conn)


def _make_audio_row(
    annotation_id: str,
    sample_id: str,
    predicted_class: str = "bark",
    class_probs: str = '{"bark": 0.9, "meow": 0.1}',
    annotator_id: str = "audio_cnn_v1",
) -> AudioAnnotationRow:
    return AudioAnnotationRow(
        annotation_id=annotation_id,
        sample_id=sample_id,
        annotator_type="cnn",
        annotator_id=annotator_id,
        predicted_class=predicted_class,
        class_probs=class_probs,
    )


class TestExportAudioLabelsWritesJsonl:
    def test_writes_two_lines(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """Two inserted rows → two JSONL lines written, returns 2."""
        store = _make_store(db_conn)
        store.insert_annotation(_make_audio_row("ann-1", "sample-001"))
        store.insert_annotation(_make_audio_row("ann-2", "sample-002"))

        out = tmp_path / "audio_labels.jsonl"
        count = export_audio_labels(store, out)

        assert count == 2
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "sample_id" in obj
            assert "storage_uri" in obj
            assert "label" in obj
            assert "class_probs" in obj
            assert "annotator_id" in obj


class TestExportAudioLabelsEmptyDb:
    def test_empty_db_returns_zero(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """Empty store → 0 lines written, returns 0."""
        store = _make_store(db_conn)
        out = tmp_path / "empty.jsonl"
        count = export_audio_labels(store, out)

        assert count == 0
        assert out.exists()
        assert out.read_text() == ""


class TestExportAudioLabelsClassProbsParsed:
    def test_class_probs_is_dict(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """class_probs stored as JSON string in DB must be a dict in output."""
        store = _make_store(db_conn)
        store.insert_annotation(
            _make_audio_row("ann-1", "sample-001", class_probs='{"bark": 0.8, "meow": 0.2}')
        )

        out = tmp_path / "labels.jsonl"
        export_audio_labels(store, out)

        obj = json.loads(out.read_text().strip())
        assert isinstance(obj["class_probs"], dict)
        assert obj["class_probs"] == {"bark": 0.8, "meow": 0.2}


class TestExportAudioLabelsStorageUriFormat:
    def test_storage_uri_derived_from_sample_id(
        self, db_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """storage_uri is derived as ``local://<sample_id>`` (no storage_uri column in DB)."""
        store = _make_store(db_conn)
        store.insert_annotation(_make_audio_row("ann-1", "sample-abc"))

        out = tmp_path / "labels.jsonl"
        export_audio_labels(store, out)

        obj = json.loads(out.read_text().strip())
        assert obj["storage_uri"] == "local://sample-abc"
        assert obj["sample_id"] == "sample-abc"


class TestExportAudioLabelsLimitParam:
    def test_limit_restricts_output(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """limit=1 → only 1 row written even if 3 are available."""
        store = _make_store(db_conn)
        for i in range(3):
            store.insert_annotation(_make_audio_row(f"ann-{i}", f"sample-{i:03d}"))

        out = tmp_path / "limited.jsonl"
        count = export_audio_labels(store, out, limit=1)

        assert count == 1
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1


class TestExportAudioLabelsParentDirCreated:
    def test_creates_parent_dirs(self, db_conn: sqlite3.Connection, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        store = _make_store(db_conn)
        store.insert_annotation(_make_audio_row("ann-1", "sample-001"))

        out = tmp_path / "nested" / "deep" / "labels.jsonl"
        export_audio_labels(store, out)

        assert out.exists()


class TestExportAudioLabelsMalformedClassProbs:
    def test_export_audio_labels_raises_on_malformed_class_probs(
        self, db_conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """Malformed class_probs JSON raises ValueError with sample_id context."""
        store = _make_store(db_conn)
        store._conn.execute(
            "INSERT INTO audio_annotations (annotation_id, sample_id, annotator_type, "
            "annotator_id, predicted_class, class_probs, modality, schema_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ann-bad",
                "sample-bad",
                "cnn",
                "audio_cnn_v1",
                "bark",
                "{not valid}",
                "audio",
                "2.0.0",
            ),
        )
        store._conn.commit()
        out = tmp_path / "audio.jsonl"
        with pytest.raises(ValueError, match="sample-bad"):
            export_audio_labels(store, out)
        store.close()
