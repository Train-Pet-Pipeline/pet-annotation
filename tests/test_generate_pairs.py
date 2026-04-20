"""Tests for DPO pair generation."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from pet_annotation.dpo.generate_pairs import generate_cross_model_pairs
from pet_annotation.store import AnnotationRecord, AnnotationStore, ComparisonRecord


def _insert_frame(conn, fid):
    """Insert a minimal frame row."""
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root, annotation_status) "
        "VALUES (?,?,?,?,?,?)",
        (fid, "v1", "selfshot", "f.jpg", "/data", "approved"),
    )
    conn.commit()


VALID_OUTPUT = (
    '{{"schema_version":"1.0","pet_present":true,"pet_count":1,'
    '"pet":{{"species":"cat","breed_estimate":"bsh","id_tag":"grey","id_confidence":0.8,'
    '"action":{{"primary":"eating","distribution":{{"eating":0.7,"drinking":0.05,'
    '"sniffing_only":0.1,"leaving_bowl":0.05,"sitting_idle":0.05,"other":0.05}}}},'
    '"eating_metrics":{{"speed":{{"fast":0.1,"normal":0.7,"slow":0.2}},"engagement":0.8,"abandoned_midway":0.1}},'
    '"mood":{{"alertness":0.3,"anxiety":0.1,"engagement":0.8}},'
    '"body_signals":{{"posture":"relaxed","ear_position":"forward"}},'
    '"anomaly_signals":{{"vomit_gesture":0.0,"food_rejection":0.0,'
    '"excessive_sniffing":0.0,"lethargy":0.0,"aggression":0.0}}}},'
    '"bowl":{{"food_fill_ratio":0.5,"water_fill_ratio":null,"food_type_visible":"dry"}},'
    '"scene":{{"lighting":"bright","image_quality":"clear","confidence_overall":{conf}}},'
    '"narrative":"{narrative}"}}'
)


def _mock_valid_result():
    """Return a mock validation result that passes."""
    r = MagicMock()
    r.valid = True
    r.errors = []
    return r


class TestGenerateCrossModelPairs:
    @patch("pet_annotation.dpo.validate_pairs.validate_output", return_value=_mock_valid_result())
    def test_generates_pair_when_primary_higher_confidence(self, mock_validate, db_conn):
        """Generates a DPO pair when primary model has higher confidence."""
        store = AnnotationStore(conn=db_conn)
        _insert_frame(db_conn, "f1")

        store.insert_annotation(
            AnnotationRecord(
                annotation_id=str(uuid.uuid4()),
                frame_id="f1",
                model_name="primary",
                prompt_hash="h1",
                raw_response=VALID_OUTPUT.format(conf="0.90", narrative="Primary annotation."),
                schema_valid=True,
                confidence_overall=0.90,
                review_status="approved",
            )
        )
        store.insert_comparison(
            ComparisonRecord(
                comparison_id=str(uuid.uuid4()),
                frame_id="f1",
                model_name="secondary",
                prompt_hash="h1",
                raw_response=VALID_OUTPUT.format(conf="0.70", narrative="Secondary annotation."),
                schema_valid=True,
                confidence_overall=0.70,
            )
        )

        pairs = generate_cross_model_pairs(store, primary_model="primary")
        assert len(pairs) == 1
        assert pairs[0]["chosen"]["scene"]["confidence_overall"] == 0.90

    @patch("pet_annotation.dpo.validate_pairs.validate_output", return_value=_mock_valid_result())
    def test_skips_when_primary_lower_confidence(self, mock_validate, db_conn):
        """Skips pair when primary model has lower confidence."""
        store = AnnotationStore(conn=db_conn)
        _insert_frame(db_conn, "f1")

        store.insert_annotation(
            AnnotationRecord(
                annotation_id=str(uuid.uuid4()),
                frame_id="f1",
                model_name="primary",
                prompt_hash="h1",
                raw_response=VALID_OUTPUT.format(conf="0.60", narrative="Primary low."),
                schema_valid=True,
                confidence_overall=0.60,
                review_status="approved",
            )
        )
        store.insert_comparison(
            ComparisonRecord(
                comparison_id=str(uuid.uuid4()),
                frame_id="f1",
                model_name="secondary",
                prompt_hash="h1",
                raw_response=VALID_OUTPUT.format(conf="0.90", narrative="Secondary high."),
                schema_valid=True,
                confidence_overall=0.90,
            )
        )

        pairs = generate_cross_model_pairs(store, primary_model="primary")
        assert len(pairs) == 0
