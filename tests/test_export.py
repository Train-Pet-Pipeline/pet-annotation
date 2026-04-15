"""Tests for export modules."""
from __future__ import annotations

import json
import uuid
from unittest.mock import patch

from pet_annotation.export.to_dpo_pairs import export_dpo_pairs
from pet_annotation.export.to_sharegpt import export_sharegpt
from pet_annotation.store import AnnotationRecord, AnnotationStore

VALID_RAW = (
    '{"schema_version":"1.0","pet_present":false,"pet_count":0,"pet":null,'
    '"bowl":{"food_fill_ratio":0.5,"water_fill_ratio":null,"food_type_visible":"dry"},'
    '"scene":{"lighting":"bright","image_quality":"clear","confidence_overall":0.9},'
    '"narrative":"Empty bowl."}'
)


def _insert_frame(conn, fid):
    """Insert a minimal frame row."""
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root, annotation_status) "
        "VALUES (?,?,?,?,?,?)",
        (fid, "v1", "selfshot", "frames/001.jpg", "/data", "approved"),
    )
    conn.commit()


class TestExportShareGPT:
    @patch(
        "pet_annotation.export.to_sharegpt.render_prompt",
        return_value=("sys prompt", "usr prompt"),
    )
    def test_exports_approved_annotations(self, mock_render, db_conn, tmp_path):
        """Exports approved annotations to ShareGPT JSONL format."""
        store = AnnotationStore(conn=db_conn)
        _insert_frame(db_conn, "f1")
        store.insert_annotation(AnnotationRecord(
            annotation_id=str(uuid.uuid4()), frame_id="f1",
            model_name="primary", prompt_hash="h1",
            raw_response=VALID_RAW, parsed_output=VALID_RAW,
            schema_valid=True, confidence_overall=0.9,
            review_status="approved",
        ))

        out = tmp_path / "sft.jsonl"
        count = export_sharegpt(store, out, schema_version="1.0")
        assert count == 1

        line = json.loads(out.read_text().strip())
        assert line["conversations"][0]["from"] == "system"
        assert line["conversations"][1]["from"] == "human"
        assert line["conversations"][2]["from"] == "gpt"
        assert "frames/001.jpg" in line["images"]


class TestExportDPO:
    @patch("pet_annotation.export.to_dpo_pairs.render_prompt", return_value=("sys", "usr"))
    def test_exports_dpo_pairs(self, mock_render, tmp_path):
        """Exports DPO pairs to JSONL format."""
        pairs = [{
            "chosen": {"narrative": "A", "scene": {"confidence_overall": 0.9}},
            "rejected": {"narrative": "B", "scene": {"confidence_overall": 0.7}},
            "metadata": {"pair_source": "model_comparison"},
            "frame_path": "frames/001.jpg",
        }]

        out = tmp_path / "dpo.jsonl"
        count = export_dpo_pairs(pairs, out)
        assert count == 1

        line = json.loads(out.read_text().strip())
        assert line["chosen"][1]["role"] == "assistant"
        assert line["rejected"][1]["role"] == "assistant"
        assert "frames/001.jpg" in line["images"]
