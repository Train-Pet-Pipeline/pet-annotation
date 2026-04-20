"""Dataset plugin exposing pet_annotation's vision annotations as VisionAnnotation iterator."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS
from pet_schema.annotations import BaseAnnotation

from pet_annotation.adapter import vision_row_to_annotation
from pet_annotation.store import VisionAnnotationRow

if TYPE_CHECKING:
    import datasets as hf_datasets


@DATASETS.register_module(name="pet_annotation.vision_annotations", force=True)
class VisionAnnotationsDataset(BaseDataset):
    """VisionAnnotation iterator over the pet-annotation annotations table.

    `dataset_config` keys:
        db_path: str — required, path to pet-annotation sqlite file

    Registered as the flat key ``pet_annotation.vision_annotations`` in
    :data:`pet_infra.registry.DATASETS`. Lookup via ``DATASETS.module_dict``
    (mmengine's ``.get()`` parses dots as ``scope.module`` and returns None
    for flat dotted names — this matches the preflight lookup pattern in
    pet-infra v2.0.0).
    """

    def modality(self) -> Literal["vision", "audio", "sensor", "multimodal"]:
        """Return the modality handled by this dataset plugin."""
        return "vision"

    def build(self, dataset_config: dict) -> Iterable[BaseAnnotation]:
        """Yield VisionAnnotation objects from the annotations table (modality='vision')."""
        db_path = dataset_config["db_path"]
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT * FROM annotations WHERE modality = 'vision'")
            for row in cur.fetchall():
                row_dict = dict(row)
                ra = VisionAnnotationRow(
                    annotation_id=row_dict["annotation_id"],
                    frame_id=row_dict["frame_id"],
                    model_name=row_dict["model_name"],
                    prompt_hash=row_dict["prompt_hash"],
                    raw_response=row_dict["raw_response"],
                    schema_valid=row_dict["schema_valid"],
                    parsed_output=row_dict.get("parsed_output"),
                    validation_errors=row_dict.get("validation_errors"),
                    confidence_overall=row_dict.get("confidence_overall"),
                    review_status=row_dict.get("review_status", "pending"),
                    reviewer=row_dict.get("reviewer"),
                    review_notes=row_dict.get("review_notes"),
                    prompt_tokens=row_dict.get("prompt_tokens"),
                    completion_tokens=row_dict.get("completion_tokens"),
                    total_tokens=row_dict.get("total_tokens"),
                    api_latency_ms=row_dict.get("api_latency_ms"),
                    modality=row_dict.get("modality", "vision"),
                    storage_uri=row_dict.get("storage_uri"),
                )
                yield vision_row_to_annotation(ra)
        finally:
            conn.close()

    def to_hf_dataset(self, dataset_config: dict) -> hf_datasets.Dataset:
        """Return a HuggingFace Dataset materialised from :meth:`build`."""
        import datasets as hf_datasets

        records = [s.model_dump(mode="json") for s in self.build(dataset_config)]
        return hf_datasets.Dataset.from_list(records)
