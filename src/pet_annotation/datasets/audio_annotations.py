"""Dataset plugin exposing pet_annotation's audio annotations as AudioAnnotation iterator."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import pet_schema
from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS
from pet_schema.annotations import BaseAnnotation

from pet_annotation.adapter import audio_row_to_annotation
from pet_annotation.store import AudioAnnotationRow

if TYPE_CHECKING:
    import datasets as hf_datasets


@DATASETS.register_module(name="pet_annotation.audio_annotations", force=True)
class AudioAnnotationsDataset(BaseDataset):
    """AudioAnnotation iterator over the pet-annotation audio_annotations table.

    `dataset_config` keys:
        db_path: str — required, path to pet-annotation sqlite file

    Registered as the flat key ``pet_annotation.audio_annotations`` in
    :data:`pet_infra.registry.DATASETS`. Lookup via ``DATASETS.module_dict``
    (mmengine's ``.get()`` parses dots as ``scope.module`` and returns None
    for flat dotted names — this matches the preflight lookup pattern in
    pet-infra v2.0.0).
    """

    def modality(self) -> Literal["vision", "audio", "sensor", "multimodal"]:
        """Return the modality handled by this dataset plugin."""
        return "audio"

    def build(self, dataset_config: dict) -> Iterable[BaseAnnotation]:
        """Yield AudioAnnotation objects from the audio_annotations table.

        All rows in audio_annotations are audio by CHECK constraint.
        """
        db_path = dataset_config["db_path"]
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT * FROM audio_annotations")
            for row in cur.fetchall():
                row_dict = dict(row)
                ra = AudioAnnotationRow(
                    annotation_id=row_dict["annotation_id"],
                    sample_id=row_dict["sample_id"],
                    annotator_type=row_dict["annotator_type"],
                    annotator_id=row_dict["annotator_id"],
                    predicted_class=row_dict["predicted_class"],
                    class_probs=row_dict["class_probs"],
                    modality=row_dict.get("modality", "audio"),
                    schema_version=row_dict.get("schema_version") or pet_schema.SCHEMA_VERSION,
                    logits=row_dict.get("logits"),
                )
                yield audio_row_to_annotation(ra)
        finally:
            conn.close()

    def to_hf_dataset(self, dataset_config: dict) -> hf_datasets.Dataset:
        """Return a HuggingFace Dataset materialised from :meth:`build`."""
        import datasets as hf_datasets

        records = [s.model_dump(mode="json") for s in self.build(dataset_config)]
        return hf_datasets.Dataset.from_list(records)
