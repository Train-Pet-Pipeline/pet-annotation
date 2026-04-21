"""DATASETS plugin: pet_annotation.human — HumanAnnotation iterator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS

if TYPE_CHECKING:
    import datasets as hf_datasets


@DATASETS.register_module(name="pet_annotation.human", force=True)
class HumanAnnotationDataset(BaseDataset):
    """Iterator over the human_annotations table.

    ``dataset_config`` keys:
        db_path: str — required, path to pet-annotation sqlite file.
    """

    def modality(self) -> Literal["vision", "audio", "sensor", "multimodal"]:
        """Return multimodal since human paradigm spans all modalities."""
        return "multimodal"

    def build(self, dataset_config: dict):
        """Yield HumanAnnotation objects from human_annotations table.

        Args:
            dataset_config: Config dict with ``db_path`` key.
        """
        from pet_annotation.store import AnnotationStore

        store = AnnotationStore(dataset_config["db_path"])
        store.init_schema()
        target_ids = [
            r[0]
            for r in store._conn.execute(
                "SELECT DISTINCT target_id FROM human_annotations"
            ).fetchall()
        ]
        for tid in target_ids:
            yield from store.fetch_human_by_target(tid)

    def to_hf_dataset(self, dataset_config: dict) -> hf_datasets.Dataset:
        """Return a HuggingFace Dataset from human_annotations.

        Args:
            dataset_config: Config dict with ``db_path`` key.
        """
        import datasets as hf_datasets

        records = [s.model_dump(mode="json") for s in self.build(dataset_config)]
        return hf_datasets.Dataset.from_list(records)
