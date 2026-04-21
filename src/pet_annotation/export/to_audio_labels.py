"""Export audio classification labels for the audio CNN training pipeline.

v2.0.0: Audio export is migrated to use the classifier_annotations table.
Use fetch_classifier_by_target() to query audio-modality classifier rows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def export_audio_labels(
    store: AnnotationStore,
    output_path: Path,
    *,
    limit: int | None = None,
) -> int:
    """Export classifier-paradigm audio annotation rows to a JSONL file.

    Queries the classifier_annotations table for rows with modality='audio'.
    Each output line has keys: sample_id, storage_uri, label, class_probs, annotator_id.

    Args:
        store: An initialised AnnotationStore (4-paradigm schema).
        output_path: Destination file path.
        limit: If given, cap the number of rows fetched.

    Returns:
        Number of JSONL lines written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = (
        "SELECT annotation_id, target_id, annotator_id, predicted_class, class_probs "
        "FROM classifier_annotations WHERE modality = 'audio'"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    rows = store._conn.execute(query).fetchall()
    count = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            sample_id = row[1]
            class_probs = json.loads(row[4]) if row[4] else {}
            obj = {
                "sample_id": sample_id,
                "storage_uri": f"local://{sample_id}",
                "label": row[3],
                "class_probs": class_probs,
                "annotator_id": row[2],
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d audio labels to %s", count, output_path)
    return count
