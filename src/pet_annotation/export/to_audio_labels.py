"""Export audio classification labels for the audio CNN training pipeline.

Output format: JSONL (one JSON object per line), where each object has:

    {
        "sample_id":    "<sample identifier>",
        "storage_uri":  "local://<sample_id>",
        "label":        "<predicted_class>",
        "class_probs":  {"<class>": <prob>, ...},
        "annotator_id": "<annotator_id>"
    }

Schema decisions:
- ``audio_annotations`` has **no** ``review_status`` column → all rows exported.
- ``audio_annotations`` has **no** ``storage_uri`` column → derived as
  ``local://<sample_id>``.  Update this logic when a storage_uri column is added
  to the migration.
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
    """Export all audio annotation rows to a JSONL file.

    Each line in the output file is a JSON object with keys:
    ``sample_id``, ``storage_uri``, ``label``, ``class_probs``, ``annotator_id``.

    Because ``audio_annotations`` has no ``review_status`` column, all rows are
    exported (no approved-only filter).  This decision is documented in the module
    docstring above.

    Args:
        store: An initialised :class:`AnnotationStore`.
        output_path: Destination file path (created along with parent dirs).
        limit: If given, cap the number of rows fetched from the DB.

    Returns:
        Number of JSONL lines written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = store.fetch_audio_annotations(limit=limit)
    count = 0

    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            sample_id = row["sample_id"]
            obj = {
                "sample_id": sample_id,
                "storage_uri": f"local://{sample_id}",
                "label": row["predicted_class"],
                "class_probs": json.loads(row["class_probs"]) if row["class_probs"] else {},
                "annotator_id": row["annotator_id"],
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d audio labels to %s", count, output_path)
    return count
