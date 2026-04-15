"""Export approved annotations to ShareGPT JSONL format for SFT training."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pet_schema.renderer import render_prompt

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def export_sharegpt(
    store: AnnotationStore,
    output_path: Path,
    schema_version: str = "1.0",
    data_root: str | None = None,
) -> int:
    """Export approved annotations to ShareGPT JSONL.

    Args:
        store: AnnotationStore instance.
        output_path: Path to write the JSONL file.
        schema_version: Schema version for prompt rendering.
        data_root: If provided, prepend to frame_path to create absolute
            image paths.  Required for LLaMA-Factory to locate images.

    Returns:
        Number of records exported.
    """
    system_prompt, user_prompt = render_prompt(version=schema_version)
    rows = store.fetch_approved_annotations(limit=100000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            frame_path = row["frame_path"]
            if data_root:
                frame_path = str(Path(data_root) / frame_path)

            record = {
                "id": f"sft_{count:05d}",
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": f"<image>\n{user_prompt}"},
                    {"from": "gpt", "value": row["parsed_output"] or row["raw_response"]},
                ],
                "images": [frame_path],
                "metadata": {
                    "source": row["source"],
                    "schema_version": schema_version,
                    "prompt_version": schema_version,
                    "annotator": row["model_name"],
                    "review_status": row["review_status"],
                    "frame_id": row["frame_id"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info('{"event": "export_sharegpt", "count": %d, "path": "%s"}', count, output_path)
    return count
