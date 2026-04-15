"""Export validated DPO pairs to JSONL format for DPO training."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pet_schema.renderer import render_prompt

logger = logging.getLogger(__name__)


def export_dpo_pairs(
    pairs: list[dict],
    output_path: Path,
    schema_version: str = "1.0",
) -> int:
    """Export DPO pairs to JSONL.

    Each pair includes the frame image path from pair metadata.

    Args:
        pairs: List of validated pair dicts from generate_pairs.
        output_path: Path to write the JSONL file.
        schema_version: Schema version for prompt rendering.

    Returns:
        Number of pairs exported.
    """
    system_prompt, user_prompt = render_prompt(version=schema_version)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "id": f"dpo_{count:05d}",
                "system": system_prompt,
                "prompt": f"<image>\n{user_prompt}",
                "images": [pair.get("frame_path", "")],
                "chosen": [
                    {"role": "user", "content": f"<image>\n{user_prompt}"},
                    {
                        "role": "assistant",
                        "content": json.dumps(pair["chosen"], ensure_ascii=False),
                    },
                ],
                "rejected": [
                    {"role": "user", "content": f"<image>\n{user_prompt}"},
                    {
                        "role": "assistant",
                        "content": json.dumps(pair["rejected"], ensure_ascii=False),
                    },
                ],
                "metadata": pair["metadata"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info('{"event": "export_dpo", "count": %d, "path": "%s"}', count, output_path)
    return count
