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
    data_root: str | None = None,
) -> int:
    """Export DPO pairs to JSONL.

    Each pair includes the frame image path from pair metadata.

    Args:
        pairs: List of validated pair dicts from generate_pairs.
        output_path: Path to write the JSONL file.
        schema_version: Schema version for prompt rendering.
        data_root: If provided, prepend to frame_path to create absolute
            image paths.

    Returns:
        Number of pairs exported.
    """
    system_prompt, user_prompt = render_prompt(version=schema_version)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            frame_path = pair.get("frame_path", "")
            if data_root and frame_path:
                frame_path = str(Path(data_root) / frame_path)

            # LLaMA-Factory ShareGPT DPO format:
            # - conversations: prompt turns (system + user)
            # - chosen/rejected: single assistant response dict
            record = {
                "id": f"dpo_{count:05d}",
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": f"<image>\n{user_prompt}"},
                ],
                "chosen": {
                    "from": "gpt",
                    "value": json.dumps(pair["chosen"], ensure_ascii=False),
                },
                "rejected": {
                    "from": "gpt",
                    "value": json.dumps(pair["rejected"], ensure_ascii=False),
                },
                "images": [frame_path],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info('{"event": "export_dpo", "count": %d, "path": "%s"}', count, output_path)
    return count
