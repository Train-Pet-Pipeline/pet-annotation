"""DPO pair generation — cross-model and user feedback pairing."""

from __future__ import annotations

import json
import logging

from pet_annotation.dpo.validate_pairs import validate_pair
from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def generate_cross_model_pairs(
    store: AnnotationStore,
    primary_model: str,
    schema_version: str = "1.0",
    *,
    modality: str = "vision",
) -> list[dict]:
    """Generate DPO pairs by comparing primary model vs secondary models.

    Only generates a pair when primary model confidence > secondary model confidence.
    Each pair is validated before inclusion.

    Args:
        store: AnnotationStore instance.
        primary_model: Name of the primary model.
        schema_version: Schema version for validation.
        modality: Modality to filter by (default ``"vision"``).  Audio DPO is
            out of scope for Phase 2 — passing ``"audio"`` raises
            ``NotImplementedError``.

    Returns:
        List of valid DPO pair dicts with chosen/rejected/metadata.

    Raises:
        NotImplementedError: If ``modality="audio"``.
        ValueError: If ``modality`` is an unrecognised value.
    """
    if modality == "audio":
        raise NotImplementedError(
            "audio DPO is out of Phase 2 scope — only vision DPO pairs are supported"
        )
    elif modality != "vision":
        raise ValueError(f"Unknown modality: {modality!r}")

    approved = store.fetch_approved_annotations(limit=100000, modality=modality)

    pairs = []

    for row in approved:
        if row["model_name"] != primary_model:
            continue
        if not row["schema_valid"]:
            continue

        frame_id = row["frame_id"]
        primary_conf = row["confidence_overall"]
        primary_output = json.loads(row["raw_response"])

        # Find comparison results for this frame
        comparisons = store.fetch_comparisons_for_frame(frame_id, modality=modality)
        for comp in comparisons:
            if not comp["schema_valid"]:
                continue
            comp_conf = comp["confidence_overall"]
            if comp_conf is None or primary_conf is None:
                continue
            if primary_conf <= comp_conf:
                continue  # Only pair when primary is better

            rejected_output = json.loads(comp["raw_response"])

            pair_meta = {
                "pair_source": "model_comparison",
                "chosen_model": primary_model,
                "rejected_model": comp["model_name"],
            }

            ok, errors = validate_pair(
                primary_output,
                rejected_output,
                pair_meta,
                schema_version=schema_version,
            )
            if ok:
                pairs.append(
                    {
                        "chosen": primary_output,
                        "rejected": rejected_output,
                        "metadata": pair_meta,
                        "frame_id": frame_id,
                        "frame_path": row["frame_path"],
                    }
                )
            else:
                logger.info(
                    '{"event": "pair_rejected", "frame_id": "%s", "errors": %s}',
                    frame_id,
                    json.dumps(errors),
                )

    logger.info('{"event": "pairs_generated", "count": %d}', len(pairs))
    return pairs


if __name__ == "__main__":
    import json as json_mod
    from pathlib import Path

    from pet_annotation.config import load_config, setup_logging

    setup_logging()
    config = load_config()
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        pairs = generate_cross_model_pairs(
            store, config.annotation.primary_model, config.annotation.schema_version
        )
        stats = {"total_pairs": len(pairs)}
        print(json_mod.dumps(stats, indent=2))
    finally:
        store.close()
