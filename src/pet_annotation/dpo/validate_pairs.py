"""DPO pair validation — enforces 5 rules from DEVELOPMENT_GUIDE."""

from __future__ import annotations

import json
import logging

from pet_schema.validator import validate_output

logger = logging.getLogger(__name__)


def validate_pair(
    chosen: dict,
    rejected: dict,
    pair_meta: dict,
    schema_version: str = "1.0",
) -> tuple[bool, list[str]]:
    """Validate a DPO training pair. Any failure discards the pair.

    Rules:
        1. chosen passes schema validation.
        2. rejected passes schema validation (format valid, content wrong).
        3. chosen and rejected narratives are not identical.
        4. User feedback pairs: rejected must have inference_id.
        5. chosen.confidence_overall >= rejected.confidence_overall.

    Args:
        chosen: The preferred output dict.
        rejected: The dispreferred output dict.
        pair_meta: Metadata dict with at least 'pair_source'.
        schema_version: Schema version for validation.

    Returns:
        Tuple of (is_valid, list_of_error_strings).
    """
    errors: list[str] = []

    # Rule 1: chosen passes schema
    chosen_result = validate_output(json.dumps(chosen), version=schema_version)
    if not chosen_result.valid:
        errors.append(f"chosen schema 验证失败: {chosen_result.errors}")

    # Rule 2: rejected passes schema
    rejected_result = validate_output(json.dumps(rejected), version=schema_version)
    if not rejected_result.valid:
        errors.append(f"rejected schema 验证失败: {rejected_result.errors}")

    # Rule 3: narratives must differ
    chosen_narrative = chosen.get("narrative", "")
    rejected_narrative = rejected.get("narrative", "")
    if chosen_narrative == rejected_narrative:
        errors.append("chosen 和 rejected 的 narrative 完全相同")

    # Rule 4: user feedback requires inference_id
    if pair_meta.get("pair_source") == "user_feedback":
        if "inference_id" not in pair_meta:
            errors.append("user_feedback pair 缺少 inference_id 追踪")

    # Rule 5: chosen confidence >= rejected confidence
    chosen_conf = chosen.get("scene", {}).get("confidence_overall", 0)
    rejected_conf = rejected.get("scene", {}).get("confidence_overall", 0)
    if chosen_conf < rejected_conf:
        errors.append(f"chosen confidence ({chosen_conf}) < rejected confidence ({rejected_conf})")

    return len(errors) == 0, errors
