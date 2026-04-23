"""SFT and DPO export functions for the 4-paradigm annotation store.

Exports annotation data to JSONL format suitable for SFT/DPO training.

SFT format (one line per done annotation):
    {"sample_id": str, "annotator_id": str, "annotator_type": str,
     "input": str, "output": str, "storage_uri": str | null}

DPO format (one line per chosen/rejected pair, from LLM annotations for same target):
    {"sample_id": str, "chosen": str, "rejected": str,
     "chosen_annotator_id": str, "rejected_annotator_id": str}

DPO pairs are derived from targets annotated by 2+ LLM annotators; the annotation
with higher schema validation confidence is chosen. If only one annotation per target,
that target is skipped for DPO (no rejected to pair against).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def _iter_done_llm_rows(store: AnnotationStore) -> Iterable[dict]:
    """Yield all rows from llm_annotations joined to done annotation_targets.

    Yields dicts with keys: annotation_id, target_id, annotator_id, raw_response,
    parsed_output, storage_uri, schema_version, modality.
    """
    cur = store._conn.execute(
        "SELECT la.annotation_id, la.target_id, la.annotator_id, "
        "la.raw_response, la.parsed_output, la.storage_uri, "
        "la.schema_version, la.modality "
        "FROM llm_annotations la "
        "JOIN annotation_targets at_ ON "
        "la.target_id = at_.target_id AND la.annotator_id = at_.annotator_id "
        "WHERE at_.state = 'done' AND at_.annotator_type = 'llm'"
    )
    for row in cur.fetchall():
        yield {
            "annotation_id": row[0],
            "target_id": row[1],
            "annotator_id": row[2],
            "raw_response": row[3],
            "parsed_output": json.loads(row[4]) if row[4] else {},
            "storage_uri": row[5],
            "schema_version": row[6],
            "modality": row[7],
        }


def _iter_done_classifier_rows(store: AnnotationStore) -> Iterable[dict]:
    """Yield all rows from classifier_annotations joined to done annotation_targets."""
    cur = store._conn.execute(
        "SELECT ca.annotation_id, ca.target_id, ca.annotator_id, "
        "ca.predicted_class, ca.class_probs, ca.storage_uri "
        "FROM classifier_annotations ca "
        "JOIN annotation_targets at_ ON "
        "ca.target_id = at_.target_id AND ca.annotator_id = at_.annotator_id "
        "WHERE at_.state = 'done' AND at_.annotator_type = 'classifier'"
    )
    for row in cur.fetchall():
        yield {
            "annotation_id": row[0],
            "target_id": row[1],
            "annotator_id": row[2],
            "predicted_class": row[3],
            "class_probs": json.loads(row[4]) if row[4] else {},
            "storage_uri": row[5],
        }


def _iter_done_rule_rows(store: AnnotationStore) -> Iterable[dict]:
    """Yield all rows from rule_annotations joined to done annotation_targets."""
    cur = store._conn.execute(
        "SELECT ra.annotation_id, ra.target_id, ra.annotator_id, "
        "ra.rule_id, ra.rule_output, ra.storage_uri "
        "FROM rule_annotations ra "
        "JOIN annotation_targets at_ ON "
        "ra.target_id = at_.target_id AND ra.annotator_id = at_.annotator_id "
        "WHERE at_.state = 'done' AND at_.annotator_type = 'rule'"
    )
    for row in cur.fetchall():
        yield {
            "annotation_id": row[0],
            "target_id": row[1],
            "annotator_id": row[2],
            "rule_id": row[3],
            "rule_output": json.loads(row[4]) if row[4] else {},
            "storage_uri": row[5],
        }


def _iter_done_human_rows(store: AnnotationStore) -> Iterable[dict]:
    """Yield all rows from human_annotations joined to done annotation_targets."""
    cur = store._conn.execute(
        "SELECT ha.annotation_id, ha.target_id, ha.annotator_id, "
        "ha.reviewer, ha.decision, ha.notes, ha.storage_uri "
        "FROM human_annotations ha "
        "JOIN annotation_targets at_ ON "
        "ha.target_id = at_.target_id AND ha.annotator_id = at_.annotator_id "
        "WHERE at_.state = 'done' AND at_.annotator_type = 'human'"
    )
    for row in cur.fetchall():
        yield {
            "annotation_id": row[0],
            "target_id": row[1],
            "annotator_id": row[2],
            "reviewer": row[3],
            "decision": row[4],
            "notes": row[5],
            "storage_uri": row[6],
        }


def to_sft_samples(
    store: AnnotationStore,
    annotator_type: str = "llm",
    output_path: Path | None = None,
) -> list[dict]:
    """Export done annotations to SFT JSONL format.

    Each sample is a dict with keys: sample_id, annotator_id, annotator_type,
    input (= storage_uri or target_id), output (= annotation content as JSON string).

    Args:
        store: Initialised AnnotationStore.
        annotator_type: Which paradigm table to read from ('llm', 'classifier',
            'rule', 'human'). Defaults to 'llm'.
        output_path: If provided, write JSONL to this path. Otherwise, return
            list of sample dicts without writing.

    Returns:
        List of sample dicts (one per done annotation).
    """
    samples: list[dict] = []

    if annotator_type == "llm":
        for row in _iter_done_llm_rows(store):
            samples.append({
                "sample_id": row["target_id"],
                "annotator_id": row["annotator_id"],
                "annotator_type": "llm",
                "input": row["storage_uri"] or row["target_id"],
                "output": row["raw_response"],
                "storage_uri": row["storage_uri"],
            })

    elif annotator_type == "classifier":
        for row in _iter_done_classifier_rows(store):
            samples.append({
                "sample_id": row["target_id"],
                "annotator_id": row["annotator_id"],
                "annotator_type": "classifier",
                "input": row["storage_uri"] or row["target_id"],
                "output": json.dumps({
                    "predicted_class": row["predicted_class"],
                    "class_probs": row["class_probs"],
                }),
                "storage_uri": row["storage_uri"],
            })

    elif annotator_type == "rule":
        for row in _iter_done_rule_rows(store):
            samples.append({
                "sample_id": row["target_id"],
                "annotator_id": row["annotator_id"],
                "annotator_type": "rule",
                "input": row["storage_uri"] or row["target_id"],
                "output": json.dumps(row["rule_output"]),
                "storage_uri": row["storage_uri"],
            })

    elif annotator_type == "human":
        for row in _iter_done_human_rows(store):
            samples.append({
                "sample_id": row["target_id"],
                "annotator_id": row["annotator_id"],
                "annotator_type": "human",
                "input": row["storage_uri"] or row["target_id"],
                "output": json.dumps({
                    "decision": row["decision"],
                    "reviewer": row["reviewer"],
                    "notes": row["notes"],
                }),
                "storage_uri": row["storage_uri"],
            })

    else:
        raise ValueError(
            f"Unknown annotator_type '{annotator_type}'. "
            "Must be one of: llm, classifier, rule, human."
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Exported %d SFT samples to %s", len(samples), output_path)

    return samples


def to_dpo_pairs(
    store: AnnotationStore,
    annotator_type: str = "llm",
    output_path: Path | None = None,
) -> list[dict]:
    """Export done annotations to DPO JSONL format.

    For LLM paradigm: groups annotations by target_id; if a target has 2+
    annotations from different annotators, the annotation with higher
    confidence_overall becomes 'chosen' and the other 'rejected'.

    For classifier/rule/human: no natural chosen/rejected pairing; emits
    each annotation as a self-paired sample with chosen == rejected marked
    with a note (single-annotation target), so downstream pipelines can filter.

    Args:
        store: Initialised AnnotationStore.
        annotator_type: Which paradigm table to read from.
        output_path: If provided, write JSONL to this path.

    Returns:
        List of pair dicts.
    """
    pairs: list[dict] = []

    if annotator_type == "llm":
        # Group by target_id
        target_rows: dict[str, list[dict]] = {}
        for row in _iter_done_llm_rows(store):
            target_rows.setdefault(row["target_id"], []).append(row)

        for target_id, rows in target_rows.items():
            if len(rows) < 2:
                continue  # Need at least 2 annotations to form a pair

            # Sort by confidence_overall (best-effort extraction from parsed_output)
            def _confidence(r: dict) -> float:
                try:
                    return float(
                        r["parsed_output"].get("scene", {}).get("confidence_overall", 0)
                    )
                except (TypeError, ValueError):
                    return 0.0

            sorted_rows = sorted(rows, key=_confidence, reverse=True)
            chosen = sorted_rows[0]
            rejected = sorted_rows[-1]

            pairs.append({
                "sample_id": target_id,
                "chosen": chosen["raw_response"],
                "rejected": rejected["raw_response"],
                "chosen_annotator_id": chosen["annotator_id"],
                "rejected_annotator_id": rejected["annotator_id"],
                "storage_uri": chosen["storage_uri"],
            })

    else:
        # For non-LLM paradigms, emit single-annotation samples as DPO-ready format.
        # Chosen and rejected are identical; downstream can augment/filter.
        samples = to_sft_samples(store, annotator_type=annotator_type)
        for s in samples:
            pairs.append({
                "sample_id": s["sample_id"],
                "chosen": s["output"],
                "rejected": s["output"],
                "chosen_annotator_id": s["annotator_id"],
                "rejected_annotator_id": s["annotator_id"],
                "storage_uri": s["storage_uri"],
            })

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("Exported %d DPO pairs to %s", len(pairs), output_path)

    return pairs
