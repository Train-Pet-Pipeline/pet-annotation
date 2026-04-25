"""SFT and DPO export functions for the 4-paradigm annotation store.

Exports annotation data to JSONL format suitable for LLaMA-Factory consumption.

SFT format (LLaMA-Factory ShareGPT conversations, one line per LLM/human annotation):
    {"conversations": [{"from": "system", "value": <system_prompt>},
                       {"from": "human", "value": <user_prompt>},
                       {"from": "gpt", "value": <annotation_output>}],
     "sample_id": str, "source_target_id": str, "annotator_id": str}

DPO format (LLaMA-Factory Alpaca DPO, one line per chosen/rejected pair):
    {"prompt": <prompt_text>, "chosen": str, "rejected": str}

Notes
-----
- Classifier and rule paradigms are **not** emitted as SFT/DPO: their data is consumed
  by the audio_cnn_trainer and rule-based pipelines in pet-train, not by LLaMA-Factory.
  Calling to_sft_samples(annotator_type="classifier"|"rule") emits a warning and returns
  an empty list.
- Human paradigm produces valid SFT samples: the gpt turn is the serialised human
  decision (decision/reviewer/notes) as JSON.
- Prompt text is reconstructed from ``schema_version`` via
  ``pet_schema.renderer.render_prompt(schema_version)``.  The reconstruction is
  deterministic for a given schema version but does **not** include runtime-variable
  parts (e.g. storage_uri injected into the vision prompt by the provider at inference
  time).  This is flagged as a §9 followup concern ("prompt storage", migration 006:
  store the rendered prompt per annotation row so reconstruction is exact).
- Every emitted sample is validated with ``ShareGPTSFTSample.model_validate()`` or
  ``DPOSample.model_validate()`` before write.  A ``pydantic.ValidationError`` is
  propagated immediately so exporter drift is caught at produce time, not consume time.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

from pet_schema import DPOSample, ShareGPTSFTSample, ShareGPTTurn
from pet_schema.renderer import render_prompt

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal row iterators (unchanged from pre-rewrite)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Prompt reconstruction helper
# ---------------------------------------------------------------------------

# Cache rendered prompts per (schema_version, few_shot) to avoid re-reading disk
_PROMPT_CACHE: dict[tuple[str, bool], tuple[str, str]] = {}


def _get_prompt(schema_version: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a schema_version.

    Prompts are cached in-process.  Falls back to a placeholder if the schema
    version directory does not exist (e.g. schema_version="1.0" used in tests
    where prompt files may not be present under pet-schema's installed copy).
    The fallback emits a warning so callers are aware.

    §9 concern: This reconstruction is prompt-template-level only; it does not
    include runtime image tokens / storage_uri that the LLM provider injected
    at inference time.  A future migration 006 should store the rendered prompt
    per llm_annotations row to eliminate this gap.
    """
    key = (schema_version, True)
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    try:
        result = render_prompt(version=schema_version, few_shot=True)
        _PROMPT_CACHE[key] = result
        return result
    except FileNotFoundError:
        warnings.warn(
            f"Prompt files not found for schema_version={schema_version!r}; "
            "using placeholder prompt text.  This is expected in tests but "
            "MUST be resolved in production (see §9 followup: migration 006).",
            stacklevel=3,
        )
        placeholder = (
            f"[system prompt for schema_version={schema_version!r} — "
            "prompt files not found; see §9 migration 006]"
        )
        fallback = (placeholder, placeholder)
        _PROMPT_CACHE[key] = fallback
        return fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_sft_samples(
    store: AnnotationStore,
    annotator_type: str = "llm",
    output_path: Path | None = None,
) -> list[dict]:
    """Export done annotations to SFT JSONL format (LLaMA-Factory ShareGPT).

    Each sample is a ``ShareGPTSFTSample`` with a ``conversations`` list.  For
    LLM and human paradigms, conversations follow the human/gpt turn pattern.
    Classifier and rule paradigms are skipped (not natural SFT/DPO shape —
    those data pipelines go through pet-train's non-LLaMA-Factory plugins).

    Every emitted sample is validated via ``ShareGPTSFTSample.model_validate()``
    before write; a ``pydantic.ValidationError`` propagates immediately.

    Args:
        store: Initialised AnnotationStore.
        annotator_type: Which paradigm table to read from ('llm', 'classifier',
            'rule', 'human'). Defaults to 'llm'.
        output_path: If provided, write JSONL to this path.  Otherwise return
            list of sample dicts without writing.

    Returns:
        List of sample dicts (by_alias=True — JSON field names).

    Raises:
        ValueError: If annotator_type is unrecognised.
        pydantic.ValidationError: If any emitted sample fails schema validation.
    """
    samples: list[dict] = []

    if annotator_type in ("classifier", "rule"):
        warnings.warn(
            f"to_sft_samples(annotator_type={annotator_type!r}) called: "
            f"{annotator_type} data is not SFT/DPO-compatible (text-to-text) and "
            "is consumed by pet-train's dedicated classifier/rule plugins, not by "
            "LLaMA-Factory SFT.  Returning empty list.",
            UserWarning,
            stacklevel=2,
        )
        return []

    elif annotator_type == "llm":
        from urllib.parse import urlparse

        def _resolve_image_path(uri: str | None) -> str | None:
            """Resolve pet-data URI (RFC 3986) to local path or pass-through (F005 helper).

            For VLM SFT, LLaMA-Factory wants real filesystem paths or URLs in the
            ``images`` field. ``local:///abs/path`` → ``/abs/path``; http(s)/s3
            pass through as-is.
            """
            if not uri:
                return None
            parsed = urlparse(uri)
            if parsed.scheme in ("", "file", "local"):
                return parsed.path or uri
            return uri

        for row in _iter_done_llm_rows(store):
            system_prompt, user_prompt = _get_prompt(row["schema_version"])
            output_text = row["raw_response"] or json.dumps(row["parsed_output"])
            # F001 (v3.3.0): inject <image> placeholder + populate images field
            # for VLM SFT. Phase 3A production format restored after v3.2.0
            # regression. text-only fallback when storage_uri is empty.
            resolved = _resolve_image_path(row.get("storage_uri"))
            if resolved:
                images_list: list[str] | None = [resolved]
                user_value = "<image>\n" + user_prompt
            else:
                images_list = None
                user_value = user_prompt
            sample = ShareGPTSFTSample(
                conversations=[
                    ShareGPTTurn(**{"from": "system", "value": system_prompt}),
                    ShareGPTTurn(**{"from": "human", "value": user_value}),
                    ShareGPTTurn(**{"from": "gpt", "value": output_text}),
                ],
                images=images_list,
                sample_id=row["target_id"],
                source_target_id=row["target_id"],
                annotator_id=row["annotator_id"],
            )
            ShareGPTSFTSample.model_validate(sample.model_dump(by_alias=True))
            samples.append(sample.model_dump(by_alias=True))

    elif annotator_type == "human":
        # Human annotations are serialised as JSON in the gpt turn; system/user
        # prompts are placeholders since human review does not use an LLM prompt.
        for row in _iter_done_human_rows(store):
            human_out = json.dumps({
                "decision": row["decision"],
                "reviewer": row["reviewer"],
                "notes": row["notes"],
            }, ensure_ascii=False)
            prompt_text = (
                f"Review annotation for target {row['target_id']!r}. "
                "Provide decision (accept/reject), reviewer ID, and notes."
            )
            sample = ShareGPTSFTSample(
                conversations=[
                    ShareGPTTurn(**{"from": "human", "value": prompt_text}),
                    ShareGPTTurn(**{"from": "gpt", "value": human_out}),
                ],
                sample_id=row["target_id"],
                source_target_id=row["target_id"],
                annotator_id=row["annotator_id"],
            )
            ShareGPTSFTSample.model_validate(sample.model_dump(by_alias=True))
            samples.append(sample.model_dump(by_alias=True))

    else:
        raise ValueError(
            f"Unknown annotator_type {annotator_type!r}. "
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
    """Export done annotations to DPO JSONL format (LLaMA-Factory Alpaca DPO).

    For LLM paradigm: groups annotations by target_id; if a target has 2+
    annotations from different annotators, the annotation with higher
    ``confidence_overall`` becomes 'chosen' and the other 'rejected'.  The
    ``prompt`` field is reconstructed from the schema_version via
    ``render_prompt()`` (concatenation of system + user prompt).

    For classifier/rule/human: returns empty list with a warning.  Classifier
    and rule data are not consumed by LLaMA-Factory DPO; human DPO pairs are
    not formed automatically here (no natural chosen/rejected signal).

    Every emitted pair is validated via ``DPOSample.model_validate()`` before
    write; a ``pydantic.ValidationError`` propagates immediately.

    Args:
        store: Initialised AnnotationStore.
        annotator_type: Which paradigm to export pairs for.
        output_path: If provided, write JSONL to this path.

    Returns:
        List of pair dicts.

    Raises:
        ValueError: If annotator_type is unrecognised.
        pydantic.ValidationError: If any emitted pair fails schema validation.
    """
    pairs: list[dict] = []

    if annotator_type != "llm":
        warnings.warn(
            f"to_dpo_pairs(annotator_type={annotator_type!r}) called: "
            "DPO pair formation is only meaningful for the 'llm' paradigm.  "
            "Classifier/rule pairs have no confidence signal; human DPO pairs "
            "require manual preference signal.  Returning empty list.",
            UserWarning,
            stacklevel=2,
        )
        if annotator_type not in ("classifier", "rule", "human"):
            raise ValueError(
                f"Unknown annotator_type {annotator_type!r}. "
                "Must be one of: llm, classifier, rule, human."
            )
        return []

    # Group by target_id
    target_rows: dict[str, list[dict]] = {}
    for row in _iter_done_llm_rows(store):
        target_rows.setdefault(row["target_id"], []).append(row)

    for target_id, rows in target_rows.items():
        if len(rows) < 2:
            continue  # Need at least 2 annotations to form a pair

        def _confidence(r: dict) -> float:
            """Extract confidence_overall from parsed_output."""
            try:
                return float(
                    r["parsed_output"].get("scene", {}).get("confidence_overall", 0)
                )
            except (TypeError, ValueError):
                return 0.0

        sorted_rows = sorted(rows, key=_confidence, reverse=True)
        chosen = sorted_rows[0]
        rejected = sorted_rows[-1]

        # Reconstruct prompt from schema_version (same for all rows in a target group)
        system_prompt, user_prompt = _get_prompt(chosen["schema_version"])
        prompt_text = f"{system_prompt}\n\n{user_prompt}"

        # pet-schema v3.2.1 added prompt field to DPOSample — complete Alpaca DPO
        # contract: producer builds + validates in one step, no dict injection.
        pair = DPOSample(
            prompt=prompt_text,
            sample_id=target_id,
            chosen=chosen["raw_response"],
            rejected=rejected["raw_response"],
            chosen_annotator_id=chosen["annotator_id"],
            rejected_annotator_id=rejected["annotator_id"],
            storage_uri=chosen["storage_uri"],
        )
        # Round-trip validation as fail-fast on any future shape drift.
        DPOSample.model_validate(pair.model_dump())
        pairs.append(pair.model_dump())

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("Exported %d DPO pairs to %s", len(pairs), output_path)

    return pairs
