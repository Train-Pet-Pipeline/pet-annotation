"""Export completed Label Studio annotations back to the database.

Updates review_status and optionally overwrites parsed_output if the
reviewer made corrections.  Corrected annotations also generate DPO
pairs (original=rejected, corrected=chosen).
"""
from __future__ import annotations

import json
import logging

import requests

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def export_reviewed(
    store: AnnotationStore,
    ls_url: str,
    session: requests.Session,
) -> int:
    """Pull completed annotations from Label Studio and update DB.

    For each completed task:
    - 'approve' → review_status='reviewed', frame_status='approved'
    - 'reject'  → review_status='rejected', frame_status='rejected'
    - 'correct' → review_status='reviewed', parsed_output overwritten,
                   frame_status='approved', and a DPO pair is stored

    Args:
        store: AnnotationStore instance.
        ls_url: Label Studio server URL.
        session: Authenticated requests.Session (from ``ls_auth.get_ls_session``).

    Returns:
        Number of annotations updated.
    """
    # Find the project
    resp = session.get(f"{ls_url}/api/projects", timeout=30)
    resp.raise_for_status()
    project_id = None
    for proj in resp.json().get("results", []):
        if proj["title"] == "pet-annotation-review":
            project_id = proj["id"]
            break

    if project_id is None:
        logger.info('{"event": "ls_export_skip", "reason": "no_project"}')
        return 0

    # Fetch completed tasks
    resp = session.get(
        f"{ls_url}/api/projects/{project_id}/tasks",
        params={"fields": "all", "page_size": 10000},
        timeout=60,
    )
    resp.raise_for_status()
    tasks = resp.json() if isinstance(resp.json(), list) else resp.json().get("tasks", [])

    updated = 0
    for task in tasks:
        completions = task.get("annotations", [])
        if not completions:
            continue  # Not yet reviewed

        data = task.get("data", {})
        annotation_id = data.get("annotation_id")
        frame_id = data.get("frame_id")
        if not annotation_id or not frame_id:
            continue

        # Take the latest completion
        latest = completions[-1]
        result_map = {}
        for r in latest.get("result", []):
            result_map[r.get("from_name", "")] = r.get("value", {})

        decision_val = result_map.get("review_decision", {})
        decision = decision_val.get("choices", ["approve"])[0] if decision_val else "approve"

        if decision == "approve":
            store.update_review_and_frame_status(
                annotation_id, "reviewed", frame_id, "approved"
            )
            updated += 1

        elif decision == "reject":
            store.update_review_and_frame_status(
                annotation_id, "rejected", frame_id, "rejected"
            )
            updated += 1

        elif decision == "correct":
            # Get corrected output from TextArea (field may be named
            # "corrected_output" or "vlm_output" depending on project config)
            corrected_val = result_map.get("corrected_output") or result_map.get("vlm_output", {})
            corrected_texts = corrected_val.get("text", [])
            corrected_output = corrected_texts[0] if corrected_texts else None

            if corrected_output:
                # Validate corrected JSON
                try:
                    json.loads(corrected_output)
                except json.JSONDecodeError:
                    logger.warning(
                        '{"event": "ls_invalid_correction", "annotation_id": "%s"}',
                        annotation_id,
                    )
                    continue

                # Save original as DPO rejected, corrected as chosen
                _store_human_dpo_pair(store, annotation_id, frame_id, corrected_output)

                # Update annotation with corrected output
                store.update_annotation_parsed_output(annotation_id, corrected_output)

            store.update_review_and_frame_status(
                annotation_id, "reviewed", frame_id, "approved"
            )
            updated += 1

    logger.info('{"event": "ls_export_done", "updated": %d}', updated)
    return updated


def _store_human_dpo_pair(
    store: AnnotationStore,
    annotation_id: str,
    frame_id: str,
    corrected_output: str,
) -> None:
    """Store a DPO pair from human correction.

    Args:
        store: AnnotationStore instance.
        annotation_id: The corrected annotation ID.
        frame_id: The frame ID.
        corrected_output: Human-corrected JSON string.
    """
    # Fetch original raw response for the rejected side
    conn = store._conn  # noqa: SLF001
    row = conn.execute(
        "SELECT raw_response, model_name FROM annotations WHERE annotation_id=?",
        (annotation_id,),
    ).fetchone()
    if not row:
        return

    original_output = row["raw_response"]

    # Insert into dpo_pairs table if it exists
    try:
        import uuid

        pair_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO dpo_pairs (pair_id, frame_id, chosen, rejected, source, metadata)
            VALUES (?, ?, ?, ?, 'human_correction', ?)
            """,
            (
                pair_id,
                frame_id,
                corrected_output,
                original_output,
                json.dumps({
                    "annotation_id": annotation_id,
                    "model_name": row["model_name"],
                }),
            ),
        )
        conn.commit()
        logger.info(
            '{"event": "dpo_pair_stored", "pair_id": "%s", "source": "human_correction"}',
            pair_id,
        )
    except Exception:
        logger.debug("dpo_pairs table not available, skipping DPO storage")
