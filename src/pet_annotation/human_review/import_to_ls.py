"""Import VLM outputs to Label Studio as review tasks.

Pre-fills VLM output as predictions so reviewers see the model's answer
and can quickly confirm or correct.
"""
from __future__ import annotations

import json
import logging

import requests

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)

# Label Studio project config for pet annotation review.
# TextArea allows reviewer to edit the VLM JSON output directly.
LABELING_CONFIG = """
<View>
  <Header value="Frame Image" />
  <Image name="image" value="$image_url" />

  <Header value="Species" />
  <Text name="species_display" value="$species" />

  <Header value="VLM Output (edit if corrections needed)" />
  <TextArea name="corrected_output" toName="image"
            value="$vlm_output" rows="20" editable="true"
            maxSubmissions="1" />

  <Header value="Review Decision" />
  <Choices name="review_decision" toName="image" choice="single">
    <Choice value="approve" />
    <Choice value="reject" />
    <Choice value="correct" />
  </Choices>
</View>
"""


def _ensure_project(ls_url: str, session: requests.Session, data_root: str) -> int:
    """Find or create the pet-annotation-review project.

    Also ensures a local file storage is connected so that images served
    via ``/data/local-files/`` resolve correctly.

    Args:
        ls_url: Label Studio base URL.
        session: Authenticated requests session.
        data_root: Absolute path to the data root directory.

    Returns:
        Label Studio project ID.
    """
    resp = session.get(f"{ls_url}/api/projects", timeout=30)
    resp.raise_for_status()
    for proj in resp.json().get("results", []):
        if proj["title"] == "pet-annotation-review":
            project_id = proj["id"]
            _ensure_local_storage(ls_url, session, project_id, data_root)
            return project_id

    # Create new project
    payload = {
        "title": "pet-annotation-review",
        "label_config": LABELING_CONFIG,
    }
    resp = session.post(
        f"{ls_url}/api/projects", json=payload, timeout=30
    )
    resp.raise_for_status()
    project_id = resp.json()["id"]
    logger.info('{"event": "ls_project_created", "project_id": %d}', project_id)

    _ensure_local_storage(ls_url, session, project_id, data_root)
    return project_id


def _ensure_local_storage(
    ls_url: str, session: requests.Session, project_id: int, data_root: str,
) -> None:
    """Create a local file storage connection if one doesn't exist.

    Args:
        ls_url: Label Studio base URL.
        session: Authenticated requests session.
        project_id: Label Studio project ID.
        data_root: Absolute path to the data root (frames parent directory).
    """
    # Check existing storages
    resp = session.get(
        f"{ls_url}/api/storages/localfiles",
        params={"project": project_id},
        timeout=10,
    )
    if resp.ok and resp.json():
        return  # Storage already exists

    # Create local storage pointing to frames subdirectory.
    # LS requires the path to be a subdirectory of DOCUMENT_ROOT,
    # not DOCUMENT_ROOT itself.
    frames_path = f"{data_root}/frames"
    resp = session.post(
        f"{ls_url}/api/storages/localfiles",
        json={
            "project": project_id,
            "path": frames_path,
            "regex_filter": r".*\.png$",
            "use_blob_urls": True,
            "title": "pet-frames",
        },
        timeout=10,
    )
    if resp.ok:
        logger.info(
            '{"event": "ls_storage_created", "project_id": %d, "path": "%s"}',
            project_id, frames_path,
        )
    else:
        logger.warning(
            '{"event": "ls_storage_failed", "status": %d, "detail": "%s"}',
            resp.status_code, resp.text[:200],
        )


def import_needs_review(
    store: AnnotationStore,
    ls_url: str,
    session: requests.Session,
    data_root: str = "",
) -> int:
    """Create Label Studio tasks for annotations needing review.

    Queries annotations with review_status='needs_review', builds LS tasks
    with VLM output pre-filled as predictions, and creates them via LS API.

    Args:
        store: AnnotationStore instance.
        ls_url: Label Studio server URL.
        session: Authenticated requests.Session (from ``ls_auth.get_ls_session``).
        data_root: Absolute path to the data root directory.

    Returns:
        Number of tasks created.
    """
    project_id = _ensure_project(ls_url, session, data_root)

    rows = store.fetch_needs_review_annotations()
    if not rows:
        logger.info('{"event": "ls_import_skip", "reason": "no_needs_review"}')
        return 0

    tasks = []
    for row in rows:
        # Use frame_path relative to data_root; Label Studio's
        # LOCAL_FILES_DOCUMENT_ROOT should point to data_root so the
        # full disk path = DOCUMENT_ROOT / frame_path.
        frame_path = row["frame_path"]
        vlm_output = row["raw_response"]

        # Pretty-print JSON for readability in LS
        try:
            vlm_output = json.dumps(json.loads(vlm_output), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass

        tasks.append({
            "data": {
                "image_url": f"/data/local-files/?d={frame_path}",
                "vlm_output": vlm_output,
                "species": row["species"] if row["species"] else "unknown",
                "annotation_id": row["annotation_id"],
                "frame_id": row["frame_id"],
                "model_name": row["model_name"],
            },
        })

    # Bulk import tasks
    resp = session.post(
        f"{ls_url}/api/projects/{project_id}/import",
        json=tasks,
        timeout=60,
    )
    resp.raise_for_status()
    created = resp.json().get("task_count", len(tasks))

    logger.info(
        '{"event": "ls_tasks_imported", "count": %d, "project_id": %d}',
        created, project_id,
    )
    return created
