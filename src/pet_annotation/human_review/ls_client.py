"""Label Studio REST API client.

Thin wrapper over an authenticated requests.Session (from ls_auth.py) that
exposes the two operations needed by the human paradigm orchestrator:

  - submit_tasks(): POST tasks to a project for human review
  - fetch_completed_annotations(): GET completed task annotations
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Number of tasks per import request to LS (keep under LS payload limit).
_IMPORT_CHUNK_SIZE = 100


class LSClient:
    """Label Studio REST API client for a single project.

    Args:
        base_url: Label Studio instance URL (e.g. http://localhost:8080).
        project_id: LS project ID to operate on.
        session: Authenticated requests.Session (from ls_auth.get_ls_session).
    """

    def __init__(
        self,
        base_url: str,
        project_id: int,
        session: requests.Session,
    ) -> None:
        """Initialise the client.

        Args:
            base_url: Label Studio instance URL.
            project_id: LS project ID.
            session: Authenticated requests.Session.
        """
        self._base_url = base_url.rstrip("/")
        self._project_id = project_id
        self._session = session

    def submit_tasks(self, tasks: list[dict[str, Any]]) -> list[int]:
        """POST tasks to the LS project and return assigned task IDs.

        Sends tasks in chunks of up to _IMPORT_CHUNK_SIZE to stay under LS
        payload limits. Returns the LS-assigned task IDs in submission order.

        Args:
            tasks: List of task dicts. Each dict must have at minimum a 'data'
                key; a 'meta' key can carry target_id for mapping back on pull.
                Example: {"data": {"image": "s3://bucket/frame.jpg"},
                          "meta": {"target_id": "frame-001"}}

        Returns:
            List of LS integer task IDs in the order they were submitted.

        Raises:
            requests.HTTPError: On non-2xx HTTP response from LS.
        """
        assigned_ids: list[int] = []
        for i in range(0, len(tasks), _IMPORT_CHUNK_SIZE):
            chunk = tasks[i : i + _IMPORT_CHUNK_SIZE]
            url = f"{self._base_url}/api/projects/{self._project_id}/import"
            resp = self._session.post(url, json=chunk, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            # LS /import returns {"task_count": N, "task_ids": [...]}
            ids: list[int] = payload.get("task_ids", [])
            assigned_ids.extend(ids)
            logger.info(
                '{"event": "ls_tasks_submitted", "project_id": %d, '
                '"chunk_size": %d, "returned_ids": %d}',
                self._project_id,
                len(chunk),
                len(ids),
            )
        return assigned_ids

    def fetch_completed_annotations(
        self, updated_after: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fetch completed task annotations from the LS project.

        Queries /api/tasks?project=<id>&annotation_results=true and filters
        to tasks that have at least one completed annotation. Optionally filters
        by updated_at > updated_after.

        Args:
            updated_after: If provided, only return tasks updated after this
                datetime (UTC). None returns all completed tasks.

        Returns:
            List of task dicts as returned by LS, each containing at minimum:
                - id: LS task ID (int)
                - meta: dict with target_id (if set at submit time)
                - annotations: list of annotation result dicts

        Raises:
            requests.HTTPError: On non-2xx HTTP response from LS.
        """
        url = f"{self._base_url}/api/tasks"
        params: dict[str, Any] = {
            "project": self._project_id,
            "page_size": 1000,
        }
        if updated_after is not None:
            # LS uses ISO 8601; updated_at__gt is supported as a query param.
            params["updated_at__gt"] = updated_after.isoformat()

        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        # LS returns {"tasks": [...]} or just a list depending on version.
        if isinstance(payload, dict):
            tasks: list[dict[str, Any]] = payload.get("tasks", [])
        else:
            tasks = payload

        # Filter to tasks with at least one completed annotation.
        completed = [t for t in tasks if t.get("annotations")]
        logger.info(
            '{"event": "ls_fetch_completed", "project_id": %d, '
            '"total_tasks": %d, "completed": %d}',
            self._project_id,
            len(tasks),
            len(completed),
        )
        return completed
