"""Annotation store — 4 annotator-paradigm tables + annotation_targets state machine.

Spec: docs/superpowers/specs/2026-04-21-phase-2-debt-repayment-design.md §2
Phase 4 additions: annotation_targets table for per-(target, annotator) state tracking.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pet_schema import ClassifierAnnotation, HumanAnnotation, LLMAnnotation, RuleAnnotation

_MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"
_APPLIED_MIGRATIONS_TABLE = "_applied_migrations"


def _dumps(d) -> str:
    """Serialise dict/list to JSON string with sorted keys for determinism."""
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _loads(s: str | None):
    """Parse JSON string; return None if s is None."""
    return json.loads(s) if s else None


class AnnotationStore:
    """SQLite-backed store for annotation data — 4 annotator-paradigm tables.

    Args:
        db_path: Path string to the SQLite database file.
        busy_timeout_ms: SQLite busy_timeout in milliseconds; loaded from
            params.yaml database.busy_timeout_ms (default 10000).
    """

    def __init__(self, db_path: str, busy_timeout_ms: int = 10000) -> None:
        """Initialise store and create migrations tracking table.

        Args:
            db_path: Path to the SQLite database file.
            busy_timeout_ms: Milliseconds to wait for a locked database before
                raising an OperationalError (PRAGMA busy_timeout).
        """
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        self._ensure_migrations_table()

    def init_schema(self) -> None:
        """Run all migrations/ .sql files in sorted order (skip already-applied)."""
        for mig_path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            name = mig_path.name
            if self._already_applied(name):
                continue
            self._apply_migration(name, mig_path.read_text())

    def _ensure_migrations_table(self) -> None:
        """Create the _applied_migrations tracking table if it does not exist."""
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {_APPLIED_MIGRATIONS_TABLE} "
            "(name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        self._conn.commit()

    def _already_applied(self, name: str) -> bool:
        """Return True if migration *name* has already been recorded as applied."""
        row = self._conn.execute(
            f"SELECT 1 FROM {_APPLIED_MIGRATIONS_TABLE} WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    def _apply_migration(self, name: str, sql: str) -> None:
        """Execute *sql* statements and record *name* as applied."""
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            self._conn.execute(stmt)
        self._conn.execute(
            f"INSERT INTO {_APPLIED_MIGRATIONS_TABLE}(name) VALUES (?)", (name,)
        )
        self._conn.commit()

    # ---- LLM ----

    def insert_llm(self, ann: LLMAnnotation) -> None:
        """Insert an LLMAnnotation row into llm_annotations.

        Args:
            ann: The LLMAnnotation to persist.
        """
        self._conn.execute(
            "INSERT INTO llm_annotations"
            "(annotation_id, target_id, annotator_id, annotator_type, "
            "modality, schema_version, created_at, storage_uri, "
            "prompt_hash, raw_response, parsed_output) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                ann.annotation_id,
                ann.target_id,
                ann.annotator_id,
                ann.annotator_type,
                ann.modality,
                ann.schema_version,
                ann.created_at.isoformat(),
                ann.storage_uri,
                ann.prompt_hash,
                ann.raw_response,
                _dumps(ann.parsed_output),
            ),
        )
        self._conn.commit()

    def fetch_llm_by_target(self, target_id: str) -> list[LLMAnnotation]:
        """Fetch all LLMAnnotation rows for *target_id*.

        Args:
            target_id: The target identifier to filter by.

        Returns:
            List of LLMAnnotation objects.
        """
        cur = self._conn.execute(
            "SELECT annotation_id, target_id, annotator_id, annotator_type, modality, "
            "schema_version, created_at, storage_uri, prompt_hash, raw_response, parsed_output "
            "FROM llm_annotations WHERE target_id = ?",
            (target_id,),
        )
        return [
            LLMAnnotation(
                annotation_id=r[0],
                target_id=r[1],
                annotator_id=r[2],
                annotator_type=r[3],
                modality=r[4],
                schema_version=r[5],
                created_at=r[6],
                storage_uri=r[7],
                prompt_hash=r[8],
                raw_response=r[9],
                parsed_output=_loads(r[10]),
            )
            for r in cur.fetchall()
        ]

    # ---- Classifier ----

    def insert_classifier(self, ann: ClassifierAnnotation) -> None:
        """Insert a ClassifierAnnotation row into classifier_annotations.

        Args:
            ann: The ClassifierAnnotation to persist.
        """
        self._conn.execute(
            "INSERT INTO classifier_annotations"
            "(annotation_id, target_id, annotator_id, annotator_type, "
            "modality, schema_version, created_at, storage_uri, "
            "predicted_class, class_probs, logits) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                ann.annotation_id,
                ann.target_id,
                ann.annotator_id,
                ann.annotator_type,
                ann.modality,
                ann.schema_version,
                ann.created_at.isoformat(),
                ann.storage_uri,
                ann.predicted_class,
                _dumps(ann.class_probs),
                _dumps(ann.logits) if ann.logits is not None else None,
            ),
        )
        self._conn.commit()

    def fetch_classifier_by_target(self, target_id: str) -> list[ClassifierAnnotation]:
        """Fetch all ClassifierAnnotation rows for *target_id*.

        Args:
            target_id: The target identifier to filter by.

        Returns:
            List of ClassifierAnnotation objects.
        """
        cur = self._conn.execute(
            "SELECT annotation_id, target_id, annotator_id, annotator_type, modality, "
            "schema_version, created_at, storage_uri, predicted_class, class_probs, logits "
            "FROM classifier_annotations WHERE target_id = ?",
            (target_id,),
        )
        return [
            ClassifierAnnotation(
                annotation_id=r[0],
                target_id=r[1],
                annotator_id=r[2],
                annotator_type=r[3],
                modality=r[4],
                schema_version=r[5],
                created_at=r[6],
                storage_uri=r[7],
                predicted_class=r[8],
                class_probs=_loads(r[9]),
                logits=_loads(r[10]),
            )
            for r in cur.fetchall()
        ]

    # ---- Rule ----

    def insert_rule(self, ann: RuleAnnotation) -> None:
        """Insert a RuleAnnotation row into rule_annotations.

        Args:
            ann: The RuleAnnotation to persist.
        """
        self._conn.execute(
            "INSERT INTO rule_annotations"
            "(annotation_id, target_id, annotator_id, annotator_type, "
            "modality, schema_version, created_at, storage_uri, "
            "rule_id, rule_output) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                ann.annotation_id,
                ann.target_id,
                ann.annotator_id,
                ann.annotator_type,
                ann.modality,
                ann.schema_version,
                ann.created_at.isoformat(),
                ann.storage_uri,
                ann.rule_id,
                _dumps(ann.rule_output),
            ),
        )
        self._conn.commit()

    def fetch_rule_by_target(self, target_id: str) -> list[RuleAnnotation]:
        """Fetch all RuleAnnotation rows for *target_id*.

        Args:
            target_id: The target identifier to filter by.

        Returns:
            List of RuleAnnotation objects.
        """
        cur = self._conn.execute(
            "SELECT annotation_id, target_id, annotator_id, annotator_type, modality, "
            "schema_version, created_at, storage_uri, rule_id, rule_output "
            "FROM rule_annotations WHERE target_id = ?",
            (target_id,),
        )
        return [
            RuleAnnotation(
                annotation_id=r[0],
                target_id=r[1],
                annotator_id=r[2],
                annotator_type=r[3],
                modality=r[4],
                schema_version=r[5],
                created_at=r[6],
                storage_uri=r[7],
                rule_id=r[8],
                rule_output=_loads(r[9]),
            )
            for r in cur.fetchall()
        ]

    # ---- Human ----

    def insert_human(self, ann: HumanAnnotation) -> None:
        """Insert a HumanAnnotation row into human_annotations.

        Args:
            ann: The HumanAnnotation to persist.
        """
        self._conn.execute(
            "INSERT INTO human_annotations"
            "(annotation_id, target_id, annotator_id, annotator_type, "
            "modality, schema_version, created_at, storage_uri, "
            "reviewer, decision, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                ann.annotation_id,
                ann.target_id,
                ann.annotator_id,
                ann.annotator_type,
                ann.modality,
                ann.schema_version,
                ann.created_at.isoformat(),
                ann.storage_uri,
                ann.reviewer,
                ann.decision,
                ann.notes,
            ),
        )
        self._conn.commit()

    # ---- annotation_targets (Phase 4 orchestrator wire) ----

    def ingest_pending_from_petdata(
        self,
        pet_data_db_path: str,
        annotator_ids: list[str],
        annotator_type: str,
        modality: str | None = None,
    ) -> int:
        """Read pending frames from pet-data's frames table and register new targets.

        Connects read-only to pet-data's SQLite. For each pending frame × annotator_id
        combination not already tracked, inserts a row with state='pending'.

        Args:
            pet_data_db_path: Path to pet-data's SQLite database file.
            annotator_ids: List of annotator IDs to register targets for.
            annotator_type: Annotator paradigm type (e.g. 'llm').
            modality: Optional modality filter; None means all modalities.

        Returns:
            Number of newly inserted annotation_targets rows (idempotent: 0 on re-ingest).
        """
        # Read-only URI mode (D1 enforcement); prevents accidental writes across repos.
        petdata_conn = sqlite3.connect(
            f"file:{pet_data_db_path}?mode=ro", uri=True, timeout=10
        )
        try:
            cur = petdata_conn.execute(
                "SELECT frame_id FROM frames "
                "WHERE annotation_status = 'pending' AND (modality = ? OR ? IS NULL)",
                (modality, modality),
            )
            frame_ids = [row[0] for row in cur.fetchall()]
        finally:
            petdata_conn.close()

        if not frame_ids or not annotator_ids:
            return 0

        new_count = 0
        for frame_id in frame_ids:
            for annotator_id in annotator_ids:
                cur2 = self._conn.execute(
                    "INSERT OR IGNORE INTO annotation_targets"
                    "(target_id, annotator_id, annotator_type, state, "
                    "claimed_at, finished_at, error_msg) "
                    "VALUES (?, ?, ?, 'pending', NULL, NULL, NULL)",
                    (frame_id, annotator_id, annotator_type),
                )
                new_count += cur2.rowcount
        self._conn.commit()
        return new_count

    def claim_pending_targets(self, annotator_id: str, batch_size: int) -> list[str]:
        """Atomically claim up to batch_size pending targets for annotator_id.

        Uses BEGIN IMMEDIATE to prevent concurrent double-claim. Transitions
        state from 'pending' → 'in_progress' and records claimed_at timestamp.

        Args:
            annotator_id: The annotator claiming the targets.
            batch_size: Maximum number of targets to claim.

        Returns:
            List of claimed target_id strings.
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self._conn.execute(
                "SELECT target_id FROM annotation_targets "
                "WHERE annotator_id = ? AND state = 'pending' "
                "LIMIT ?",
                (annotator_id, batch_size),
            ).fetchall()
            target_ids = [r[0] for r in rows]
            if target_ids:
                placeholders = ",".join("?" * len(target_ids))
                self._conn.execute(
                    f"UPDATE annotation_targets SET state='in_progress', claimed_at=? "
                    f"WHERE annotator_id=? AND target_id IN ({placeholders})",
                    [now, annotator_id, *target_ids],
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return target_ids

    def mark_target_done(self, target_id: str, annotator_id: str) -> None:
        """Transition target state to 'done' and record finished_at timestamp.

        Args:
            target_id: The target identifier.
            annotator_id: The annotator that completed the target.
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE annotation_targets SET state='done', finished_at=? "
            "WHERE target_id=? AND annotator_id=?",
            (now, target_id, annotator_id),
        )
        self._conn.commit()

    def mark_target_failed(self, target_id: str, annotator_id: str, error_msg: str) -> None:
        """Transition target state to 'failed' with error message.

        Args:
            target_id: The target identifier.
            annotator_id: The annotator that failed the target.
            error_msg: Error description to store for debugging.
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "UPDATE annotation_targets SET state='failed', finished_at=?, error_msg=? "
            "WHERE target_id=? AND annotator_id=?",
            (now, error_msg, target_id, annotator_id),
        )
        self._conn.commit()

    def get_target_state(self, target_id: str, annotator_id: str) -> str | None:
        """Query the current state of a (target_id, annotator_id) pair.

        Args:
            target_id: The target identifier.
            annotator_id: The annotator identifier.

        Returns:
            Current state string, or None if the row does not exist.
        """
        row = self._conn.execute(
            "SELECT state FROM annotation_targets WHERE target_id=? AND annotator_id=?",
            (target_id, annotator_id),
        ).fetchone()
        return row[0] if row else None

    def fetch_human_by_target(self, target_id: str) -> list[HumanAnnotation]:
        """Fetch all HumanAnnotation rows for *target_id*.

        Args:
            target_id: The target identifier to filter by.

        Returns:
            List of HumanAnnotation objects.
        """
        cur = self._conn.execute(
            "SELECT annotation_id, target_id, annotator_id, annotator_type, modality, "
            "schema_version, created_at, storage_uri, reviewer, decision, notes "
            "FROM human_annotations WHERE target_id = ?",
            (target_id,),
        )
        return [
            HumanAnnotation(
                annotation_id=r[0],
                target_id=r[1],
                annotator_id=r[2],
                annotator_type=r[3],
                modality=r[4],
                schema_version=r[5],
                created_at=r[6],
                storage_uri=r[7],
                reviewer=r[8],
                decision=r[9],
                notes=r[10],
            )
            for r in cur.fetchall()
        ]
