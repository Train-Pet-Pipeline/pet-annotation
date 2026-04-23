"""Annotation store — 4 annotator-paradigm tables.

Spec: docs/superpowers/specs/2026-04-21-phase-2-debt-repayment-design.md §2
"""

from __future__ import annotations

import json
import sqlite3
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
