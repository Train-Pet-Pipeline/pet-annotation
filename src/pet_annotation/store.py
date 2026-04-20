"""AnnotationStore: all SQLite access for the pet-annotation pipeline.

This module is the *only* place that touches the annotations and
model_comparisons tables.  It also issues SELECT / UPDATE against the
frames table (owned by pet-data) but never DDL on it.

Usage (production)::

    store = AnnotationStore(db_path=Path("/data/pet.db"))
    with store:
        rows = store.fetch_pending_frames(limit=32)

Usage (tests)::

    store = AnnotationStore(conn=in_memory_conn)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "migrations"


@dataclass
class AnnotationRecord:
    """Represents one row in the annotations table."""

    annotation_id: str
    frame_id: str
    model_name: str
    prompt_hash: str
    raw_response: str
    schema_valid: int
    parsed_output: str | None = None
    validation_errors: str | None = None
    confidence_overall: float | None = None
    review_status: str = "pending"
    reviewer: str | None = None
    review_notes: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    api_latency_ms: int | None = None


@dataclass
class ComparisonRecord:
    """Represents one row in the model_comparisons table."""

    comparison_id: str
    frame_id: str
    model_name: str
    prompt_hash: str
    raw_response: str
    schema_valid: int
    parsed_output: str | None = None
    validation_errors: str | None = None
    confidence_overall: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    api_latency_ms: int | None = None


class AnnotationStore:
    """SQLite-backed store for annotation and comparison data.

    Accepts either an existing ``conn`` (for testing with :memory:) or a
    ``db_path`` for production use.  On initialisation the migration SQL is
    applied (idempotent via IF NOT EXISTS) and any frames stuck in the
    ``annotating`` status are recovered to ``pending``.
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        db_path: Path | None = None,
    ) -> None:
        """Initialise the store.

        Exactly one of *conn* or *db_path* must be supplied.

        Args:
            conn: An existing SQLite connection (used in tests).
            db_path: Path to the SQLite database file (used in production).

        Raises:
            ValueError: If neither or both arguments are supplied.
        """
        if conn is None and db_path is None:
            raise ValueError("Provide either conn or db_path")
        if conn is not None and db_path is not None:
            raise ValueError("Provide either conn or db_path, not both")

        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            self._conn = sqlite3.connect(
                str(db_path), check_same_thread=False, timeout=30
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._owns_conn = True

        self._apply_migration()
        self._recover_stuck_frames()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_migration(self) -> None:
        """Run all annotation DDL migrations in sorted order (idempotent).

        Globs ``migrations/*.sql`` sorted by filename and applies each statement
        individually.  ``ALTER TABLE … ADD COLUMN`` statements that fail with
        ``duplicate column name`` are silently skipped so that re-opening an
        already-migrated database is safe.

        Note: Migration files must not contain triggers or other BEGIN/END blocks;
        the semicolon split is not parser-aware and will break on embedded semicolons
        inside string literals or trigger bodies.
        """
        if not _MIGRATIONS_DIR.exists():
            raise RuntimeError(f"Migrations directory not found: {_MIGRATIONS_DIR}")

        for sql_file in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            sql = sql_file.read_text()
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if not stmt:
                    continue
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e).lower():
                        continue
                    raise
        self._conn.commit()

    def _recover_stuck_frames(self) -> None:
        """Reset frames stuck in 'annotating' back to 'pending' on startup."""
        self._conn.execute(
            "UPDATE frames SET annotation_status='pending' WHERE annotation_status='annotating'"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Frame queries
    # ------------------------------------------------------------------

    def fetch_pending_frames(self, limit: int) -> list[sqlite3.Row]:
        """Return up to *limit* frames with annotation_status='pending'.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of sqlite3.Row objects from the frames table.
        """
        return self._conn.execute(
            "SELECT * FROM frames WHERE annotation_status='pending' LIMIT ?",
            (limit,),
        ).fetchall()

    def update_frame_status_batch(self, frame_ids: list[str], status: str) -> None:
        """Update annotation_status for a batch of frames in one transaction.

        Args:
            frame_ids: List of frame_id values to update.
            status: The new annotation_status value.
        """
        placeholders = ",".join("?" * len(frame_ids))
        self._conn.execute(
            f"UPDATE frames SET annotation_status=? WHERE frame_id IN ({placeholders})",
            [status, *frame_ids],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Cache-hit checks
    # ------------------------------------------------------------------

    def cache_hit(self, frame_id: str, model_name: str, prompt_hash: str) -> bool:
        """Return True if an annotation already exists for this cache key.

        Args:
            frame_id: The frame identifier.
            model_name: The model that produced the annotation.
            prompt_hash: Hash of the prompt used.

        Returns:
            True if a matching row exists in annotations, False otherwise.
        """
        row = self._conn.execute(
            "SELECT 1 FROM annotations WHERE frame_id=? AND model_name=? AND prompt_hash=?",
            (frame_id, model_name, prompt_hash),
        ).fetchone()
        return row is not None

    def cache_hit_comparison(self, frame_id: str, model_name: str, prompt_hash: str) -> bool:
        """Return True if a comparison already exists for this cache key.

        Args:
            frame_id: The frame identifier.
            model_name: The model that produced the comparison.
            prompt_hash: Hash of the prompt used.

        Returns:
            True if a matching row exists in model_comparisons, False otherwise.
        """
        row = self._conn.execute(
            "SELECT 1 FROM model_comparisons WHERE frame_id=? AND model_name=? AND prompt_hash=?",
            (frame_id, model_name, prompt_hash),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------

    def insert_annotation(self, rec: AnnotationRecord) -> None:
        """Insert one row into the annotations table.

        Args:
            rec: The AnnotationRecord to persist.
        """
        self._conn.execute(
            """
            INSERT INTO annotations (
                annotation_id, frame_id, model_name, prompt_hash,
                raw_response, parsed_output, schema_valid, validation_errors,
                confidence_overall, review_status, reviewer, review_notes,
                prompt_tokens, completion_tokens, total_tokens, api_latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.annotation_id, rec.frame_id, rec.model_name, rec.prompt_hash,
                rec.raw_response, rec.parsed_output, rec.schema_valid, rec.validation_errors,
                rec.confidence_overall, rec.review_status, rec.reviewer, rec.review_notes,
                rec.prompt_tokens, rec.completion_tokens, rec.total_tokens, rec.api_latency_ms,
            ),
        )
        self._conn.commit()

    def insert_annotation_and_update_status(
        self, rec: AnnotationRecord, new_status: str
    ) -> None:
        """Atomically insert an annotation and update the frame's annotation_status.

        Args:
            rec: The AnnotationRecord to persist.
            new_status: The annotation_status value to set on the parent frame.
        """
        self._conn.execute(
            """
            INSERT INTO annotations (
                annotation_id, frame_id, model_name, prompt_hash,
                raw_response, parsed_output, schema_valid, validation_errors,
                confidence_overall, review_status, reviewer, review_notes,
                prompt_tokens, completion_tokens, total_tokens, api_latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.annotation_id, rec.frame_id, rec.model_name, rec.prompt_hash,
                rec.raw_response, rec.parsed_output, rec.schema_valid, rec.validation_errors,
                rec.confidence_overall, rec.review_status, rec.reviewer, rec.review_notes,
                rec.prompt_tokens, rec.completion_tokens, rec.total_tokens, rec.api_latency_ms,
            ),
        )
        self._conn.execute(
            "UPDATE frames SET annotation_status=? WHERE frame_id=?",
            (new_status, rec.frame_id),
        )
        self._conn.commit()

    def insert_comparison(self, rec: ComparisonRecord) -> None:
        """Insert one row into the model_comparisons table.

        Args:
            rec: The ComparisonRecord to persist.
        """
        self._conn.execute(
            """
            INSERT INTO model_comparisons (
                comparison_id, frame_id, model_name, prompt_hash,
                raw_response, parsed_output, schema_valid, validation_errors,
                confidence_overall, prompt_tokens, completion_tokens, total_tokens, api_latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.comparison_id, rec.frame_id, rec.model_name, rec.prompt_hash,
                rec.raw_response, rec.parsed_output, rec.schema_valid, rec.validation_errors,
                rec.confidence_overall, rec.prompt_tokens, rec.completion_tokens,
                rec.total_tokens, rec.api_latency_ms,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_annotation(
        self, frame_id: str, model_name: str, prompt_hash: str
    ) -> AnnotationRecord | None:
        """Retrieve an annotation by its cache key triple.

        Args:
            frame_id: The frame identifier.
            model_name: The model that produced the annotation.
            prompt_hash: Hash of the prompt used.

        Returns:
            An AnnotationRecord, or None if not found.
        """
        row = self._conn.execute(
            """
            SELECT * FROM annotations
            WHERE frame_id=? AND model_name=? AND prompt_hash=?
            """,
            (frame_id, model_name, prompt_hash),
        ).fetchone()
        if row is None:
            return None
        return AnnotationRecord(
            annotation_id=row["annotation_id"],
            frame_id=row["frame_id"],
            model_name=row["model_name"],
            prompt_hash=row["prompt_hash"],
            raw_response=row["raw_response"],
            parsed_output=row["parsed_output"],
            schema_valid=row["schema_valid"],
            validation_errors=row["validation_errors"],
            confidence_overall=row["confidence_overall"],
            review_status=row["review_status"],
            reviewer=row["reviewer"],
            review_notes=row["review_notes"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            api_latency_ms=row["api_latency_ms"],
        )

    def get_comparison(
        self, frame_id: str, model_name: str, prompt_hash: str
    ) -> ComparisonRecord | None:
        """Retrieve a comparison by its cache key triple.

        Args:
            frame_id: The frame identifier.
            model_name: The model that produced the comparison.
            prompt_hash: Hash of the prompt used.

        Returns:
            A ComparisonRecord, or None if not found.
        """
        row = self._conn.execute(
            """
            SELECT * FROM model_comparisons
            WHERE frame_id=? AND model_name=? AND prompt_hash=?
            """,
            (frame_id, model_name, prompt_hash),
        ).fetchone()
        if row is None:
            return None
        return ComparisonRecord(
            comparison_id=row["comparison_id"],
            frame_id=row["frame_id"],
            model_name=row["model_name"],
            prompt_hash=row["prompt_hash"],
            raw_response=row["raw_response"],
            parsed_output=row["parsed_output"],
            schema_valid=row["schema_valid"],
            validation_errors=row["validation_errors"],
            confidence_overall=row["confidence_overall"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            api_latency_ms=row["api_latency_ms"],
        )

    # ------------------------------------------------------------------
    # Reporting / pipeline queries
    # ------------------------------------------------------------------

    def fetch_approved_annotations(self, limit: int) -> list[sqlite3.Row]:
        """Return annotations with review_status in ('approved', 'reviewed'), joined to frames.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of sqlite3.Row objects joining annotations and frames.
        """
        return self._conn.execute(
            """
            SELECT a.*, f.frame_path, f.video_id, f.source, f.species, f.breed
            FROM annotations a
            JOIN frames f ON a.frame_id = f.frame_id
            WHERE a.review_status IN ('approved', 'reviewed')
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def fetch_comparisons_for_frame(self, frame_id: str) -> list[sqlite3.Row]:
        """Return all comparison rows for a given frame.

        Args:
            frame_id: The frame to look up comparisons for.

        Returns:
            List of sqlite3.Row objects from model_comparisons.
        """
        return self._conn.execute(
            "SELECT * FROM model_comparisons WHERE frame_id=? ORDER BY created_at",
            (frame_id,),
        ).fetchall()

    def fetch_auto_checked_annotations(self, primary_model: str) -> list[sqlite3.Row]:
        """Return annotations produced by *primary_model* that are pending quality review.

        Used by the quality-check module to find annotations ready for
        automated QA before human review.

        Args:
            primary_model: Name of the primary annotation model.

        Returns:
            List of sqlite3.Row objects from annotations where review_status='pending'.
        """
        return self._conn.execute(
            """
            SELECT * FROM annotations
            WHERE model_name=? AND review_status='pending'
            ORDER BY created_at
            """,
            (primary_model,),
        ).fetchall()

    def fetch_needs_review_annotations(self) -> list[sqlite3.Row]:
        """Return annotations with review_status='needs_review', joined to frames.

        Used by Label Studio import to build review tasks.

        Returns:
            List of sqlite3.Row objects joining annotations and frames.
        """
        return self._conn.execute(
            """
            SELECT a.*, f.frame_path, f.data_root, f.species, f.breed
            FROM annotations a
            JOIN frames f ON a.frame_id = f.frame_id
            WHERE a.review_status = 'needs_review'
            ORDER BY a.created_at
            """,
        ).fetchall()

    def update_annotation_parsed_output(
        self, annotation_id: str, parsed_output: str
    ) -> None:
        """Overwrite parsed_output for an annotation (human correction).

        Args:
            annotation_id: Primary key of the annotation.
            parsed_output: The corrected JSON output.
        """
        self._conn.execute(
            "UPDATE annotations SET parsed_output=? WHERE annotation_id=?",
            (parsed_output, annotation_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Review-status mutations
    # ------------------------------------------------------------------

    def update_review_status(self, annotation_id: str, status: str) -> None:
        """Update the review_status of a single annotation.

        Args:
            annotation_id: Primary key of the annotation to update.
            status: New review_status value.
        """
        self._conn.execute(
            "UPDATE annotations SET review_status=? WHERE annotation_id=?",
            (status, annotation_id),
        )
        self._conn.commit()

    def update_review_and_frame_status(
        self,
        annotation_id: str,
        review_status: str,
        frame_id: str,
        frame_status: str,
    ) -> None:
        """Atomically update annotation review_status and frame annotation_status.

        Args:
            annotation_id: Primary key of the annotation to update.
            review_status: New review_status for the annotation.
            frame_id: Primary key of the frame to update.
            frame_status: New annotation_status for the frame.
        """
        self._conn.execute(
            "UPDATE annotations SET review_status=? WHERE annotation_id=?",
            (review_status, annotation_id),
        )
        self._conn.execute(
            "UPDATE frames SET annotation_status=? WHERE frame_id=?",
            (frame_status, frame_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------

    def get_status_counts(self) -> list[sqlite3.Row]:
        """Return frame counts grouped by annotation_status.

        Returns:
            List of sqlite3.Row with columns (annotation_status, count).
        """
        return self._conn.execute(
            "SELECT annotation_status, COUNT(*) AS count FROM frames GROUP BY annotation_status"
        ).fetchall()

    def get_model_stats(self) -> list[sqlite3.Row]:
        """Return per-model annotation statistics including token usage.

        Returns:
            List of sqlite3.Row with columns
            (model_name, annotation_count, total_tokens_sum).
        """
        return self._conn.execute(
            """
            SELECT
                model_name,
                COUNT(*) AS annotation_count,
                SUM(total_tokens) AS total_tokens_sum
            FROM annotations
            GROUP BY model_name
            ORDER BY model_name
            """
        ).fetchall()

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection if this store owns it."""
        if self._owns_conn:
            self._conn.close()

    def __enter__(self) -> AnnotationStore:
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the store on context manager exit."""
        self.close()
