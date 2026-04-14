# pet-annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-model, multi-account async VLM annotation pipeline that consumes pet-data frames and produces SFT/DPO training data for pet-train.

**Architecture:** Unified AnnotationOrchestrator with asyncio+thread pool hybrid. Multiple API providers behind a BaseProvider abstraction, with rate-aware key scheduling. Primary model walks the annotation state machine; secondary models write to a comparison table for side-by-side evaluation.

**Tech Stack:** Python 3.11, asyncio, aiohttp, SQLite (WAL), pet-schema v1.0.0, pydantic, click, tenacity, Label Studio SDK, DVC, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-15-pet-annotation-design.md`

**Upstream code to reference:**
- `pet-schema/src/pet_schema/validator.py` — `validate_output(json_str, version)` returns `ValidationResult(valid, errors, warnings)`
- `pet-schema/src/pet_schema/renderer.py` — `render_prompt(version, few_shot)` returns `tuple[str, str]`
- `pet-data/src/pet_data/storage/store.py` — `FrameStore` pattern (SQLite + WAL + schema.sql on init)
- `pet-data/src/pet_data/storage/schema.sql` — `frames` table DDL with `annotation_status` column

---

## Task 1: Project scaffolding + pyproject.toml + Makefile

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `requirements.in`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/pet_annotation/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pet-annotation"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "pet-schema @ git+https://github.com/Train-Pet-Pipeline/pet-schema.git@v1.0.0",
    "aiohttp>=3.9,<4.0",
    "tenacity>=8.0,<9.0",
    "click>=8.0,<9.0",
    "pydantic>=2.0,<3.0",
    "pyyaml>=6.0,<7.0",
    "label-studio-sdk>=1.0,<2.0",
    "dvc>=3.40,<4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "aioresponses>=0.7", "ruff", "mypy", "pip-tools"]

[project.scripts]
pet-annotation = "pet_annotation.cli:cli"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create `Makefile`**

```makefile
.PHONY: setup test lint clean

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/

clean:
	rm -rf .mypy_cache .pytest_cache __pycache__ *.egg-info dist build
```

- [ ] **Step 3: Create `requirements.in`**

```
pet-schema @ git+https://github.com/Train-Pet-Pipeline/pet-schema.git@v1.0.0
aiohttp>=3.9,<4.0
tenacity>=8.0,<9.0
click>=8.0,<9.0
pydantic>=2.0,<3.0
pyyaml>=6.0,<7.0
label-studio-sdk>=1.0,<2.0
dvc>=3.40,<4.0
```

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.egg-info/
dist/
build/
.mypy_cache/
.pytest_cache/
.ruff_cache/
*.pyc
.env
*.db
exports/
reports/
```

- [ ] **Step 5: Create `.env.example`**

```bash
# pet-annotation environment variables
# Copy to .env and fill in actual values

# Qwen / DashScope API Keys
QWEN_API_KEY_1=sk-xxx
QWEN_API_KEY_2=sk-xxx

# Doubao / Volcengine API Keys
DOUBAO_API_KEY_1=xxx

# Label Studio
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=xxx
```

- [ ] **Step 6: Create `src/pet_annotation/__init__.py`**

```python
"""pet-annotation: VLM annotation pipeline for pet feeder training data."""
```

- [ ] **Step 7: Create `tests/__init__.py` (empty) and `tests/conftest.py`**

`tests/__init__.py` — empty file.

```python
"""Shared test fixtures for pet-annotation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

SCHEMA_SQL = Path(__file__).parent.parent / "migrations" / "001_create_annotation_tables.sql"
# pet-data's schema for the frames table
FRAMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL,
    source          TEXT NOT NULL,
    frame_path      TEXT NOT NULL,
    data_root       TEXT NOT NULL,
    timestamp_ms    INTEGER,
    species         TEXT,
    breed           TEXT,
    lighting        TEXT,
    bowl_type       TEXT,
    quality_flag    TEXT NOT NULL DEFAULT 'normal',
    blur_score      REAL,
    phash           BLOB,
    aug_quality     TEXT,
    aug_seed        INTEGER,
    parent_frame_id TEXT,
    is_anomaly_candidate INTEGER NOT NULL DEFAULT 0,
    anomaly_score   REAL,
    annotation_status TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@pytest.fixture
def db_conn() -> sqlite3.Connection:
    """In-memory SQLite with frames + annotation tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(FRAMES_SCHEMA)
    if SCHEMA_SQL.exists():
        conn.executescript(SCHEMA_SQL.read_text())
    conn.commit()
    return conn
```

- [ ] **Step 8: Run lint to verify scaffolding**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-annotation && make lint`
Expected: PASS (no files to lint yet besides conftest, should be clean)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(pet-annotation): project scaffolding with pyproject.toml, Makefile, conftest"
```

---

## Task 2: Database migration + AnnotationStore

**Files:**
- Create: `migrations/001_create_annotation_tables.sql`
- Create: `src/pet_annotation/store.py`
- Create: `tests/test_store.py`

**Context:** The `AnnotationStore` class opens its own connection to pet-data's SQLite DB. It creates `annotations` and `model_comparisons` tables via `CREATE TABLE IF NOT EXISTS`. It only does `SELECT`/`UPDATE annotation_status` on the `frames` table. Reference `pet-data/src/pet_data/storage/store.py` for the WAL + Row factory pattern.

- [ ] **Step 1: Create migration SQL**

Create `migrations/001_create_annotation_tables.sql`:

```sql
-- pet-annotation tables (owned by pet-annotation, lives in pet-data's SQLite DB)
-- Created via CREATE TABLE IF NOT EXISTS in AnnotationStore.__init__

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id     TEXT PRIMARY KEY,
    frame_id          TEXT NOT NULL REFERENCES frames(frame_id),
    model_name        TEXT NOT NULL,
    prompt_hash       TEXT NOT NULL,
    raw_response      TEXT NOT NULL,
    parsed_output     TEXT,
    schema_valid      INTEGER NOT NULL,
    validation_errors TEXT,
    confidence_overall REAL,
    review_status     TEXT NOT NULL DEFAULT 'pending'
        CHECK(review_status IN ('pending','approved','needs_review','reviewed','rejected')),
    reviewer          TEXT,
    review_notes      TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    api_latency_ms    INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(frame_id, model_name, prompt_hash)
);

CREATE INDEX IF NOT EXISTS idx_annotations_frame  ON annotations(frame_id);
CREATE INDEX IF NOT EXISTS idx_annotations_model  ON annotations(model_name);
CREATE INDEX IF NOT EXISTS idx_annotations_review ON annotations(review_status);
CREATE INDEX IF NOT EXISTS idx_annotations_conf   ON annotations(confidence_overall);

CREATE TABLE IF NOT EXISTS model_comparisons (
    comparison_id     TEXT PRIMARY KEY,
    frame_id          TEXT NOT NULL REFERENCES frames(frame_id),
    model_name        TEXT NOT NULL,
    prompt_hash       TEXT NOT NULL,
    raw_response      TEXT NOT NULL,
    parsed_output     TEXT,
    schema_valid      INTEGER NOT NULL,
    validation_errors TEXT,
    confidence_overall REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    api_latency_ms    INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(frame_id, model_name, prompt_hash)
);

CREATE INDEX IF NOT EXISTS idx_comparisons_frame ON model_comparisons(frame_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_model ON model_comparisons(model_name);
```

- [ ] **Step 2: Write failing tests for AnnotationStore**

Create `tests/test_store.py`:

```python
"""Tests for AnnotationStore."""
from __future__ import annotations

import uuid

import pytest

from pet_annotation.store import AnnotationRecord, AnnotationStore, ComparisonRecord


def _insert_frame(conn, frame_id: str = "frame_001") -> str:
    """Helper: insert a minimal frame row for FK satisfaction."""
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) "
        "VALUES (?, ?, ?, ?, ?)",
        (frame_id, "vid_001", "selfshot", "frames/001.jpg", "/data"),
    )
    conn.commit()
    return frame_id


class TestAnnotationStore:
    def test_init_creates_tables(self, db_conn):
        store = AnnotationStore(db_conn)
        tables = {
            r[0]
            for r in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "annotations" in tables
        assert "model_comparisons" in tables

    def test_insert_annotation(self, db_conn):
        store = AnnotationStore(db_conn)
        fid = _insert_frame(db_conn)
        rec = AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            frame_id=fid,
            model_name="qwen2.5-vl-72b",
            prompt_hash="abc123",
            raw_response='{"schema_version":"1.0"}',
            schema_valid=True,
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            api_latency_ms=1500,
        )
        store.insert_annotation(rec)
        result = store.get_annotation(fid, "qwen2.5-vl-72b", "abc123")
        assert result is not None
        assert result.frame_id == fid

    def test_cache_hit(self, db_conn):
        store = AnnotationStore(db_conn)
        fid = _insert_frame(db_conn)
        assert not store.cache_hit(fid, "qwen2.5-vl-72b", "abc123")
        rec = AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            frame_id=fid,
            model_name="qwen2.5-vl-72b",
            prompt_hash="abc123",
            raw_response="{}",
            schema_valid=True,
        )
        store.insert_annotation(rec)
        assert store.cache_hit(fid, "qwen2.5-vl-72b", "abc123")

    def test_insert_comparison(self, db_conn):
        store = AnnotationStore(db_conn)
        fid = _insert_frame(db_conn)
        rec = ComparisonRecord(
            comparison_id=str(uuid.uuid4()),
            frame_id=fid,
            model_name="doubao-vision",
            prompt_hash="abc123",
            raw_response="{}",
            schema_valid=True,
        )
        store.insert_comparison(rec)
        result = store.get_comparison(fid, "doubao-vision", "abc123")
        assert result is not None

    def test_fetch_pending_frames(self, db_conn):
        store = AnnotationStore(db_conn)
        _insert_frame(db_conn, "f1")
        _insert_frame(db_conn, "f2")
        db_conn.execute(
            "UPDATE frames SET annotation_status='approved' WHERE frame_id='f2'"
        )
        db_conn.commit()
        pending = store.fetch_pending_frames(limit=10)
        assert len(pending) == 1
        assert pending[0]["frame_id"] == "f1"

    def test_update_annotation_status_atomic(self, db_conn):
        """Verify annotation insert + status update happen in one transaction."""
        store = AnnotationStore(db_conn)
        fid = _insert_frame(db_conn)
        rec = AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            frame_id=fid,
            model_name="qwen2.5-vl-72b",
            prompt_hash="abc123",
            raw_response="{}",
            schema_valid=True,
        )
        store.insert_annotation_and_update_status(rec, "auto_checked")
        row = db_conn.execute(
            "SELECT annotation_status FROM frames WHERE frame_id=?", (fid,)
        ).fetchone()
        assert row[0] == "auto_checked"

    def test_recover_annotating_on_init(self, db_conn):
        """Frames stuck in 'annotating' should reset to 'pending' on init."""
        _insert_frame(db_conn, "stuck")
        db_conn.execute(
            "UPDATE frames SET annotation_status='annotating' WHERE frame_id='stuck'"
        )
        db_conn.commit()
        store = AnnotationStore(db_conn)
        row = db_conn.execute(
            "SELECT annotation_status FROM frames WHERE frame_id='stuck'"
        ).fetchone()
        assert row[0] == "pending"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-annotation && pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pet_annotation.store'`

- [ ] **Step 4: Implement `AnnotationStore`**

Create `src/pet_annotation/store.py`:

```python
"""Storage layer for pet-annotation: annotations + model comparisons.

Opens its own connection to pet-data's SQLite DB. Owns the `annotations`
and `model_comparisons` tables. Only SELECT/UPDATE on the `frames` table.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATION_SQL = Path(__file__).parent.parent.parent / "migrations" / "001_create_annotation_tables.sql"


@dataclass
class AnnotationRecord:
    """A single annotation result from the primary model."""

    annotation_id: str
    frame_id: str
    model_name: str
    prompt_hash: str
    raw_response: str
    schema_valid: bool
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
    """A single annotation result from a secondary (comparison) model."""

    comparison_id: str
    frame_id: str
    model_name: str
    prompt_hash: str
    raw_response: str
    schema_valid: bool
    parsed_output: str | None = None
    validation_errors: str | None = None
    confidence_overall: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    api_latency_ms: int | None = None


class AnnotationStore:
    """SQLite-backed store for annotation records.

    Opens its own connection to the same database used by pet-data's FrameStore.
    Creates annotations/model_comparisons tables if they don't exist.
    Recovers frames stuck in 'annotating' state on init.

    Args:
        conn: An existing SQLite connection (for testing with :memory:),
              or pass None and provide db_path.
        db_path: Path to the SQLite database file.
    """

    def __init__(self, conn: sqlite3.Connection | None = None, db_path: Path | None = None) -> None:
        """Initialize store: create tables if needed, recover stuck frames."""
        if conn is not None:
            self._conn = conn
        elif db_path is not None:
            self._conn = sqlite3.connect(str(db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        else:
            msg = "Either conn or db_path must be provided"
            raise ValueError(msg)
        self._conn.row_factory = sqlite3.Row
        self._ensure_tables()
        self._recover_stuck_frames()

    def _ensure_tables(self) -> None:
        """Create annotation tables if they don't exist."""
        if MIGRATION_SQL.exists():
            self._conn.executescript(MIGRATION_SQL.read_text())
            self._conn.commit()

    def _recover_stuck_frames(self) -> None:
        """Reset frames stuck in 'annotating' back to 'pending'."""
        cursor = self._conn.execute(
            "UPDATE frames SET annotation_status='pending' WHERE annotation_status='annotating'"
        )
        if cursor.rowcount > 0:
            self._conn.commit()
            logger.warning(
                '{"event": "recover_stuck_frames", "count": %d}', cursor.rowcount
            )

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def __enter__(self) -> AnnotationStore:
        """Support context manager usage."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on exit."""
        self.close()

    # ------------------------------------------------------------------
    # Frames (read-only + annotation_status update)
    # ------------------------------------------------------------------

    def fetch_pending_frames(self, limit: int = 16) -> list[sqlite3.Row]:
        """Fetch frames with annotation_status='pending'.

        Args:
            limit: Maximum number of frames to return.

        Returns:
            List of frame rows.
        """
        return self._conn.execute(
            "SELECT * FROM frames WHERE annotation_status='pending' "
            "ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()

    def update_frame_status_batch(self, frame_ids: list[str], status: str) -> None:
        """Update annotation_status for a batch of frames.

        Args:
            frame_ids: List of frame IDs to update.
            status: New annotation_status value.
        """
        placeholders = ",".join("?" for _ in frame_ids)
        self._conn.execute(
            f"UPDATE frames SET annotation_status=? WHERE frame_id IN ({placeholders})",
            [status, *frame_ids],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Annotations (primary model)
    # ------------------------------------------------------------------

    def cache_hit(self, frame_id: str, model_name: str, prompt_hash: str) -> bool:
        """Check if an annotation already exists (cache lookup).

        Args:
            frame_id: The frame identifier.
            model_name: The model that produced the annotation.
            prompt_hash: Hash of the prompt used.

        Returns:
            True if a matching record exists.
        """
        row = self._conn.execute(
            "SELECT 1 FROM annotations WHERE frame_id=? AND model_name=? AND prompt_hash=?",
            (frame_id, model_name, prompt_hash),
        ).fetchone()
        return row is not None

    def cache_hit_comparison(self, frame_id: str, model_name: str, prompt_hash: str) -> bool:
        """Check if a comparison record already exists.

        Args:
            frame_id: The frame identifier.
            model_name: The model that produced the annotation.
            prompt_hash: Hash of the prompt used.

        Returns:
            True if a matching record exists.
        """
        row = self._conn.execute(
            "SELECT 1 FROM model_comparisons WHERE frame_id=? AND model_name=? AND prompt_hash=?",
            (frame_id, model_name, prompt_hash),
        ).fetchone()
        return row is not None

    def insert_annotation(self, rec: AnnotationRecord) -> None:
        """Insert an annotation record.

        Args:
            rec: The annotation record to insert.

        Raises:
            sqlite3.IntegrityError: If the unique constraint is violated.
        """
        self._conn.execute(
            "INSERT INTO annotations "
            "(annotation_id, frame_id, model_name, prompt_hash, raw_response, parsed_output, "
            "schema_valid, validation_errors, confidence_overall, review_status, "
            "prompt_tokens, completion_tokens, total_tokens, api_latency_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.annotation_id, rec.frame_id, rec.model_name, rec.prompt_hash,
                rec.raw_response, rec.parsed_output,
                int(rec.schema_valid), rec.validation_errors, rec.confidence_overall,
                rec.review_status,
                rec.prompt_tokens, rec.completion_tokens, rec.total_tokens, rec.api_latency_ms,
            ),
        )
        self._conn.commit()

    def insert_annotation_and_update_status(self, rec: AnnotationRecord, new_status: str) -> None:
        """Insert annotation and update frame status atomically.

        Args:
            rec: The annotation record to insert.
            new_status: New annotation_status for the frame.
        """
        try:
            self._conn.execute(
                "INSERT INTO annotations "
                "(annotation_id, frame_id, model_name, prompt_hash, raw_response, parsed_output, "
                "schema_valid, validation_errors, confidence_overall, review_status, "
                "prompt_tokens, completion_tokens, total_tokens, api_latency_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec.annotation_id, rec.frame_id, rec.model_name, rec.prompt_hash,
                    rec.raw_response, rec.parsed_output,
                    int(rec.schema_valid), rec.validation_errors, rec.confidence_overall,
                    rec.review_status,
                    rec.prompt_tokens, rec.completion_tokens, rec.total_tokens, rec.api_latency_ms,
                ),
            )
            self._conn.execute(
                "UPDATE frames SET annotation_status=? WHERE frame_id=?",
                (new_status, rec.frame_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_annotation(
        self, frame_id: str, model_name: str, prompt_hash: str
    ) -> AnnotationRecord | None:
        """Fetch a single annotation by its cache key.

        Args:
            frame_id: The frame identifier.
            model_name: The model name.
            prompt_hash: The prompt hash.

        Returns:
            AnnotationRecord or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM annotations WHERE frame_id=? AND model_name=? AND prompt_hash=?",
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
            schema_valid=bool(row["schema_valid"]),
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

    # ------------------------------------------------------------------
    # Model comparisons (secondary models)
    # ------------------------------------------------------------------

    def insert_comparison(self, rec: ComparisonRecord) -> None:
        """Insert a comparison record.

        Args:
            rec: The comparison record to insert.
        """
        self._conn.execute(
            "INSERT INTO model_comparisons "
            "(comparison_id, frame_id, model_name, prompt_hash, raw_response, parsed_output, "
            "schema_valid, validation_errors, confidence_overall, "
            "prompt_tokens, completion_tokens, total_tokens, api_latency_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec.comparison_id, rec.frame_id, rec.model_name, rec.prompt_hash,
                rec.raw_response, rec.parsed_output,
                int(rec.schema_valid), rec.validation_errors, rec.confidence_overall,
                rec.prompt_tokens, rec.completion_tokens, rec.total_tokens, rec.api_latency_ms,
            ),
        )
        self._conn.commit()

    def get_comparison(
        self, frame_id: str, model_name: str, prompt_hash: str
    ) -> ComparisonRecord | None:
        """Fetch a single comparison by its cache key.

        Args:
            frame_id: The frame identifier.
            model_name: The model name.
            prompt_hash: The prompt hash.

        Returns:
            ComparisonRecord or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM model_comparisons WHERE frame_id=? AND model_name=? AND prompt_hash=?",
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
            schema_valid=bool(row["schema_valid"]),
            validation_errors=row["validation_errors"],
            confidence_overall=row["confidence_overall"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            api_latency_ms=row["api_latency_ms"],
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def fetch_approved_annotations(self, limit: int = 1000) -> list[sqlite3.Row]:
        """Fetch annotations with review_status in ('approved', 'reviewed').

        Args:
            limit: Maximum number of records.

        Returns:
            List of annotation rows joined with frame data.
        """
        return self._conn.execute(
            "SELECT a.*, f.frame_path, f.data_root, f.source "
            "FROM annotations a JOIN frames f ON a.frame_id = f.frame_id "
            "WHERE a.review_status IN ('approved', 'reviewed') "
            "LIMIT ?",
            (limit,),
        ).fetchall()

    def fetch_comparisons_for_frame(self, frame_id: str) -> list[sqlite3.Row]:
        """Fetch all comparison records for a given frame.

        Args:
            frame_id: The frame identifier.

        Returns:
            List of comparison rows.
        """
        return self._conn.execute(
            "SELECT * FROM model_comparisons WHERE frame_id=?", (frame_id,)
        ).fetchall()
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-annotation && pytest tests/test_store.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add migrations/ src/pet_annotation/store.py tests/test_store.py
git commit -m "feat(pet-annotation): AnnotationStore with annotations + model_comparisons tables"
```

---

## Task 3: Config module (params.yaml loading)

**Files:**
- Create: `params.yaml`
- Create: `src/pet_annotation/config.py`
- Create: `tests/test_config.py`

**Context:** Uses Pydantic Settings to parse `params.yaml`. Each model entry has provider type, base_url, accounts with env var names and rate limits. Key env var names reference `.env` values.

- [ ] **Step 1: Create `params.yaml`**

```yaml
database:
  path: "/data/pet-data/pet_data.db"
  data_root: "/data/pet-data"

annotation:
  batch_size: 16
  max_concurrent: 50
  max_daily_tokens: 10_000_000
  review_sampling_rate: 0.15
  low_confidence_threshold: 0.70
  primary_model: "qwen2.5-vl-72b"
  schema_version: "1.0"

models:
  qwen2.5-vl-72b:
    provider: "openai_compat"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: "qwen2.5-vl-72b-instruct"
    accounts:
      - key_env: "QWEN_API_KEY_1"
        rpm: 60
        tpm: 100000
      - key_env: "QWEN_API_KEY_2"
        rpm: 60
        tpm: 100000
    timeout: 60
    max_retries: 3

  doubao-vision:
    provider: "doubao"
    base_url: "https://ark.cn-beijing.volces.com/api/v3"
    model_name: "doubao-vision-pro-32k"
    accounts:
      - key_env: "DOUBAO_API_KEY_1"
        rpm: 30
        tpm: 50000
    timeout: 60
    max_retries: 3

  local-vllm:
    provider: "vllm"
    base_url: "http://localhost:8000/v1"
    model_name: "Qwen/Qwen2.5-VL-72B-Instruct"
    accounts:
      - key_env: ""
        rpm: 999
        tpm: 999999
    timeout: 120
    max_retries: 2

dpo:
  min_pairs_per_release: 500

dvc:
  remote: "local"
  remote_path: "/data/dvc-cache"
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_config.py`:

```python
"""Tests for config loading."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pet_annotation.config import AccountConfig, AnnotationConfig, ModelConfig, load_config


@pytest.fixture
def minimal_params(tmp_path: Path) -> Path:
    """Create a minimal params.yaml for testing."""
    params = {
        "database": {"path": str(tmp_path / "test.db"), "data_root": str(tmp_path)},
        "annotation": {
            "batch_size": 8,
            "max_concurrent": 10,
            "max_daily_tokens": 1000,
            "review_sampling_rate": 0.15,
            "low_confidence_threshold": 0.70,
            "primary_model": "test-model",
            "schema_version": "1.0",
        },
        "models": {
            "test-model": {
                "provider": "openai_compat",
                "base_url": "http://localhost:8000/v1",
                "model_name": "test",
                "accounts": [{"key_env": "TEST_KEY", "rpm": 10, "tpm": 1000}],
                "timeout": 30,
                "max_retries": 2,
            }
        },
        "dpo": {"min_pairs_per_release": 100},
    }
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(params))
    return p


def test_load_config(minimal_params: Path):
    config = load_config(minimal_params)
    assert config.annotation.batch_size == 8
    assert config.annotation.primary_model == "test-model"
    assert "test-model" in config.models
    assert config.models["test-model"].provider == "openai_compat"


def test_primary_model_must_exist_in_models(tmp_path: Path):
    params = {
        "database": {"path": str(tmp_path / "test.db"), "data_root": str(tmp_path)},
        "annotation": {
            "batch_size": 8, "max_concurrent": 10, "max_daily_tokens": 1000,
            "review_sampling_rate": 0.15, "low_confidence_threshold": 0.70,
            "primary_model": "nonexistent", "schema_version": "1.0",
        },
        "models": {},
        "dpo": {"min_pairs_per_release": 100},
    }
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(params))
    with pytest.raises(ValueError, match="primary_model.*not found"):
        load_config(p)


def test_account_key_resolution(minimal_params: Path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-secret")
    config = load_config(minimal_params)
    account = config.models["test-model"].accounts[0]
    assert account.resolve_key() == "sk-secret"


def test_empty_key_env_returns_empty(minimal_params: Path):
    """VLLMProvider has key_env='' which should resolve to ''."""
    config = load_config(minimal_params)
    # Override account to have empty key_env
    config.models["test-model"].accounts[0].key_env = ""
    assert config.models["test-model"].accounts[0].resolve_key() == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement config module**

Create `src/pet_annotation/config.py`:

```python
"""Configuration loader: params.yaml → typed Pydantic models."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AccountConfig(BaseModel):
    """A single API account (key + rate limits)."""

    key_env: str
    rpm: int
    tpm: int

    def resolve_key(self) -> str:
        """Resolve the API key from environment variable.

        Returns:
            The API key string, or empty string if key_env is empty.
        """
        if not self.key_env:
            return ""
        return os.environ.get(self.key_env, "")


class ModelConfig(BaseModel):
    """Configuration for a single annotation model."""

    provider: str
    base_url: str
    model_name: str
    accounts: list[AccountConfig]
    timeout: int = 60
    max_retries: int = 3


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    path: str
    data_root: str


class AnnotationParams(BaseModel):
    """Annotation parameters from params.yaml."""

    batch_size: int = 16
    max_concurrent: int = 50
    max_daily_tokens: int = 10_000_000
    review_sampling_rate: float = 0.15
    low_confidence_threshold: float = 0.70
    primary_model: str
    schema_version: str = "1.0"


class DpoParams(BaseModel):
    """DPO parameters."""

    min_pairs_per_release: int = 500


class AnnotationConfig(BaseModel):
    """Top-level configuration."""

    database: DatabaseConfig
    annotation: AnnotationParams
    models: dict[str, ModelConfig]
    dpo: DpoParams


def load_config(params_path: Path | None = None) -> AnnotationConfig:
    """Load and validate configuration from params.yaml.

    Args:
        params_path: Path to params.yaml. Defaults to ./params.yaml.

    Returns:
        Validated AnnotationConfig.

    Raises:
        FileNotFoundError: If params.yaml doesn't exist.
        ValueError: If primary_model references a model not in models dict.
    """
    if params_path is None:
        params_path = Path("params.yaml")
    if not params_path.exists():
        msg = f"params.yaml not found: {params_path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(params_path.read_text())
    config = AnnotationConfig(**raw)

    if config.annotation.primary_model not in config.models:
        msg = (
            f"primary_model '{config.annotation.primary_model}' "
            f"not found in models: {list(config.models.keys())}"
        )
        raise ValueError(msg)

    logger.info(
        '{"event": "config_loaded", "models": %s, "primary": "%s"}',
        json.dumps(list(config.models.keys())),
        config.annotation.primary_model,
    )
    return config


def setup_logging() -> None:
    """Configure structured JSON logging for all pet_annotation modules."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
    )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add params.yaml src/pet_annotation/config.py tests/test_config.py
git commit -m "feat(pet-annotation): config module with Pydantic params.yaml loader"
```

---

## Task 4: RateTracker (sliding window rate limiter)

**Files:**
- Create: `src/pet_annotation/teacher/__init__.py`
- Create: `src/pet_annotation/teacher/rate_tracker.py`
- Create: `tests/test_rate_tracker.py`

**Context:** Each model has N API keys, each with RPM/TPM limits. RateTracker uses a 60-second sliding window per key, picks the key with most headroom. When all keys are saturated, it awaits until one frees up.

- [ ] **Step 1: Create `src/pet_annotation/teacher/__init__.py` (empty)**

- [ ] **Step 2: Write failing tests**

Create `tests/test_rate_tracker.py`:

```python
"""Tests for RateTracker."""
from __future__ import annotations

import asyncio
import time

import pytest

from pet_annotation.config import AccountConfig
from pet_annotation.teacher.rate_tracker import RateTracker


@pytest.fixture
def two_keys() -> list[AccountConfig]:
    return [
        AccountConfig(key_env="K1", rpm=2, tpm=10000),
        AccountConfig(key_env="K2", rpm=2, tpm=10000),
    ]


class TestRateTracker:
    async def test_acquire_returns_key(self, two_keys):
        tracker = RateTracker(two_keys, key_resolver=lambda k: k.key_env)
        key = await tracker.acquire()
        assert key in ("K1", "K2")

    async def test_round_robin_when_equal(self, two_keys):
        tracker = RateTracker(two_keys, key_resolver=lambda k: k.key_env)
        k1 = await tracker.acquire()
        tracker.record(k1, tokens=100)
        k2 = await tracker.acquire()
        # After recording on k1, k2 should have more headroom
        assert k2 != k1

    async def test_respects_rpm_limit(self):
        accounts = [AccountConfig(key_env="K1", rpm=1, tpm=999999)]
        tracker = RateTracker(accounts, key_resolver=lambda k: k.key_env)
        k = await tracker.acquire()
        tracker.record(k, tokens=10)
        # Next acquire should wait since RPM=1 and we just used it
        # Use a short timeout to verify it blocks
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(tracker.acquire(), timeout=0.2)

    async def test_prefers_least_loaded_key(self, two_keys):
        tracker = RateTracker(two_keys, key_resolver=lambda k: k.key_env)
        # Load up K1
        tracker.record("K1", tokens=5000)
        # K2 should be preferred
        k = await tracker.acquire()
        assert k == "K2"

    async def test_tpm_awareness(self):
        accounts = [
            AccountConfig(key_env="K1", rpm=100, tpm=100),
            AccountConfig(key_env="K2", rpm=100, tpm=100),
        ]
        tracker = RateTracker(accounts, key_resolver=lambda k: k.key_env)
        # Exhaust K1's TPM
        tracker.record("K1", tokens=95)
        k = await tracker.acquire(estimated_tokens=10)
        assert k == "K2"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_rate_tracker.py -v`
Expected: FAIL

- [ ] **Step 4: Implement RateTracker**

Create `src/pet_annotation/teacher/rate_tracker.py`:

```python
"""Rate-aware API key scheduler with sliding window tracking."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Callable

from pet_annotation.config import AccountConfig

logger = logging.getLogger(__name__)

# Sliding window duration in seconds
WINDOW_SECONDS = 60.0


class _KeyWindow:
    """Sliding window counters for a single API key."""

    def __init__(self, key: str, rpm: int, tpm: int) -> None:
        self.key = key
        self.rpm = rpm
        self.tpm = tpm
        self._request_times: deque[float] = deque()
        self._token_entries: deque[tuple[float, int]] = deque()

    def _prune(self) -> None:
        """Remove entries older than the window."""
        cutoff = time.monotonic() - WINDOW_SECONDS
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        while self._token_entries and self._token_entries[0][0] < cutoff:
            self._token_entries.popleft()

    @property
    def current_rpm(self) -> int:
        """Current requests in the window."""
        self._prune()
        return len(self._request_times)

    @property
    def current_tpm(self) -> int:
        """Current tokens in the window."""
        self._prune()
        return sum(t for _, t in self._token_entries)

    def rpm_headroom(self) -> float:
        """Fraction of RPM remaining (1.0 = fully available)."""
        return max(0.0, 1.0 - self.current_rpm / self.rpm)

    def tpm_headroom(self, estimated_tokens: int = 0) -> float:
        """Fraction of TPM remaining after estimated usage."""
        used = self.current_tpm + estimated_tokens
        return max(0.0, 1.0 - used / self.tpm)

    def can_acquire(self, estimated_tokens: int = 0) -> bool:
        """Check if this key can accept another request."""
        return self.current_rpm < self.rpm and (self.current_tpm + estimated_tokens) < self.tpm

    def record(self, tokens: int) -> None:
        """Record a completed request."""
        now = time.monotonic()
        self._request_times.append(now)
        self._token_entries.append((now, tokens))


class RateTracker:
    """Manages multiple API keys with rate-aware scheduling.

    Selects the key with the most headroom. When all keys are saturated,
    awaits until one becomes available.

    Args:
        accounts: List of account configurations with rate limits.
        key_resolver: Function to extract the actual API key string from an AccountConfig.
    """

    def __init__(
        self,
        accounts: list[AccountConfig],
        key_resolver: Callable[[AccountConfig], str] | None = None,
    ) -> None:
        resolver = key_resolver or (lambda a: a.resolve_key())
        self._windows: dict[str, _KeyWindow] = {}
        for acc in accounts:
            key = resolver(acc)
            self._windows[key] = _KeyWindow(key, acc.rpm, acc.tpm)
        self._event = asyncio.Event()
        self._event.set()

    async def acquire(self, estimated_tokens: int = 0) -> str:
        """Select the API key with the most headroom.

        Blocks until a key is available if all are saturated.

        Args:
            estimated_tokens: Expected token usage for this request.

        Returns:
            The selected API key string.
        """
        while True:
            best_key = None
            best_score = -1.0

            for key, window in self._windows.items():
                if not window.can_acquire(estimated_tokens):
                    continue
                # Score = min(rpm_headroom, tpm_headroom) — bottleneck-aware
                score = min(window.rpm_headroom(), window.tpm_headroom(estimated_tokens))
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key is not None:
                return best_key

            # All keys saturated — wait and retry
            logger.info('{"event": "rate_tracker_waiting", "reason": "all_keys_saturated"}')
            self._event.clear()
            try:
                await asyncio.wait_for(self._event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass  # Retry after timeout

    def record(self, key: str, tokens: int) -> None:
        """Record a completed request's token usage.

        Args:
            key: The API key that was used.
            tokens: Number of tokens consumed.
        """
        if key in self._windows:
            self._windows[key].record(tokens)
        self._event.set()
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_rate_tracker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pet_annotation/teacher/ tests/test_rate_tracker.py
git commit -m "feat(pet-annotation): RateTracker with sliding window rate-aware key scheduling"
```

---

## Task 5: CostTracker (daily token budget)

**Files:**
- Create: `src/pet_annotation/teacher/cost_tracker.py`
- Create: `tests/test_cost_tracker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cost_tracker.py`:

```python
"""Tests for CostTracker."""
from __future__ import annotations

from pet_annotation.teacher.cost_tracker import CostTracker


class TestCostTracker:
    def test_under_limit(self):
        tracker = CostTracker(max_daily_tokens=1000)
        assert tracker.check_and_record(500) is True
        assert tracker.remaining == 500

    def test_at_limit(self):
        tracker = CostTracker(max_daily_tokens=1000)
        tracker.check_and_record(999)
        assert tracker.check_and_record(1) is False

    def test_over_limit(self):
        tracker = CostTracker(max_daily_tokens=100)
        assert tracker.check_and_record(150) is False

    def test_per_model_tracking(self):
        tracker = CostTracker(max_daily_tokens=10000)
        tracker.check_and_record(100, model_name="qwen")
        tracker.check_and_record(200, model_name="doubao")
        stats = tracker.get_stats()
        assert stats["qwen"] == 100
        assert stats["doubao"] == 200
        assert stats["total"] == 300
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cost_tracker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CostTracker**

Create `src/pet_annotation/teacher/cost_tracker.py`:

```python
"""Daily token budget tracker — stops annotation when limit is reached."""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks daily token usage across all models.

    Args:
        max_daily_tokens: Maximum tokens allowed per day.
    """

    def __init__(self, max_daily_tokens: int) -> None:
        self._max = max_daily_tokens
        self._total = 0
        self._per_model: dict[str, int] = defaultdict(int)

    @property
    def remaining(self) -> int:
        """Tokens remaining in the daily budget."""
        return max(0, self._max - self._total)

    def check_and_record(self, tokens: int, model_name: str = "_all") -> bool:
        """Record token usage and check if still within budget.

        Args:
            tokens: Number of tokens consumed.
            model_name: Which model used them.

        Returns:
            True if within budget after recording, False if limit reached.
        """
        if self._total + tokens >= self._max:
            logger.warning(
                '{"event": "daily_token_limit", "total": %d, "max": %d, "model": "%s"}',
                self._total + tokens,
                self._max,
                model_name,
            )
            return False
        self._total += tokens
        self._per_model[model_name] += tokens
        return True

    def get_stats(self) -> dict[str, int]:
        """Return per-model and total token usage.

        Returns:
            Dict with model names as keys and token counts as values,
            plus a 'total' key.
        """
        stats = dict(self._per_model)
        stats["total"] = self._total
        return stats
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cost_tracker.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pet_annotation/teacher/cost_tracker.py tests/test_cost_tracker.py
git commit -m "feat(pet-annotation): CostTracker with daily token budget enforcement"
```

---

## Task 6: Provider abstraction + OpenAICompatProvider

**Files:**
- Create: `src/pet_annotation/teacher/provider.py`
- Create: `src/pet_annotation/teacher/providers/__init__.py`
- Create: `src/pet_annotation/teacher/providers/openai_compat.py`
- Create: `tests/test_provider.py`

**Context:** `BaseProvider` is an ABC with `async annotate(image_path, prompt, api_key) -> ProviderResult`. `OpenAICompatProvider` covers Qwen/DashScope and any OpenAI-compatible endpoint. Image is base64-encoded in the message content.

- [ ] **Step 1: Write failing tests**

Create `tests/test_provider.py`:

```python
"""Tests for Provider abstraction and OpenAICompatProvider."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from aioresponses import aioresponses

from pet_annotation.teacher.provider import BaseProvider, PromptPair, ProviderResult
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


class TestProviderResult:
    def test_dataclass_fields(self):
        r = ProviderResult(
            raw_response='{"test": true}',
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=150,
        )
        assert r.total_tokens == 30


class TestOpenAICompatProvider:
    @pytest.fixture
    def provider(self):
        return OpenAICompatProvider(
            base_url="http://test-api.example.com/v1",
            model_name="test-model",
            timeout=30,
            max_retries=1,
        )

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> Path:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
        return img

    async def test_annotate_success(self, provider, sample_image):
        prompt: PromptPair = ("system prompt", "user prompt")
        mock_response = {
            "choices": [{"message": {"content": '{"schema_version": "1.0"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        with aioresponses() as m:
            m.post("http://test-api.example.com/v1/chat/completions", payload=mock_response)
            result = await provider.annotate(str(sample_image), prompt, api_key="sk-test")

        assert result.raw_response == '{"schema_version": "1.0"}'
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.latency_ms > 0

    async def test_annotate_includes_image_as_base64(self, provider, sample_image):
        prompt: PromptPair = ("sys", "usr")
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with aioresponses() as m:
            m.post("http://test-api.example.com/v1/chat/completions", payload=mock_response,
                   repeat=True)
            await provider.annotate(str(sample_image), prompt, api_key="sk-test")
            # Verify the request was made (aioresponses tracks calls)
            assert len(m.requests) == 1

    def test_supports_batch(self, provider):
        assert provider.supports_batch() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_provider.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BaseProvider and ProviderResult**

Create `src/pet_annotation/teacher/provider.py`:

```python
"""Provider abstraction for VLM API annotation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Type alias matching pet_schema.render_prompt() return type
PromptPair = tuple[str, str]  # (system_prompt, user_prompt)


@dataclass
class ProviderResult:
    """Result from a single annotation API call.

    Attributes:
        raw_response: Raw JSON string from the model's response content.
        prompt_tokens: Number of prompt tokens used.
        completion_tokens: Number of completion tokens generated.
        latency_ms: Request round-trip time in milliseconds.
    """

    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens


class BaseProvider(ABC):
    """Abstract base class for VLM annotation providers.

    All providers must implement annotate() for single-frame inference.
    """

    @abstractmethod
    async def annotate(
        self, image_path: str, prompt: PromptPair, api_key: str
    ) -> ProviderResult:
        """Send a single frame for annotation.

        Args:
            image_path: Absolute path to the frame image.
            prompt: Tuple of (system_prompt, user_prompt).
            api_key: API key for authentication.

        Returns:
            ProviderResult with the model's response and usage stats.

        Raises:
            aiohttp.ClientError: On network errors (retriable).
            asyncio.TimeoutError: On request timeout (retriable).
        """

    @abstractmethod
    def supports_batch(self) -> bool:
        """Whether this provider supports native batch API.

        Returns:
            True if batch endpoint is available.
        """
```

- [ ] **Step 4: Implement OpenAICompatProvider**

Create `src/pet_annotation/teacher/providers/__init__.py` (empty).

Create `src/pet_annotation/teacher/providers/openai_compat.py`:

```python
"""OpenAI-compatible API provider for VLM annotation.

Covers: DashScope (Qwen), any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pet_annotation.teacher.provider import BaseProvider, PromptPair, ProviderResult

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseProvider):
    """Provider for OpenAI-compatible chat completion endpoints.

    Sends images as base64-encoded content parts in the user message.

    Args:
        base_url: Base URL for the API (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1).
        model_name: Model identifier for the API.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts on transient errors.
    """

    def __init__(
        self, base_url: str, model_name: str, timeout: int = 60, max_retries: int = 3
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries

    async def annotate(
        self, image_path: str, prompt: PromptPair, api_key: str
    ) -> ProviderResult:
        """Send a single frame for annotation via chat completions endpoint.

        Args:
            image_path: Path to the image file.
            prompt: (system_prompt, user_prompt) tuple.
            api_key: API key for Authorization header.

        Returns:
            ProviderResult with parsed response.
        """
        system_prompt, user_prompt = prompt
        image_b64 = self._encode_image(image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{self._base_url}/chat/completions"

        return await self._call_api(url, payload, headers)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
    )
    async def _call_api(
        self, url: str, payload: dict, headers: dict
    ) -> ProviderResult:
        """Make the API call with retry logic.

        Args:
            url: Full endpoint URL.
            payload: Request body.
            headers: HTTP headers.

        Returns:
            ProviderResult.
        """
        start = time.monotonic()
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()

        latency_ms = int((time.monotonic() - start) * 1000)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return ProviderResult(
            raw_response=content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read and base64-encode an image file.

        Args:
            image_path: Path to the image.

        Returns:
            Base64-encoded string.
        """
        return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    def supports_batch(self) -> bool:
        """OpenAI-compat endpoints don't support native batch."""
        return False
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_provider.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pet_annotation/teacher/provider.py src/pet_annotation/teacher/providers/ tests/test_provider.py
git commit -m "feat(pet-annotation): BaseProvider abstraction + OpenAICompatProvider"
```

---

## Task 7: DoubaoProvider + VLLMProvider

**Files:**
- Create: `src/pet_annotation/teacher/providers/doubao.py`
- Create: `src/pet_annotation/teacher/providers/vllm.py`
- Modify: `tests/test_provider.py` (append new test classes)

**Context:** DoubaoProvider follows Volcengine's API format. VLLMProvider is nearly identical to OpenAICompatProvider but skips auth header when key is empty.

- [ ] **Step 1: Write failing tests — append to `tests/test_provider.py`**

```python
from pet_annotation.teacher.providers.doubao import DoubaoProvider
from pet_annotation.teacher.providers.vllm import VLLMProvider


class TestDoubaoProvider:
    async def test_annotate_success(self, tmp_path):
        provider = DoubaoProvider(
            base_url="http://doubao-api.example.com/api/v3",
            model_name="doubao-vision-pro",
            timeout=30,
            max_retries=1,
        )
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 50)
        mock_response = {
            "choices": [{"message": {"content": '{"test": true}'}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30},
        }
        with aioresponses() as m:
            m.post("http://doubao-api.example.com/api/v3/chat/completions", payload=mock_response)
            result = await provider.annotate(str(img), ("sys", "usr"), api_key="test-key")
        assert result.raw_response == '{"test": true}'

    def test_supports_batch(self):
        provider = DoubaoProvider("http://x", "m")
        assert provider.supports_batch() is False


class TestVLLMProvider:
    async def test_annotate_no_auth_header(self, tmp_path):
        provider = VLLMProvider(
            base_url="http://localhost:8000/v1",
            model_name="local-model",
            timeout=30,
            max_retries=1,
        )
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 50)
        mock_response = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        with aioresponses() as m:
            m.post("http://localhost:8000/v1/chat/completions", payload=mock_response)
            result = await provider.annotate(str(img), ("sys", "usr"), api_key="")
        assert result.prompt_tokens == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_provider.py::TestDoubaoProvider -v`
Expected: FAIL

- [ ] **Step 3: Implement DoubaoProvider**

Create `src/pet_annotation/teacher/providers/doubao.py`:

```python
"""Doubao (Volcengine) VLM provider.

Uses the Volcengine ARK API which is OpenAI-compatible with minor differences.
"""
from __future__ import annotations

from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


class DoubaoProvider(OpenAICompatProvider):
    """Provider for Doubao/Volcengine vision models.

    Inherits from OpenAICompatProvider since the Volcengine ARK API
    is OpenAI-compatible for chat completions.

    Args:
        base_url: Volcengine ARK API base URL.
        model_name: Doubao model identifier (endpoint ID).
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
    """

    pass  # Protocol is compatible; override only if Volcengine diverges
```

- [ ] **Step 4: Implement VLLMProvider**

Create `src/pet_annotation/teacher/providers/vllm.py`:

```python
"""Self-hosted vLLM provider.

Uses OpenAI-compatible API served by vLLM. Skips auth when key is empty.
"""
from __future__ import annotations

from pet_annotation.teacher.provider import PromptPair, ProviderResult
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider


class VLLMProvider(OpenAICompatProvider):
    """Provider for self-hosted vLLM instances.

    Inherits OpenAI-compatible protocol. Skips Authorization header
    when api_key is empty (local deployment).

    Args:
        base_url: vLLM server URL (e.g. http://localhost:8000/v1).
        model_name: Model name as served by vLLM.
        timeout: Request timeout in seconds (default higher for local).
        max_retries: Maximum retry attempts.
    """

    def __init__(
        self, base_url: str, model_name: str, timeout: int = 120, max_retries: int = 2
    ) -> None:
        super().__init__(base_url, model_name, timeout, max_retries)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_provider.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pet_annotation/teacher/providers/doubao.py src/pet_annotation/teacher/providers/vllm.py tests/test_provider.py
git commit -m "feat(pet-annotation): DoubaoProvider + VLLMProvider implementations"
```

---

## Task 8: ProviderRegistry

**Files:**
- Modify: `src/pet_annotation/teacher/provider.py` (add ProviderRegistry class)
- Create: `tests/test_registry.py`

**Context:** ProviderRegistry reads config, instantiates the correct Provider subclass for each model, and pairs it with a RateTracker. Provides `get_primary()` and `get_all()`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_registry.py`:

```python
"""Tests for ProviderRegistry."""
from __future__ import annotations

import pytest

from pet_annotation.config import AccountConfig, AnnotationConfig, AnnotationParams, DatabaseConfig, DpoParams, ModelConfig
from pet_annotation.teacher.provider import ProviderRegistry
from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider
from pet_annotation.teacher.providers.doubao import DoubaoProvider
from pet_annotation.teacher.providers.vllm import VLLMProvider


@pytest.fixture
def config() -> AnnotationConfig:
    return AnnotationConfig(
        database=DatabaseConfig(path=":memory:", data_root="/tmp"),
        annotation=AnnotationParams(
            batch_size=8, max_concurrent=10, max_daily_tokens=1000,
            review_sampling_rate=0.15, low_confidence_threshold=0.70,
            primary_model="model-a", schema_version="1.0",
        ),
        models={
            "model-a": ModelConfig(
                provider="openai_compat", base_url="http://a/v1", model_name="a",
                accounts=[AccountConfig(key_env="K1", rpm=10, tpm=1000)],
            ),
            "model-b": ModelConfig(
                provider="doubao", base_url="http://b/v1", model_name="b",
                accounts=[AccountConfig(key_env="K2", rpm=10, tpm=1000)],
            ),
            "model-c": ModelConfig(
                provider="vllm", base_url="http://c/v1", model_name="c",
                accounts=[AccountConfig(key_env="", rpm=999, tpm=999999)],
            ),
        },
        dpo=DpoParams(min_pairs_per_release=100),
    )


class TestProviderRegistry:
    def test_get_primary(self, config):
        reg = ProviderRegistry(config)
        name, provider, tracker = reg.get_primary()
        assert name == "model-a"
        assert isinstance(provider, OpenAICompatProvider)

    def test_get_all(self, config):
        reg = ProviderRegistry(config)
        all_providers = reg.get_all()
        assert len(all_providers) == 3
        names = {n for n, _, _ in all_providers}
        assert names == {"model-a", "model-b", "model-c"}

    def test_provider_types(self, config):
        reg = ProviderRegistry(config)
        types = {n: type(p).__name__ for n, p, _ in reg.get_all()}
        assert types["model-a"] == "OpenAICompatProvider"
        assert types["model-b"] == "DoubaoProvider"
        assert types["model-c"] == "VLLMProvider"

    def test_unknown_provider_raises(self):
        config = AnnotationConfig(
            database=DatabaseConfig(path=":memory:", data_root="/tmp"),
            annotation=AnnotationParams(
                primary_model="bad", schema_version="1.0",
            ),
            models={
                "bad": ModelConfig(
                    provider="unknown_type", base_url="http://x", model_name="x",
                    accounts=[AccountConfig(key_env="K", rpm=1, tpm=1)],
                ),
            },
            dpo=DpoParams(),
        )
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderRegistry(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ProviderRegistry**

Append to `src/pet_annotation/teacher/provider.py`:

```python
import logging

from pet_annotation.config import AnnotationConfig
from pet_annotation.teacher.rate_tracker import RateTracker

logger = logging.getLogger(__name__)

# Provider type registry — maps provider string to class
_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a provider class by name.

    Args:
        name: Provider type string (matches params.yaml 'provider' field).
        cls: The provider class.
    """
    _PROVIDER_CLASSES[name] = cls


class ProviderRegistry:
    """Instantiates and manages providers + rate trackers from config.

    Args:
        config: The loaded AnnotationConfig.
    """

    def __init__(self, config: AnnotationConfig) -> None:
        # Lazy import to avoid circular imports
        from pet_annotation.teacher.providers.openai_compat import OpenAICompatProvider
        from pet_annotation.teacher.providers.doubao import DoubaoProvider
        from pet_annotation.teacher.providers.vllm import VLLMProvider

        register_provider("openai_compat", OpenAICompatProvider)
        register_provider("doubao", DoubaoProvider)
        register_provider("vllm", VLLMProvider)

        self._primary_name = config.annotation.primary_model
        self._entries: dict[str, tuple[BaseProvider, RateTracker]] = {}

        for model_name, model_cfg in config.models.items():
            cls = _PROVIDER_CLASSES.get(model_cfg.provider)
            if cls is None:
                msg = f"Unknown provider type: '{model_cfg.provider}' for model '{model_name}'"
                raise ValueError(msg)

            provider = cls(
                base_url=model_cfg.base_url,
                model_name=model_cfg.model_name,
                timeout=model_cfg.timeout,
                max_retries=model_cfg.max_retries,
            )
            tracker = RateTracker(model_cfg.accounts)
            self._entries[model_name] = (provider, tracker)

    def get_primary(self) -> tuple[str, BaseProvider, RateTracker]:
        """Return the primary model's (name, provider, tracker).

        Returns:
            Tuple of (model_name, provider_instance, rate_tracker).
        """
        provider, tracker = self._entries[self._primary_name]
        return self._primary_name, provider, tracker

    def get_all(self) -> list[tuple[str, BaseProvider, RateTracker]]:
        """Return all configured models.

        Returns:
            List of (model_name, provider_instance, rate_tracker) tuples.
        """
        return [(name, p, t) for name, (p, t) in self._entries.items()]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_registry.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pet_annotation/teacher/provider.py tests/test_registry.py
git commit -m "feat(pet-annotation): ProviderRegistry maps config to provider + tracker instances"
```

---

## Task 9: Orchestrator

**Files:**
- Create: `src/pet_annotation/teacher/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Context:** This is the core engine. Pulls pending frames, dispatches to all models concurrently via asyncio, writes results (primary → annotations table, secondary → model_comparisons), updates frame status. Uses Semaphore for concurrency control, run_in_executor for DB ops.

- [ ] **Step 1: Write failing tests**

Create `tests/test_orchestrator.py`:

```python
"""Tests for AnnotationOrchestrator."""
from __future__ import annotations

import asyncio
import sqlite3
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pet_annotation.config import (
    AccountConfig, AnnotationConfig, AnnotationParams,
    DatabaseConfig, DpoParams, ModelConfig,
)
from pet_annotation.store import AnnotationStore
from pet_annotation.teacher.cost_tracker import CostTracker
from pet_annotation.teacher.orchestrator import AnnotationOrchestrator
from pet_annotation.teacher.provider import ProviderResult


def _make_config(db_path: str = ":memory:") -> AnnotationConfig:
    return AnnotationConfig(
        database=DatabaseConfig(path=db_path, data_root="/data"),
        annotation=AnnotationParams(
            batch_size=2, max_concurrent=5, max_daily_tokens=100000,
            review_sampling_rate=0.0, low_confidence_threshold=0.70,
            primary_model="primary", schema_version="1.0",
        ),
        models={
            "primary": ModelConfig(
                provider="openai_compat", base_url="http://a/v1", model_name="a",
                accounts=[AccountConfig(key_env="K1", rpm=100, tpm=999999)],
            ),
        },
        dpo=DpoParams(min_pairs_per_release=100),
    )


def _insert_frames(conn: sqlite3.Connection, count: int) -> list[str]:
    ids = []
    for i in range(count):
        fid = f"frame_{i:03d}"
        conn.execute(
            "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) "
            "VALUES (?, ?, ?, ?, ?)",
            (fid, "v1", "selfshot", f"frames/{fid}.jpg", "/data"),
        )
        ids.append(fid)
    conn.commit()
    return ids


class TestOrchestrator:
    async def test_processes_pending_frames(self, db_conn):
        config = _make_config()
        store = AnnotationStore(db_conn)
        _insert_frames(db_conn, 3)

        mock_result = ProviderResult(
            raw_response='{"schema_version":"1.0","pet_present":false,"pet_count":0,'
            '"pet":null,"bowl":{"food_fill_ratio":0.5,"water_fill_ratio":null,'
            '"food_type_visible":"dry"},"scene":{"lighting":"bright",'
            '"image_quality":"clear","confidence_overall":0.9},"narrative":"test"}',
            prompt_tokens=100, completion_tokens=50, latency_ms=500,
        )

        orch = AnnotationOrchestrator(config=config, store=store)

        with patch.object(orch, "_call_provider", new_callable=AsyncMock, return_value=mock_result):
            with patch("pet_annotation.teacher.orchestrator.render_prompt", return_value=("sys", "usr")):
                await orch.run()

        # All 3 frames should now be auto_checked
        rows = db_conn.execute(
            "SELECT annotation_status FROM frames ORDER BY frame_id"
        ).fetchall()
        assert all(r[0] == "auto_checked" for r in rows)

    async def test_skips_cached_frames(self, db_conn):
        config = _make_config()
        store = AnnotationStore(db_conn)
        _insert_frames(db_conn, 1)

        # Pre-insert a cached annotation
        from pet_annotation.store import AnnotationRecord
        store.insert_annotation(AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            frame_id="frame_000",
            model_name="primary",
            prompt_hash="test_hash",
            raw_response="{}",
            schema_valid=True,
        ))

        call_count = 0
        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return ProviderResult("{}", 10, 5, 100)

        orch = AnnotationOrchestrator(config=config, store=store)
        with patch.object(orch, "_call_provider", side_effect=mock_call):
            with patch("pet_annotation.teacher.orchestrator.render_prompt", return_value=("sys", "usr")):
                with patch("pet_annotation.teacher.orchestrator.compute_prompt_hash", return_value="test_hash"):
                    await orch.run()

        assert call_count == 0  # Should have been skipped via cache

    async def test_failed_frame_reverts_to_pending(self, db_conn):
        config = _make_config()
        store = AnnotationStore(db_conn)
        _insert_frames(db_conn, 1)

        async def mock_fail(*args, **kwargs):
            raise RuntimeError("API error")

        orch = AnnotationOrchestrator(config=config, store=store)
        with patch.object(orch, "_call_provider", side_effect=mock_fail):
            with patch("pet_annotation.teacher.orchestrator.render_prompt", return_value=("sys", "usr")):
                await orch.run()

        row = db_conn.execute(
            "SELECT annotation_status FROM frames WHERE frame_id='frame_000'"
        ).fetchone()
        assert row[0] == "pending"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Orchestrator**

Create `src/pet_annotation/teacher/orchestrator.py`:

```python
"""Annotation orchestrator — the core batch annotation engine.

Pulls pending frames, dispatches to all configured models concurrently,
validates results, writes to DB, and advances the annotation state machine.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pet_schema import render_prompt, validate_output

from pet_annotation.config import AnnotationConfig, load_config
from pet_annotation.store import AnnotationRecord, AnnotationStore, ComparisonRecord
from pet_annotation.teacher.cost_tracker import CostTracker
from pet_annotation.teacher.provider import BaseProvider, PromptPair, ProviderRegistry, ProviderResult
from pet_annotation.teacher.rate_tracker import RateTracker

logger = logging.getLogger(__name__)


def compute_prompt_hash(system_prompt: str, user_prompt: str, schema_version: str) -> str:
    """Compute cache key hash from prompt contents.

    Uses the raw template text (not rendered output) + schema version.

    Args:
        system_prompt: System prompt text.
        user_prompt: User prompt template text.
        schema_version: Schema version string.

    Returns:
        SHA-256 hex digest.
    """
    content = f"{system_prompt}|{user_prompt}|{schema_version}"
    return hashlib.sha256(content.encode()).hexdigest()


class AnnotationOrchestrator:
    """Orchestrates multi-model annotation with async concurrency.

    Pulls pending frames in batches, dispatches each frame to all configured
    models concurrently, validates results with pet-schema, writes to DB,
    and advances frames through the annotation state machine.

    Args:
        config: Loaded annotation configuration.
        store: AnnotationStore instance.
    """

    def __init__(self, config: AnnotationConfig, store: AnnotationStore) -> None:
        self._config = config
        self._store = store
        self._registry = ProviderRegistry(config)
        self._cost_tracker = CostTracker(config.annotation.max_daily_tokens)
        self._semaphore = asyncio.Semaphore(config.annotation.max_concurrent)
        self._db_executor = ThreadPoolExecutor(max_workers=1)
        self._shutdown = False

    async def run(self) -> dict:
        """Main entry point: process all pending frames in batches.

        Returns:
            Stats dict with counts of processed/skipped/failed frames.
        """
        self._setup_signal_handlers()
        prompt = render_prompt(version=self._config.annotation.schema_version)
        prompt_hash = compute_prompt_hash(prompt[0], prompt[1], self._config.annotation.schema_version)
        batch_size = self._config.annotation.batch_size

        stats = {"processed": 0, "skipped": 0, "failed": 0}

        while not self._shutdown:
            frames = await self._run_in_executor(
                self._store.fetch_pending_frames, batch_size
            )
            if not frames:
                break

            frame_ids = [f["frame_id"] for f in frames]
            await self._run_in_executor(
                self._store.update_frame_status_batch, frame_ids, "annotating"
            )

            results = await asyncio.gather(
                *[self._process_frame(f, prompt, prompt_hash) for f in frames],
                return_exceptions=True,
            )

            for frame, result in zip(frames, results):
                fid = frame["frame_id"]
                if isinstance(result, Exception):
                    logger.error(
                        '{"event": "frame_failed", "frame_id": "%s", "error": "%s"}',
                        fid, str(result),
                    )
                    await self._run_in_executor(
                        self._store.update_frame_status_batch, [fid], "pending"
                    )
                    stats["failed"] += 1
                elif result == "skipped":
                    await self._run_in_executor(
                        self._store.update_frame_status_batch, [fid], "auto_checked"
                    )
                    stats["skipped"] += 1
                else:
                    await self._run_in_executor(
                        self._store.update_frame_status_batch, [fid], "auto_checked"
                    )
                    stats["processed"] += 1

            if not self._cost_tracker.check_and_record(0):
                logger.warning('{"event": "daily_limit_reached"}')
                break

        logger.info('{"event": "orchestrator_done", "stats": %s}', json.dumps(stats))
        return stats

    async def _process_frame(
        self, frame: sqlite3.Row, prompt: PromptPair, prompt_hash: str
    ) -> str:
        """Process a single frame across all configured models.

        Args:
            frame: Frame row from DB.
            prompt: (system, user) prompt pair.
            prompt_hash: Cache key hash.

        Returns:
            'processed' or 'skipped'.
        """
        image_path = str(Path(frame["data_root"]) / frame["frame_path"])
        primary_name = self._config.annotation.primary_model

        tasks = []
        for name, provider, tracker in self._registry.get_all():
            is_primary = name == primary_name
            tasks.append(
                self._annotate_one(
                    frame["frame_id"], name, provider, tracker,
                    image_path, prompt, prompt_hash, is_primary,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # If primary model failed, propagate the exception
        primary_idx = next(
            i for i, (n, _, _) in enumerate(self._registry.get_all()) if n == primary_name
        )
        if isinstance(results[primary_idx], Exception):
            raise results[primary_idx]

        return results[primary_idx]

    async def _annotate_one(
        self,
        frame_id: str,
        model_name: str,
        provider: BaseProvider,
        tracker: RateTracker,
        image_path: str,
        prompt: PromptPair,
        prompt_hash: str,
        is_primary: bool,
    ) -> str:
        """Annotate a single frame with a single model.

        Args:
            frame_id: Frame identifier.
            model_name: Model name.
            provider: Provider instance.
            tracker: Rate tracker for this model.
            image_path: Full path to the image.
            prompt: Prompt pair.
            prompt_hash: Cache key hash.
            is_primary: Whether this is the primary model.

        Returns:
            'processed' or 'skipped'.
        """
        async with self._semaphore:
            # Cache check
            if is_primary:
                hit = await self._run_in_executor(
                    self._store.cache_hit, frame_id, model_name, prompt_hash
                )
            else:
                hit = await self._run_in_executor(
                    self._store.cache_hit_comparison, frame_id, model_name, prompt_hash
                )
            if hit:
                return "skipped"

            # Get API key
            key = await tracker.acquire(estimated_tokens=2000)

            # Call API
            result = await self._call_provider(provider, image_path, prompt, key)
            tracker.record(key, result.total_tokens)

            # Cost tracking
            self._cost_tracker.check_and_record(result.total_tokens, model_name)

            # Validate
            validation = validate_output(result.raw_response, self._config.annotation.schema_version)

            # Extract confidence_overall from parsed output
            confidence = None
            parsed = None
            if validation.valid:
                try:
                    data = json.loads(result.raw_response)
                    confidence = data.get("scene", {}).get("confidence_overall")
                    parsed = result.raw_response
                except json.JSONDecodeError:
                    pass

            errors_json = json.dumps(validation.errors) if validation.errors else None

            # Write to DB
            if is_primary:
                rec = AnnotationRecord(
                    annotation_id=str(uuid.uuid4()),
                    frame_id=frame_id,
                    model_name=model_name,
                    prompt_hash=prompt_hash,
                    raw_response=result.raw_response,
                    parsed_output=parsed,
                    schema_valid=validation.valid,
                    validation_errors=errors_json,
                    confidence_overall=confidence,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    api_latency_ms=result.latency_ms,
                )
                await self._run_in_executor(self._store.insert_annotation, rec)
            else:
                rec = ComparisonRecord(
                    comparison_id=str(uuid.uuid4()),
                    frame_id=frame_id,
                    model_name=model_name,
                    prompt_hash=prompt_hash,
                    raw_response=result.raw_response,
                    parsed_output=parsed,
                    schema_valid=validation.valid,
                    validation_errors=errors_json,
                    confidence_overall=confidence,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                    api_latency_ms=result.latency_ms,
                )
                await self._run_in_executor(self._store.insert_comparison, rec)

            logger.info(
                '{"event": "annotated", "frame_id": "%s", "model": "%s", '
                '"valid": %s, "tokens": %d, "latency_ms": %d}',
                frame_id, model_name, str(validation.valid).lower(),
                result.total_tokens, result.latency_ms,
            )
            return "processed"

    async def _call_provider(
        self, provider: BaseProvider, image_path: str, prompt: PromptPair, api_key: str
    ) -> ProviderResult:
        """Call provider's annotate method. Separated for easy mocking in tests.

        Args:
            provider: The provider instance.
            image_path: Path to the image.
            prompt: Prompt pair.
            api_key: API key.

        Returns:
            ProviderResult.
        """
        return await provider.annotate(image_path, prompt, api_key)

    async def _run_in_executor(self, fn, *args):
        """Run a sync function in the DB thread pool.

        Args:
            fn: Synchronous function to call.
            *args: Arguments to pass.

        Returns:
            The function's return value.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._db_executor, fn, *args)

    def _setup_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                pass  # Windows

    def _handle_shutdown(self) -> None:
        """Set shutdown flag to stop after current batch."""
        logger.info('{"event": "shutdown_requested"}')
        self._shutdown = True


async def main() -> None:
    """CLI entry point for the annotate command."""
    from pet_annotation.config import setup_logging
    setup_logging()
    config = load_config()
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        orch = AnnotationOrchestrator(config=config, store=store)
        stats = await orch.run()
        print(json.dumps(stats, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_orchestrator.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pet_annotation/teacher/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(pet-annotation): AnnotationOrchestrator — core async batch annotation engine"
```

---

## Task 10: Quality module (auto_check + sampling)

**Files:**
- Create: `src/pet_annotation/quality/__init__.py`
- Create: `src/pet_annotation/quality/auto_check.py`
- Create: `src/pet_annotation/quality/sampling.py`
- Create: `tests/test_auto_check.py`
- Create: `tests/test_sampling.py`

- [ ] **Step 1: Write failing tests for sampling.py**

Create `tests/test_sampling.py`:

```python
"""Tests for sampling module."""
from __future__ import annotations

from unittest.mock import patch

from pet_annotation.quality.sampling import decide_review


class TestDecideReview:
    def test_invalid_schema_forces_review(self):
        assert decide_review(schema_valid=False, confidence=0.95, sampling_rate=0.0, threshold=0.70) == "needs_review"

    def test_low_confidence_forces_review(self):
        assert decide_review(schema_valid=True, confidence=0.50, sampling_rate=0.0, threshold=0.70) == "needs_review"

    def test_random_sampling(self):
        with patch("pet_annotation.quality.sampling.random.random", return_value=0.05):
            assert decide_review(schema_valid=True, confidence=0.90, sampling_rate=0.15, threshold=0.70) == "needs_review"

    def test_approved_when_passing(self):
        with patch("pet_annotation.quality.sampling.random.random", return_value=0.99):
            assert decide_review(schema_valid=True, confidence=0.90, sampling_rate=0.15, threshold=0.70) == "approved"
```

- [ ] **Step 2: Write failing tests for auto_check.py**

Create `tests/test_auto_check.py`:

```python
"""Tests for auto_check module."""
from __future__ import annotations

import uuid

import pytest

from pet_annotation.quality.auto_check import run_auto_check
from pet_annotation.store import AnnotationRecord, AnnotationStore


def _insert_frame(conn, frame_id: str = "f1"):
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) VALUES (?,?,?,?,?)",
        (frame_id, "v1", "selfshot", "f.jpg", "/data"),
    )
    conn.commit()


class TestAutoCheck:
    def test_approved_high_confidence(self, db_conn):
        store = AnnotationStore(db_conn)
        _insert_frame(db_conn, "f1")
        rec = AnnotationRecord(
            annotation_id=str(uuid.uuid4()), frame_id="f1",
            model_name="primary", prompt_hash="h1", raw_response="{}",
            schema_valid=True, confidence_overall=0.90,
        )
        store.insert_annotation(rec)
        store.update_frame_status_batch(["f1"], "auto_checked")

        run_auto_check(store, sampling_rate=0.0, threshold=0.70, primary_model="primary")

        ann = store.get_annotation("f1", "primary", "h1")
        assert ann.review_status == "approved"

    def test_needs_review_low_confidence(self, db_conn):
        store = AnnotationStore(db_conn)
        _insert_frame(db_conn, "f1")
        rec = AnnotationRecord(
            annotation_id=str(uuid.uuid4()), frame_id="f1",
            model_name="primary", prompt_hash="h1", raw_response="{}",
            schema_valid=True, confidence_overall=0.50,
        )
        store.insert_annotation(rec)
        store.update_frame_status_batch(["f1"], "auto_checked")

        run_auto_check(store, sampling_rate=0.0, threshold=0.70, primary_model="primary")

        ann = store.get_annotation("f1", "primary", "h1")
        assert ann.review_status == "needs_review"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_sampling.py tests/test_auto_check.py -v`
Expected: FAIL

- [ ] **Step 4: Implement sampling.py**

Create `src/pet_annotation/quality/__init__.py` (empty).

Create `src/pet_annotation/quality/sampling.py`:

```python
"""Sampling strategy for human review routing."""
from __future__ import annotations

import random


def decide_review(
    schema_valid: bool,
    confidence: float | None,
    sampling_rate: float,
    threshold: float,
) -> str:
    """Decide whether a frame needs human review.

    Args:
        schema_valid: Whether the annotation passed schema validation.
        confidence: confidence_overall value (0-1), or None.
        sampling_rate: Random sampling rate (0-1).
        threshold: Minimum confidence to auto-approve.

    Returns:
        'approved' or 'needs_review'.
    """
    if not schema_valid:
        return "needs_review"
    if confidence is not None and confidence < threshold:
        return "needs_review"
    if random.random() < sampling_rate:
        return "needs_review"
    return "approved"
```

- [ ] **Step 5: Implement auto_check.py**

Create `src/pet_annotation/quality/auto_check.py`:

```python
"""Auto quality check — runs sampling decisions on auto_checked annotations.

Updates annotation review_status and frame annotation_status based on
the sampling decision.
"""
from __future__ import annotations

import logging

from pet_annotation.quality.sampling import decide_review
from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def run_auto_check(
    store: AnnotationStore,
    sampling_rate: float,
    threshold: float,
    primary_model: str,
) -> dict[str, int]:
    """Run auto check on all auto_checked frames.

    Updates review_status in annotations table and annotation_status
    in frames table.

    Args:
        store: AnnotationStore instance.
        sampling_rate: Random sampling rate for human review.
        threshold: Low confidence threshold for forced review.
        primary_model: Name of the primary model.

    Returns:
        Stats dict with counts of approved/needs_review.
    """
    rows = store._conn.execute(
        "SELECT a.annotation_id, a.frame_id, a.schema_valid, a.confidence_overall, a.prompt_hash "
        "FROM annotations a JOIN frames f ON a.frame_id = f.frame_id "
        "WHERE f.annotation_status = 'auto_checked' AND a.model_name = ? "
        "AND a.review_status = 'pending'",
        (primary_model,),
    ).fetchall()

    stats = {"approved": 0, "needs_review": 0}

    for row in rows:
        decision = decide_review(
            schema_valid=bool(row["schema_valid"]),
            confidence=row["confidence_overall"],
            sampling_rate=sampling_rate,
            threshold=threshold,
        )

        store._conn.execute(
            "UPDATE annotations SET review_status = ? WHERE annotation_id = ?",
            (decision, row["annotation_id"]),
        )

        new_frame_status = "approved" if decision == "approved" else "needs_review"
        store._conn.execute(
            "UPDATE frames SET annotation_status = ? WHERE frame_id = ?",
            (new_frame_status, row["frame_id"]),
        )
        store._conn.commit()

        stats[decision] += 1
        logger.info(
            '{"event": "auto_check", "frame_id": "%s", "decision": "%s", "confidence": %s}',
            row["frame_id"], decision,
            row["confidence_overall"] if row["confidence_overall"] is not None else "null",
        )

    return stats


if __name__ == "__main__":
    import asyncio
    import json
    from pathlib import Path
    from pet_annotation.config import load_config, setup_logging

    setup_logging()
    config = load_config()
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        stats = run_auto_check(
            store,
            config.annotation.review_sampling_rate,
            config.annotation.low_confidence_threshold,
            config.annotation.primary_model,
        )
        print(json.dumps(stats, indent=2))
    finally:
        store.close()
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_sampling.py tests/test_auto_check.py -v`
Expected: All 6 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/pet_annotation/quality/ tests/test_sampling.py tests/test_auto_check.py
git commit -m "feat(pet-annotation): quality module — auto_check + sampling decision engine"
```

---

## Task 11: DPO module (generate_pairs + validate_pairs)

**Files:**
- Create: `src/pet_annotation/dpo/__init__.py`
- Create: `src/pet_annotation/dpo/validate_pairs.py`
- Create: `src/pet_annotation/dpo/generate_pairs.py`
- Create: `tests/test_validate_pairs.py`
- Create: `tests/test_generate_pairs.py`

- [ ] **Step 1: Write failing tests for validate_pairs.py**

Create `tests/test_validate_pairs.py`:

```python
"""Tests for DPO pair validation — 5 rules from DEVELOPMENT_GUIDE."""
from __future__ import annotations

import pytest

from pet_annotation.dpo.validate_pairs import validate_pair


def _valid_output(**overrides) -> dict:
    base = {
        "schema_version": "1.0",
        "pet_present": True,
        "pet_count": 1,
        "pet": {
            "species": "cat", "breed_estimate": "british_shorthair",
            "id_tag": "grey_medium", "id_confidence": 0.85,
            "action": {
                "primary": "eating",
                "distribution": {"eating": 0.70, "drinking": 0.05, "sniffing_only": 0.10,
                                 "leaving_bowl": 0.05, "sitting_idle": 0.05, "other": 0.05},
            },
            "eating_metrics": {"speed": {"fast": 0.1, "normal": 0.7, "slow": 0.2},
                               "engagement": 0.8, "abandoned_midway": 0.1},
            "mood": {"alertness": 0.3, "anxiety": 0.1, "engagement": 0.8},
            "body_signals": {"posture": "relaxed", "ear_position": "forward"},
            "anomaly_signals": {"vomit_gesture": 0.0, "food_rejection": 0.0,
                                "excessive_sniffing": 0.0, "lethargy": 0.0, "aggression": 0.0},
        },
        "bowl": {"food_fill_ratio": 0.5, "water_fill_ratio": None, "food_type_visible": "dry"},
        "scene": {"lighting": "bright", "image_quality": "clear", "confidence_overall": 0.90},
        "narrative": "Grey cat eating normally.",
    }
    base.update(overrides)
    return base


class TestValidatePair:
    def test_valid_pair(self):
        chosen = _valid_output(narrative="Cat eating dry food calmly.")
        rejected = _valid_output(narrative="Kitty is having a wonderful dinner!", **{
            "scene": {"lighting": "bright", "image_quality": "clear", "confidence_overall": 0.75}
        })
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "model_comparison"})
        assert ok
        assert len(errors) == 0

    def test_rule3_identical_narrative_fails(self):
        chosen = _valid_output()
        rejected = _valid_output()  # same narrative
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "model_comparison"})
        assert not ok
        assert any("narrative" in e for e in errors)

    def test_rule4_user_feedback_needs_inference_id(self):
        chosen = _valid_output(narrative="Corrected by user.")
        rejected = _valid_output(narrative="Model got it wrong.")
        rejected["scene"]["confidence_overall"] = 0.60
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "user_feedback"})
        assert not ok
        assert any("inference_id" in e for e in errors)

    def test_rule5_chosen_confidence_must_be_higher(self):
        chosen = _valid_output(narrative="A")
        chosen["scene"]["confidence_overall"] = 0.50
        rejected = _valid_output(narrative="B")
        rejected["scene"]["confidence_overall"] = 0.90
        ok, errors = validate_pair(chosen, rejected, {"pair_source": "model_comparison"})
        assert not ok
        assert any("confidence" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validate_pairs.py -v`
Expected: FAIL

- [ ] **Step 3: Implement validate_pairs.py**

Create `src/pet_annotation/dpo/__init__.py` (empty).

Create `src/pet_annotation/dpo/validate_pairs.py`:

```python
"""DPO pair validation — enforces 5 rules from DEVELOPMENT_GUIDE."""
from __future__ import annotations

import json
import logging

from pet_schema import validate_output

logger = logging.getLogger(__name__)


def validate_pair(
    chosen: dict, rejected: dict, pair_meta: dict
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

    Returns:
        Tuple of (is_valid, list_of_error_strings).
    """
    errors: list[str] = []

    # Rule 1: chosen passes schema
    chosen_result = validate_output(json.dumps(chosen))
    if not chosen_result.valid:
        errors.append(f"chosen schema 验证失败: {chosen_result.errors}")

    # Rule 2: rejected passes schema
    rejected_result = validate_output(json.dumps(rejected))
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
        errors.append(
            f"chosen confidence ({chosen_conf}) < rejected confidence ({rejected_conf})"
        )

    return len(errors) == 0, errors
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_validate_pairs.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Write failing tests for generate_pairs.py**

Create `tests/test_generate_pairs.py`:

```python
"""Tests for DPO pair generation."""
from __future__ import annotations

import uuid

import pytest

from pet_annotation.dpo.generate_pairs import generate_cross_model_pairs
from pet_annotation.store import AnnotationRecord, AnnotationStore, ComparisonRecord


def _insert_frame(conn, fid):
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root, annotation_status) "
        "VALUES (?,?,?,?,?,?)",
        (fid, "v1", "selfshot", "f.jpg", "/data", "approved"),
    )
    conn.commit()


VALID_OUTPUT = (
    '{"schema_version":"1.0","pet_present":true,"pet_count":1,'
    '"pet":{"species":"cat","breed_estimate":"bsh","id_tag":"grey","id_confidence":0.8,'
    '"action":{"primary":"eating","distribution":{"eating":0.7,"drinking":0.05,'
    '"sniffing_only":0.1,"leaving_bowl":0.05,"sitting_idle":0.05,"other":0.05}},'
    '"eating_metrics":{"speed":{"fast":0.1,"normal":0.7,"slow":0.2},"engagement":0.8,"abandoned_midway":0.1},'
    '"mood":{"alertness":0.3,"anxiety":0.1,"engagement":0.8},'
    '"body_signals":{"posture":"relaxed","ear_position":"forward"},'
    '"anomaly_signals":{"vomit_gesture":0.0,"food_rejection":0.0,'
    '"excessive_sniffing":0.0,"lethargy":0.0,"aggression":0.0}},'
    '"bowl":{"food_fill_ratio":0.5,"water_fill_ratio":null,"food_type_visible":"dry"},'
    '"scene":{"lighting":"bright","image_quality":"clear","confidence_overall":%s},'
    '"narrative":"%s"}'
)


class TestGenerateCrossModelPairs:
    def test_generates_pair_when_primary_higher_confidence(self, db_conn):
        store = AnnotationStore(db_conn)
        _insert_frame(db_conn, "f1")

        store.insert_annotation(AnnotationRecord(
            annotation_id=str(uuid.uuid4()), frame_id="f1",
            model_name="primary", prompt_hash="h1",
            raw_response=VALID_OUTPUT % ("0.90", "Primary annotation."),
            schema_valid=True, confidence_overall=0.90,
            review_status="approved",
        ))
        store.insert_comparison(ComparisonRecord(
            comparison_id=str(uuid.uuid4()), frame_id="f1",
            model_name="secondary", prompt_hash="h1",
            raw_response=VALID_OUTPUT % ("0.70", "Secondary annotation."),
            schema_valid=True, confidence_overall=0.70,
        ))

        pairs = generate_cross_model_pairs(store, primary_model="primary")
        assert len(pairs) == 1
        assert pairs[0]["chosen"]["scene"]["confidence_overall"] == 0.90

    def test_skips_when_primary_lower_confidence(self, db_conn):
        store = AnnotationStore(db_conn)
        _insert_frame(db_conn, "f1")

        store.insert_annotation(AnnotationRecord(
            annotation_id=str(uuid.uuid4()), frame_id="f1",
            model_name="primary", prompt_hash="h1",
            raw_response=VALID_OUTPUT % ("0.60", "Primary low."),
            schema_valid=True, confidence_overall=0.60,
            review_status="approved",
        ))
        store.insert_comparison(ComparisonRecord(
            comparison_id=str(uuid.uuid4()), frame_id="f1",
            model_name="secondary", prompt_hash="h1",
            raw_response=VALID_OUTPUT % ("0.90", "Secondary high."),
            schema_valid=True, confidence_overall=0.90,
        ))

        pairs = generate_cross_model_pairs(store, primary_model="primary")
        assert len(pairs) == 0
```

- [ ] **Step 6: Implement generate_pairs.py**

Create `src/pet_annotation/dpo/generate_pairs.py`:

```python
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
) -> list[dict]:
    """Generate DPO pairs by comparing primary model vs secondary models.

    Only generates a pair when primary model confidence > secondary model confidence.
    Each pair is validated before inclusion.

    Args:
        store: AnnotationStore instance.
        primary_model: Name of the primary model.

    Returns:
        List of valid DPO pair dicts with chosen/rejected/metadata.
    """
    # Get all approved primary annotations
    approved = store._conn.execute(
        "SELECT a.frame_id, a.raw_response, a.confidence_overall "
        "FROM annotations a JOIN frames f ON a.frame_id = f.frame_id "
        "WHERE a.model_name = ? AND a.review_status IN ('approved', 'reviewed') "
        "AND a.schema_valid = 1",
        (primary_model,),
    ).fetchall()

    pairs = []

    for row in approved:
        frame_id = row["frame_id"]
        primary_conf = row["confidence_overall"]
        primary_output = json.loads(row["raw_response"])

        # Find comparison results for this frame
        comparisons = store.fetch_comparisons_for_frame(frame_id)
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

            ok, errors = validate_pair(primary_output, rejected_output, pair_meta)
            if ok:
                pairs.append({
                    "chosen": primary_output,
                    "rejected": rejected_output,
                    "metadata": pair_meta,
                    "frame_id": frame_id,
                })
            else:
                logger.info(
                    '{"event": "pair_rejected", "frame_id": "%s", "errors": %s}',
                    frame_id, json.dumps(errors),
                )

    logger.info('{"event": "pairs_generated", "count": %d}', len(pairs))
    return pairs
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_validate_pairs.py tests/test_generate_pairs.py -v`
Expected: All 6 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/pet_annotation/dpo/ tests/test_validate_pairs.py tests/test_generate_pairs.py
git commit -m "feat(pet-annotation): DPO module — validate_pairs + generate_pairs"
```

---

## Task 12: Export module (to_sharegpt + to_dpo_pairs)

**Files:**
- Create: `src/pet_annotation/export/__init__.py`
- Create: `src/pet_annotation/export/to_sharegpt.py`
- Create: `src/pet_annotation/export/to_dpo_pairs.py`
- Create: `tests/test_export.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_export.py`:

```python
"""Tests for export modules."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from pet_annotation.export.to_sharegpt import export_sharegpt
from pet_annotation.store import AnnotationRecord, AnnotationStore

VALID_RAW = (
    '{"schema_version":"1.0","pet_present":false,"pet_count":0,"pet":null,'
    '"bowl":{"food_fill_ratio":0.5,"water_fill_ratio":null,"food_type_visible":"dry"},'
    '"scene":{"lighting":"bright","image_quality":"clear","confidence_overall":0.9},'
    '"narrative":"Empty bowl."}'
)


def _insert_frame(conn, fid):
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root, annotation_status) "
        "VALUES (?,?,?,?,?,?)",
        (fid, "v1", "selfshot", "frames/001.jpg", "/data", "approved"),
    )
    conn.commit()


class TestExportShareGPT:
    def test_exports_approved_annotations(self, db_conn, tmp_path):
        store = AnnotationStore(db_conn)
        _insert_frame(db_conn, "f1")
        store.insert_annotation(AnnotationRecord(
            annotation_id=str(uuid.uuid4()), frame_id="f1",
            model_name="primary", prompt_hash="h1",
            raw_response=VALID_RAW, parsed_output=VALID_RAW,
            schema_valid=True, confidence_overall=0.9,
            review_status="approved",
        ))

        out = tmp_path / "sft.jsonl"
        count = export_sharegpt(store, out, schema_version="1.0")
        assert count == 1

        line = json.loads(out.read_text().strip())
        assert line["conversations"][0]["from"] == "system"
        assert line["conversations"][1]["from"] == "human"
        assert line["conversations"][2]["from"] == "gpt"
        assert "frames/001.jpg" in line["images"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_export.py -v`
Expected: FAIL

- [ ] **Step 3: Implement to_sharegpt.py**

Create `src/pet_annotation/export/__init__.py` (empty).

Create `src/pet_annotation/export/to_sharegpt.py`:

```python
"""Export approved annotations to ShareGPT JSONL format for SFT training."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pet_schema import render_prompt

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def export_sharegpt(
    store: AnnotationStore,
    output_path: Path,
    schema_version: str = "1.0",
) -> int:
    """Export approved annotations to ShareGPT JSONL.

    Args:
        store: AnnotationStore instance.
        output_path: Path to write the JSONL file.
        schema_version: Schema version for prompt rendering.

    Returns:
        Number of records exported.
    """
    system_prompt, user_prompt = render_prompt(version=schema_version)
    rows = store.fetch_approved_annotations(limit=100000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            record = {
                "id": f"sft_{count:05d}",
                "conversations": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": f"<image>\n{user_prompt}"},
                    {"from": "gpt", "value": row["parsed_output"] or row["raw_response"]},
                ],
                "images": [row["frame_path"]],
                "metadata": {
                    "source": row["source"],
                    "schema_version": schema_version,
                    "prompt_version": schema_version,
                    "annotator": row["model_name"],
                    "review_status": row["review_status"],
                    "frame_id": row["frame_id"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info('{"event": "export_sharegpt", "count": %d, "path": "%s"}', count, output_path)
    return count
```

- [ ] **Step 4: Implement to_dpo_pairs.py**

Create `src/pet_annotation/export/to_dpo_pairs.py`:

```python
"""Export validated DPO pairs to JSONL format for DPO training."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pet_schema import render_prompt

logger = logging.getLogger(__name__)


def export_dpo_pairs(
    pairs: list[dict],
    output_path: Path,
    schema_version: str = "1.0",
) -> int:
    """Export DPO pairs to JSONL.

    Args:
        pairs: List of validated pair dicts from generate_pairs.
        output_path: Path to write the JSONL file.
        schema_version: Schema version for prompt rendering.

    Returns:
        Number of pairs exported.
    """
    system_prompt, user_prompt = render_prompt(version=schema_version)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "id": f"dpo_{count:05d}",
                "system": system_prompt,
                "prompt": f"<image>\n{user_prompt}",
                "images": [],  # Frame path would need to be resolved
                "chosen": [
                    {"role": "user", "content": f"<image>\n{user_prompt}"},
                    {"role": "assistant", "content": json.dumps(pair["chosen"], ensure_ascii=False)},
                ],
                "rejected": [
                    {"role": "user", "content": f"<image>\n{user_prompt}"},
                    {"role": "assistant", "content": json.dumps(pair["rejected"], ensure_ascii=False)},
                ],
                "metadata": pair["metadata"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info('{"event": "export_dpo", "count": %d, "path": "%s"}', count, output_path)
    return count
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_export.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pet_annotation/export/ tests/test_export.py
git commit -m "feat(pet-annotation): export module — to_sharegpt + to_dpo_pairs JSONL"
```

---

## Task 13: CLI entry point + DVC pipeline

**Files:**
- Create: `src/pet_annotation/cli.py`
- Create: `dvc.yaml`

- [ ] **Step 1: Implement CLI**

Create `src/pet_annotation/cli.py`:

```python
"""CLI entry point for pet-annotation."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click

from pet_annotation.config import load_config, setup_logging


@click.group()
def cli():
    """pet-annotation: VLM annotation, quality check, and training data export."""
    setup_logging()


@cli.command()
@click.option("--batch-size", default=None, type=int, help="Override params.yaml batch_size")
@click.option("--dry-run", is_flag=True, help="Print plan without calling APIs")
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def annotate(batch_size, dry_run, params):
    """Batch annotate pending frames using configured models."""
    from pet_annotation.store import AnnotationStore
    from pet_annotation.teacher.orchestrator import AnnotationOrchestrator

    config = load_config(Path(params))
    if batch_size:
        config.annotation.batch_size = batch_size

    if dry_run:
        store = AnnotationStore(db_path=Path(config.database.path))
        pending = store.fetch_pending_frames(limit=9999)
        click.echo(f"Pending frames: {len(pending)}")
        click.echo(f"Models: {list(config.models.keys())}")
        click.echo(f"Primary: {config.annotation.primary_model}")
        store.close()
        return

    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        orch = AnnotationOrchestrator(config=config, store=store)
        stats = asyncio.run(orch.run())
        click.echo(json.dumps(stats, indent=2))
    finally:
        store.close()


@cli.command()
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def check(params):
    """Run quality check on auto_checked annotations."""
    from pet_annotation.quality.auto_check import run_auto_check
    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        stats = run_auto_check(
            store,
            config.annotation.review_sampling_rate,
            config.annotation.low_confidence_threshold,
            config.annotation.primary_model,
        )
        click.echo(json.dumps(stats, indent=2))
    finally:
        store.close()


@cli.command(name="export")
@click.option("--format", "fmt", type=click.Choice(["sft", "dpo", "audio"]), required=True)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def export_cmd(fmt, output, params):
    """Export training data in the specified format."""
    from pet_annotation.store import AnnotationStore

    config = load_config(Path(params))
    store = AnnotationStore(db_path=Path(config.database.path))

    try:
        if fmt == "sft":
            from pet_annotation.export.to_sharegpt import export_sharegpt
            out = Path(output) if output else Path("exports/sft_train.jsonl")
            count = export_sharegpt(store, out, config.annotation.schema_version)
            click.echo(f"Exported {count} SFT records to {out}")

        elif fmt == "dpo":
            from pet_annotation.dpo.generate_pairs import generate_cross_model_pairs
            from pet_annotation.export.to_dpo_pairs import export_dpo_pairs
            pairs = generate_cross_model_pairs(store, config.annotation.primary_model)
            out = Path(output) if output else Path("exports/dpo_pairs.jsonl")
            count = export_dpo_pairs(pairs, out, config.annotation.schema_version)
            click.echo(f"Exported {count} DPO pairs to {out}")

        elif fmt == "audio":
            click.echo("Audio label export not yet implemented")
    finally:
        store.close()


@cli.command()
@click.option("--params", default="params.yaml", type=click.Path(exists=True))
def stats(params):
    """Print annotation progress and token usage statistics."""
    config = load_config(Path(params))
    from pet_annotation.store import AnnotationStore
    store = AnnotationStore(db_path=Path(config.database.path))
    try:
        status_counts = store._conn.execute(
            "SELECT annotation_status, COUNT(*) FROM frames GROUP BY annotation_status"
        ).fetchall()
        model_counts = store._conn.execute(
            "SELECT model_name, COUNT(*), SUM(total_tokens) FROM annotations GROUP BY model_name"
        ).fetchall()
        click.echo("=== Frame Status ===")
        for row in status_counts:
            click.echo(f"  {row[0]}: {row[1]}")
        click.echo("\n=== Model Stats ===")
        for row in model_counts:
            click.echo(f"  {row[0]}: {row[1]} annotations, {row[2] or 0} tokens")
    finally:
        store.close()


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Create `dvc.yaml`**

```yaml
stages:
  annotate:
    cmd: python -m pet_annotation.cli annotate
    deps:
      - src/pet_annotation/teacher/
      - params.yaml
    params:
      - annotation
      - models
    outs:
      - reports/annotation_stats.json:
          cache: false

  quality_check:
    cmd: python -m pet_annotation.cli check
    deps:
      - src/pet_annotation/quality/
      - reports/annotation_stats.json
    params:
      - annotation.review_sampling_rate
      - annotation.low_confidence_threshold

  generate_pairs:
    cmd: python -m pet_annotation.dpo.generate_pairs
    deps:
      - src/pet_annotation/dpo/generate_pairs.py
      - src/pet_annotation/dpo/validate_pairs.py
    params:
      - dpo.min_pairs_per_release
    outs:
      - reports/dpo_pairs_stats.json:
          cache: false

  export_sft:
    cmd: python -m pet_annotation.cli export --format sft
    deps:
      - src/pet_annotation/export/to_sharegpt.py
      - reports/annotation_stats.json
    outs:
      - exports/sft_train.jsonl

  export_dpo:
    cmd: python -m pet_annotation.cli export --format dpo
    deps:
      - src/pet_annotation/export/to_dpo_pairs.py
      - reports/dpo_pairs_stats.json
    outs:
      - exports/dpo_pairs.jsonl
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pet_annotation/cli.py dvc.yaml
git commit -m "feat(pet-annotation): CLI entry point + DVC pipeline definition"
```

---

## Task 14: Human review stubs + integration wiring

**Files:**
- Create: `src/pet_annotation/human_review/__init__.py`
- Create: `src/pet_annotation/human_review/import_to_ls.py`
- Create: `src/pet_annotation/human_review/export_from_ls.py`
- Create: `src/pet_annotation/dpo/import_app_feedback.py`
- Create: `src/pet_annotation/export/to_audio_labels.py`

**Context:** These modules interact with Label Studio (external service). We create properly structured stubs with full interfaces and docstrings that can be filled in when Label Studio is set up. No tests for these yet — they require a running Label Studio instance.

- [ ] **Step 1: Create human_review module stubs**

Create `src/pet_annotation/human_review/__init__.py` (empty).

Create `src/pet_annotation/human_review/import_to_ls.py`:

```python
"""Import VLM outputs to Label Studio as review tasks.

Pre-fills VLM output as predictions so reviewers see the model's answer
and can quickly confirm or correct.
"""
from __future__ import annotations

import logging

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def import_needs_review(store: AnnotationStore, ls_url: str, ls_api_key: str) -> int:
    """Create Label Studio tasks for annotations needing review.

    Queries annotations with review_status='needs_review', builds LS tasks
    with VLM output pre-filled as predictions, and creates them via LS API.

    Args:
        store: AnnotationStore instance.
        ls_url: Label Studio server URL.
        ls_api_key: Label Studio API key.

    Returns:
        Number of tasks created.
    """
    raise NotImplementedError("Requires Label Studio integration — implement when LS is deployed")
```

Create `src/pet_annotation/human_review/export_from_ls.py`:

```python
"""Export completed Label Studio annotations back to the database.

Updates review_status and optionally overwrites parsed_output if the
reviewer made corrections.
"""
from __future__ import annotations

import logging

from pet_annotation.store import AnnotationStore

logger = logging.getLogger(__name__)


def export_reviewed(store: AnnotationStore, ls_url: str, ls_api_key: str) -> int:
    """Pull completed annotations from Label Studio and update DB.

    Args:
        store: AnnotationStore instance.
        ls_url: Label Studio server URL.
        ls_api_key: Label Studio API key.

    Returns:
        Number of annotations updated.
    """
    raise NotImplementedError("Requires Label Studio integration — implement when LS is deployed")
```

- [ ] **Step 2: Create DPO app feedback stub**

Create `src/pet_annotation/dpo/import_app_feedback.py`:

```python
"""Import user feedback from APP into Label Studio for DPO pair generation.

Pulls feeding_events with user_feedback='inaccurate' from cloud sync,
creates Label Studio tasks for human confirmation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def import_user_corrections(ls_url: str, ls_api_key: str) -> int:
    """Pull user corrections and create Label Studio tasks.

    Args:
        ls_url: Label Studio server URL.
        ls_api_key: Label Studio API key.

    Returns:
        Number of correction tasks created.
    """
    raise NotImplementedError("Requires cloud sync + Label Studio — implement post-launch")
```

- [ ] **Step 3: Create audio labels stub**

Create `src/pet_annotation/export/to_audio_labels.py`:

```python
"""Export audio classification labels for the audio CNN training pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_audio_labels(output_path: Path) -> int:
    """Export audio labels to CSV format.

    Args:
        output_path: Path to write the labels file.

    Returns:
        Number of labels exported.
    """
    raise NotImplementedError("Audio labeling pipeline not yet defined")
```

- [ ] **Step 4: Run full test suite + lint**

Run: `pytest tests/ -v && make lint`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/pet_annotation/human_review/ src/pet_annotation/dpo/import_app_feedback.py src/pet_annotation/export/to_audio_labels.py
git commit -m "feat(pet-annotation): human_review + app_feedback + audio_labels stubs"
```

---

## Task 15: Final integration test + cleanup

**Files:**
- Modify: `tests/conftest.py` (verify it works for all tests)
- Run: full test suite, lint, verify commit history

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/bamboo/Githubs/Train-Pet-Pipeline/pet-annotation && pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: PASS (no ruff or mypy errors)

- [ ] **Step 3: Verify git log**

Run: `git log --oneline`
Expected: Clean linear history with descriptive commits

- [ ] **Step 4: Final commit if any cleanup needed**

Only if there are pending changes from fixups.

---

## Summary

| Task | Module | Tests | Key Deliverable |
|------|--------|-------|-----------------|
| 1 | Scaffolding | conftest | pyproject.toml, Makefile, .gitignore |
| 2 | Store | 7 tests | AnnotationStore + migration SQL |
| 3 | Config | 4 tests | params.yaml + Pydantic loader |
| 4 | RateTracker | 5 tests | Sliding window key scheduler |
| 5 | CostTracker | 4 tests | Daily token budget |
| 6 | Provider | 4 tests | BaseProvider + OpenAICompat |
| 7 | Providers | 3 tests | Doubao + VLLM |
| 8 | Registry | 4 tests | ProviderRegistry |
| 9 | Orchestrator | 3 tests | Core async batch engine |
| 10 | Quality | 6 tests | auto_check + sampling |
| 11 | DPO | 6 tests | validate_pairs + generate_pairs |
| 12 | Export | 1 test | ShareGPT + DPO JSONL |
| 13 | CLI + DVC | — | Click CLI + dvc.yaml |
| 14 | Stubs | — | Label Studio + audio stubs |
| 15 | Integration | — | Final verification |

**Total: ~47 tests, 15 tasks, ~25 files created**
