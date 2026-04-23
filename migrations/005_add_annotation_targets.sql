-- Phase 4 orchestrator wire: track per-(target, annotator) annotation state.
-- Source of pending targets: pet-data's frames table (read-only via sqlite3.connect).
-- State semantics: pending → in_progress → (done | failed).

CREATE TABLE annotation_targets (
    target_id       TEXT NOT NULL,
    annotator_id    TEXT NOT NULL,
    annotator_type  TEXT NOT NULL CHECK (annotator_type IN ('llm','classifier','rule','human')),
    state           TEXT NOT NULL CHECK (state IN ('pending','in_progress','done','failed')),
    claimed_at      TEXT,
    finished_at     TEXT,
    error_msg       TEXT,
    PRIMARY KEY (target_id, annotator_id)
);
CREATE INDEX idx_targets_state ON annotation_targets(state);
CREATE INDEX idx_targets_type  ON annotation_targets(annotator_type);
