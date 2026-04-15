-- pet-annotation tables (owned by pet-annotation, lives in pet-data's SQLite DB)

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
