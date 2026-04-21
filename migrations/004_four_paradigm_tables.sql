-- Phase 2 还债：annotation 表按 annotator 范式拆 4 张（从 modality 换轴为 annotator_type）
-- Spec: 2026-04-21-phase-2-debt-repayment-design.md §2

-- Drop 旧表（破坏性，无数据迁移；历史靠 git checkout <v1.1.0>）
DROP TABLE IF EXISTS annotations;
DROP TABLE IF EXISTS audio_annotations;
DROP TABLE IF EXISTS model_comparisons;

CREATE TABLE llm_annotations (
    annotation_id   TEXT    PRIMARY KEY,
    target_id       TEXT    NOT NULL,
    annotator_id    TEXT    NOT NULL,
    annotator_type  TEXT    NOT NULL CHECK (annotator_type = 'llm'),
    modality        TEXT    NOT NULL CHECK (modality IN ('vision','audio','sensor','multimodal')),
    schema_version  TEXT    NOT NULL DEFAULT '2.1.0',
    created_at      TEXT    NOT NULL,
    storage_uri     TEXT,
    prompt_hash     TEXT    NOT NULL,
    raw_response    TEXT    NOT NULL,
    parsed_output   TEXT    NOT NULL,
    UNIQUE (target_id, annotator_id, prompt_hash)
);
CREATE INDEX idx_llm_target ON llm_annotations(target_id);
CREATE INDEX idx_llm_modality ON llm_annotations(modality);

CREATE TABLE classifier_annotations (
    annotation_id   TEXT    PRIMARY KEY,
    target_id       TEXT    NOT NULL,
    annotator_id    TEXT    NOT NULL,
    annotator_type  TEXT    NOT NULL CHECK (annotator_type = 'classifier'),
    modality        TEXT    NOT NULL CHECK (modality IN ('vision','audio','sensor','multimodal')),
    schema_version  TEXT    NOT NULL DEFAULT '2.1.0',
    created_at      TEXT    NOT NULL,
    storage_uri     TEXT,
    predicted_class TEXT    NOT NULL,
    class_probs     TEXT    NOT NULL,
    logits          TEXT,
    UNIQUE (target_id, annotator_id)
);
CREATE INDEX idx_cls_target ON classifier_annotations(target_id);
CREATE INDEX idx_cls_modality ON classifier_annotations(modality);

CREATE TABLE rule_annotations (
    annotation_id   TEXT    PRIMARY KEY,
    target_id       TEXT    NOT NULL,
    annotator_id    TEXT    NOT NULL,
    annotator_type  TEXT    NOT NULL CHECK (annotator_type = 'rule'),
    modality        TEXT    NOT NULL CHECK (modality IN ('vision','audio','sensor','multimodal')),
    schema_version  TEXT    NOT NULL DEFAULT '2.1.0',
    created_at      TEXT    NOT NULL,
    storage_uri     TEXT,
    rule_id         TEXT    NOT NULL,
    rule_output     TEXT    NOT NULL,
    UNIQUE (target_id, annotator_id, rule_id)
);
CREATE INDEX idx_rule_target ON rule_annotations(target_id);

CREATE TABLE human_annotations (
    annotation_id   TEXT    PRIMARY KEY,
    target_id       TEXT    NOT NULL,
    annotator_id    TEXT    NOT NULL,
    annotator_type  TEXT    NOT NULL CHECK (annotator_type = 'human'),
    modality        TEXT    NOT NULL CHECK (modality IN ('vision','audio','sensor','multimodal')),
    schema_version  TEXT    NOT NULL DEFAULT '2.1.0',
    created_at      TEXT    NOT NULL,
    storage_uri     TEXT,
    reviewer        TEXT    NOT NULL,
    decision        TEXT    NOT NULL,
    notes           TEXT,
    UNIQUE (target_id, annotator_id)
);
CREATE INDEX idx_human_target ON human_annotations(target_id);
