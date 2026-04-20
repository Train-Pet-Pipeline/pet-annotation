-- Create audio_annotations table for storing CNN / VLM / human audio-class annotations.
-- Columns round-trip to pet_schema.AudioAnnotation (v2.0.0).
-- sample_id is a plain TEXT FK to audio_samples in pet-data (different DB — no FK constraint).

CREATE TABLE IF NOT EXISTS audio_annotations (
    annotation_id  TEXT PRIMARY KEY,
    sample_id      TEXT NOT NULL,
    annotator_type TEXT NOT NULL CHECK (annotator_type IN ('vlm','cnn','human','rule')),
    annotator_id   TEXT NOT NULL,
    modality       TEXT NOT NULL DEFAULT 'audio' CHECK (modality = 'audio'),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version TEXT NOT NULL DEFAULT '2.0.0',
    predicted_class TEXT NOT NULL,
    class_probs    TEXT NOT NULL,
    logits         TEXT
);
CREATE INDEX IF NOT EXISTS idx_audio_ann_sample ON audio_annotations(sample_id);
CREATE INDEX IF NOT EXISTS idx_audio_ann_class  ON audio_annotations(predicted_class);
