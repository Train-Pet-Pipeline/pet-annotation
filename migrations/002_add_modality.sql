-- Add modality column to annotations and model_comparisons, plus a supporting index.

ALTER TABLE annotations        ADD COLUMN modality TEXT NOT NULL DEFAULT 'vision';
ALTER TABLE annotations        ADD COLUMN storage_uri TEXT;
ALTER TABLE model_comparisons  ADD COLUMN modality TEXT NOT NULL DEFAULT 'vision';
CREATE INDEX IF NOT EXISTS idx_annotations_modality ON annotations(modality)
