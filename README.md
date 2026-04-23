# pet-annotation

VLM annotation, quality check, and training data export pipeline for the Train-Pet-Pipeline project.

## Prerequisites

Install peer dependencies before this package:

```bash
pip install 'pet-infra @ git+https://github.com/Train-Pet-Pipeline/pet-infra@v2.1.0'
pip install -e ".[dev]" --no-deps
pip install -e ".[dev]"
```

## Quick Start

```bash
# Annotate with LLM paradigm
pet-annotation annotate --annotator llm --modality vision

# Dry run
pet-annotation annotate --annotator llm --modality audio --dry-run
```

## Breaking Changes in v2.0.0

Annotation tables have been rebuilt per annotator paradigm (LLMAnnotation/ClassifierAnnotation/RuleAnnotation/HumanAnnotation).
Old `annotations` / `audio_annotations` / `model_comparisons` tables are dropped + rebuilt, **with no data migration script**.
Historical data can be recovered via `git checkout v1.1.0`. Local `.db` files must be rebuilt:

```bash
./scripts/reset_db.sh
```

## Development

```bash
make setup   # install deps
make test    # run pytest
make lint    # ruff + mypy
make clean   # remove build artifacts
```

## Architecture

```
pet-schema → pet-annotation (4 annotator-paradigm tables)
                ├── llm_annotations
                ├── classifier_annotations
                ├── rule_annotations
                └── human_annotations
```

### Phase 4: Pending-target infrastructure

The Phase 4 orchestrator introduces a `annotation_targets` table that tracks
per-(target, annotator) annotation state: `pending → in_progress → (done | failed)`.

**Cross-repo read pattern (D1):** pet-annotation reads pet-data's `frames` table
read-only via `sqlite3.connect(pet_data_db_path, timeout=10)` — no code import
from pet-data, no write access. This is the same pattern used by pet-data's own
`datasets/vision_frames.py` for read-only cross-component queries. The path is
configured via `annotation.pet_data_db_path` in `params.yaml`.

**Design decisions (locked 2026-04-23):**
- D1: pending targets read from pet-data frames table (read-only sqlite3.connect)
- D2: state tracked in pet-annotation's own `annotation_targets` table (migration 005)
- D3: `annotation.llm.annotators: []` — N=0 is valid (no LLM annotation this run)
- D4: no cross-model reconcile — each annotator writes independently

## License

This project is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).
On **2030-04-22** it converts automatically to the Apache License, Version 2.0.

> Note: BSL 1.1 is **source-available**, not OSI-approved open source.
> Production / commercial use requires a separate commercial license.

![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)
