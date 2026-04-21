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
