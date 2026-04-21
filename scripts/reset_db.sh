#!/usr/bin/env bash
set -euo pipefail
DB_PATH="${1:-data/annotation.db}"
if [ -f "$DB_PATH" ]; then
  mv "$DB_PATH" "${DB_PATH}.v1backup.$(date +%s)"
  echo "Backed up old DB to ${DB_PATH}.v1backup.*"
fi
python -c "import sys; from pet_annotation.store import AnnotationStore; AnnotationStore(sys.argv[1]).init_schema()" "$DB_PATH"
echo "Fresh DB initialized at $DB_PATH (4 paradigm tables)."
