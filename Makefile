-include ../pet-infra/shared/Makefile.include

.PHONY: setup test lint clean

setup:
	python -m pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short
