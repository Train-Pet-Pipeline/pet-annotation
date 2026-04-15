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
