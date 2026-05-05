.PHONY: install lint format test download-nltk run-sentiment run-topics clean

PYTHON ?= python

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests

format:
	ruff check --fix src tests
	black src tests
	isort src tests

test:
	pytest -m "not slow" --cov=conflict_sentiment --cov-report=term-missing

download-nltk:
	$(PYTHON) -c "from conflict_sentiment.preprocessing import ensure_nltk_resources; ensure_nltk_resources()"

run-sentiment:
	conflict-sentiment sentiment

run-topics:
	conflict-sentiment topics

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
