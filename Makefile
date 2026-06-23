.PHONY: install rank precompute validate test lint docker-build docker-run

install:
	uv sync

rank:
	uv run python -m scripts.rank --candidates data/input/candidates.jsonl --jd data/input/job_description.docx --output data/output/submission.csv

precompute:
	uv run python -m scripts.precompute --candidates data/input/candidates.jsonl --output artifacts/

validate:
	uv run python -m scripts.validate --submission data/output/submission.csv

test:
	uv run pytest tests/ -v --cov=src

lint:
	uv run ruff check src/ && uv run mypy src/

docker-build:
	docker build -t resume-ranker .

docker-run:
	docker run -v $(PWD)/data/output:/app/data/output resume-ranker
