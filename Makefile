.PHONY: install rank precompute validate test lint docker-build docker-run

install:
	uv sync

rank:
	uv run rank --candidates data/input/candidates.jsonl --jd data/input/job_description.docx --output data/output/submission.csv

precompute:
	uv run precompute --candidates data/input/candidates.jsonl --jd data/input/job_description.docx

validate:
	uv run validate --submission data/output/submission.csv

test:
	uv run pytest tests/ -v --cov=src/resume_ranker

lint:
	uv run ruff check src/resume_ranker tests && uv run mypy src/resume_ranker

docker-build:
	docker build -t resume-ranker .

docker-run:
	docker run -v $(PWD)/data/output:/app/data/output resume-ranker
