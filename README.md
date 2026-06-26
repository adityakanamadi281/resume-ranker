# Resume Ranker

Production-grade offline candidate ranking system built for the Redrob Intelligent Candidate Discovery & Ranking Hackathon.

## Overview

This project ranks candidates from a large candidate pool (100,000 candidates) against a job description using semantic retrieval, structured feature scoring, behavioral signals, and deterministic reasoning.

The system is designed to be:
- **Offline-first**
- **CPU-only during ranking**
- **Reproducible**
- **Explainable**
- **Fast enough for large candidate pools**

## Key Features

- **Offline semantic retrieval** using `BAAI/bge-small-en-v1.5`
- **FAISS IndexFlatIP** vector search
- **Streaming JSONL reader** using `orjson`
- **Structured feature extraction**
- **Behavioral signal scoring**
- **Honeypot filtering**
- **Rule-based candidate explanations**
- **Deterministic ranking**
- **Audit report generation**
- **Fully reproducible pipeline**


## Technology Stack

- **Python 3.12**
- **Sentence Transformers** (`BAAI/bge-small-en-v1.5`)
- **FAISS CPU**
- **Polars**
- **NumPy**
- **PyArrow**
- **orjson**
- **RapidFuzz**
- **Typer**
- **Rich**
- **tqdm**
- **jsonschema**


## Setup

```bash
# Clone the repository
git clone https://github.com/adityakanamadi281/resume-ranker.git

# Navigate to the project directory
cd resume-ranker
```


```bash
uv sync
```

## Run

Place inputs in `data/input/`:

- `candidates.jsonl`
- `job_description.docx`

Then run:

```bash
make rank
make validate
```

Or directly:

```bash
uv run rank --candidates data/input/candidates.jsonl --jd data/input/job_description.docx --output data/output/submission.csv
uv run validate --submission data/output/submission.csv
```

Outputs:

- `data/output/submission.csv`
- `data/output/submission.audit.json`
- `artifacts/manifest.json`
- `artifacts/candidate_embeddings.npy`
- `artifacts/id_map.json`
- `artifacts/faiss.index`

## Pre-download Models

Before running the ranking pipeline in an offline environment, you must download and cache the embedding model once while you have network access:

```bash
python scripts/download_model.py
```

This downloads and caches `BAAI/bge-small-en-v1.5` in your local Hugging Face cache.


## Project Structure

```text
resume-ranker/
├── configs/                     # Documented ranking profiles
│   ├── default.toml
│   ├── fast.toml
│   └── quality.toml
├── data/
│   └── input/                   # Challenge inputs
│       ├── candidate_schema.json
│       ├── candidates.jsonl
│       └── job_description.docx
├── scripts/                     # Utility scripts
│   ├── download_model.py
│   └── smoke_run.ps1
├── src/
│   └── resume_ranker/           # Main application package
│       ├── __init__.py
│       ├── config.py
│       ├── exceptions.py
│       ├── app/                 # Orchestration, run context, output validation
│       │   ├── __init__.py
│       │   ├── output_writer.py
│       │   ├── pipeline.py
│       │   └── run_context.py
│       ├── cli/                 # Command-line interfaces
│       │   ├── __init__.py
│       │   ├── benchmark.py
│       │   ├── precompute.py
│       │   ├── rank.py
│       │   └── validate.py
│       ├── domain/              # Schema, scoring logic, honeypot rules, explanations
│       │   ├── __init__.py
│       │   ├── explanations.py
│       │   ├── honeypot_rules.py
│       │   ├── schema.py
│       │   └── scoring.py
│       ├── evaluation/          # Audit reports, metrics, benchmarks
│       │   ├── __init__.py
│       │   ├── audit.py
│       │   ├── benchmark.py
│       │   └── metrics.py
│       ├── features/            # Text builders and signal feature generation
│       │   ├── __init__.py
│       │   ├── signal_features.py
│       │   └── text_builder.py
│       └── infrastructure/      # Readers, embedding providers, vector index, artifact store
│           ├── __init__.py
│           ├── artifact_store.py
│           ├── candidate_reader.py
│           ├── embedder.py
│           ├── jd_reader.py
│           └── vector_index.py
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── integration/             # Integration tests
│   │   └── test_pipeline_small_fixture.py
│   ├── regression/              # Regression tests
│   │   └── test_artifact_manifest.py
│   └── unit/                    # Unit tests
│       ├── test_cli.py
│       ├── test_honeypot.py
│       ├── test_readers.py
│       └── test_scorer.py
├── .gitignore
├── .python-version
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## Scoring

```text
composite = 0.35 * semantic_score
          + 0.15 * title_score
          + 0.20 * skill_score
          + 0.20 * behavioral_score
          + 0.10 * experience_score
```

Weights and thresholds live in `resume_ranker.config.AppConfig` and can be
overridden with `RANKER_*` environment variables.

## Artifact Safety

The cache is guarded by `artifacts/manifest.json`, which records:

- candidate file hash
- job description file hash
- model name
- embedding mode
- config hash
- candidate count
- embedding dimension
- creation timestamp

If any relevant input or setting changes, embeddings are rebuilt instead of
silently reused.

## Tests

```bash
make test
```

The suite covers domain scoring, honeypot rules, readers, full small-fixture
pipeline behavior, and stale artifact protection.

## Docker

```bash
docker build -t resume-ranker .
docker run -v $(pwd)/data/input:/app/data/input:ro -v $(pwd)/data/output:/app/data/output resume-ranker
```
