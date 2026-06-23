# Resume Ranker

Production-grade candidate ranking system. Reads `candidates.jsonl` and
`job_description.docx`, ranks 100,000+ candidates in under 5 minutes on
a single CPU core (16 GB RAM), and outputs exactly 100 ranked rows to
`submission.csv`.

## Setup and installation

```bash
# Clone the repository
git clone https://github.com/adityakanamadi281/resume-ranker.git

# Navigate to the project directory
cd resume-ranker

# Install dependencies using uv or make
uv sync
# or:
make install
```

## Running the pipeline

Ensure your input files are placed in the `data/input` directory:
* `data/input/candidates.jsonl`
* `data/input/job_description.docx`

### Run using Makefile
```bash
make rank
```

### Run using direct uv
```bash
uv run python -m scripts.rank
```

Output will be written to `data/output/submission.csv`.

## Project structure

```
resume-ranker/
├── pyproject.toml                 # uv-managed dependencies
├── uv.lock                        # pinned versions
├── Makefile                       # convenience targets
├── Dockerfile / docker-compose.yml
├── data/
│   ├── input/
│   │   ├── candidates.jsonl       # INPUT (required)
│   │   ├── job_description.docx   # INPUT (required)
│   │   └── candidate_schema.json  # optional reference
│   └── output/
│       └── submission.csv         # OUTPUT
├── artifacts/                     # cached embeddings + FAISS index
└── src/
    ├── config.py                  # ALL tunable constants (single source of truth)
    ├── schema.py                  # Pydantic v2 candidate models
    ├── exceptions.py
    ├── pipeline.py                # end-to-end orchestration
    ├── parsers/
    │   ├── docx_parser.py         # reads job_description.docx
    │   └── candidate_parser.py    # streaming JSONL parser with validation
    ├── features/
    │   ├── text_builder.py        # rich embedding text per candidate
    │   ├── signal_processor.py    # redrob_signals -> normalized [0,1] features
    │   └── honeypot_detector.py   # 6 rule-based consistency checks
    ├── embedding/
    │   ├── embedder.py            # sentence-transformers (+ TF-IDF fallback)
    │   └── index.py               # FAISS IVF index build and search
    └── ranking/
        ├── scorer.py              # 5-factor hybrid scoring
        ├── reasoner.py            # rule-based explanation generation
        └── aggregator.py          # top-N with deterministic tie-break
```

## Scoring model

```
composite = 0.35 * semantic_score    (sentence-transformers / TF-IDF cosine)
          + 0.15 * title_score       (JD keyword overlap with candidate title)
          + 0.20 * skill_score       (JD keyword overlap with skills list)
          + 0.20 * behavioral_score  (redrob_signals: availability, activity)
          + 0.10 * experience_score  (years_of_experience vs ideal range)
```

All weights live in `src/config.py` and can be overridden at runtime via
environment variables: `RANKER_WEIGHT_SEMANTIC=0.4 uv run python -m scripts.rank`

## Honeypot detection

Six consistency rules run on every candidate before scoring. Honeypots are
excluded entirely from the index (not just ranked lower). Rules per the
spec: duration mismatch, expert+zero experience, inverted salary, job
before education, high completeness + low connections, impossible timeline.

**Note:** with the spec's rules applied to this dataset, ~30% of candidates
are flagged — see `ARCHITECTURE.md` for an explanation of why and which
rules are responsible.

## Tuning

Every constant is in `src/config.py` with a docstring. Override via env
vars (`RANKER_*`) or a `.env` file. Weights that don't sum to 1.0 raise
a `ValidationError` on startup.

## Tests

```bash
uv run pytest tests/ -v --cov=src
```

36 tests, covering parsers, honeypot rules, all scoring components, and
tie-breaking. All pass.

## Docker

```bash
docker build -t resume-ranker .
docker run -v $(pwd)/data/output:/app/data/output resume-ranker
# or: make docker-build docker-run
```

## Embedding model note

The primary path uses `all-MiniLM-L6-v2` from sentence-transformers,
which requires a one-time model download from HuggingFace (network access
needed at first run, then cached locally). If the model cannot be loaded,
the system automatically falls back to a TF-IDF + TruncatedSVD embedder
(no download required) and logs a clear warning. Both paths produce
valid, L2-normalized embeddings; the TF-IDF fallback runs slightly faster
in CPU-only sandboxes.
