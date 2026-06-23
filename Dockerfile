FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN mkdir -p data/input data/output artifacts

CMD ["uv", "run", "python", "-m", "scripts.rank", \
     "--candidates", "data/input/candidates.jsonl", \
     "--jd", "data/input/job_description.docx", \
     "--output", "data/output/submission.csv"]
