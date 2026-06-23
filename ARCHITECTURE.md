# Architecture

## Pipeline stages

```
job_description.docx   candidates.jsonl
        |                     |
        v                     v
  docx_parser.py      candidate_parser.py
  (full text)         (streaming, validates
                       each record against
                       pydantic schema)
        |                     |
        |            honeypot_detector.py
        |            (6 rules, excluded
        |             before scoring)
        |                     |
        |            text_builder.py
        |            (rich text blob
        |             per candidate)
        |                     |
        +----------+----------+
                   |
            embedder.py
            (sentence-transformers
             primary / TF-IDF fallback)
                   |
              index.py
              (FAISS IVFFlat
               inner-product)
                   |
            JD embedding query
                   |
            top-K search results
                   |
              scorer.py
              (5-factor hybrid:
               semantic + title +
               skill + behavioral
               + experience)
                   |
           aggregator.py
           (sort, top-100,
            tie-break by ID)
                   |
            reasoner.py
            (rule-based, no LLM)
                   |
           submission.csv
           (1 header + 100 rows,
            self-validated before exit)
```

## Honeypot rules and dataset-specific behaviour

The spec defines 6 honeypot detection rules. Three of them (inverted
salary, job-before-education, impossible timeline) fire on approximately
12-19% of real candidates each in this dataset, producing a combined
~30% exclusion rate. This is a consequence of applying the spec's rules
unchanged to a dataset whose generator does not fully honour those
constraints (e.g. ~19% of candidates have `salary.min > salary.max`).

Per an explicit decision, all 6 rules are kept exactly as specified. The
practical implication is that roughly 30,000 of 100,000 candidates are
excluded before FAISS indexing. The remaining ~70,000 are indexed and
the top 100 are returned; the quality of those 100 is unaffected.

## Embedding primary vs fallback

Primary: `sentence-transformers/all-MiniLM-L6-v2`, fully local CPU
inference after a one-time HuggingFace model download. Produces
384-dimensional L2-normalised vectors.

Fallback (automatic, logged as WARNING): TF-IDF + TruncatedSVD,
128-dimensional. Fitted once on all candidate texts together; every
candidate and the JD live in the same vector space. No download required.
Typically faster in CPU-only sandboxes with no HF access.

Both paths use the same FAISS IVF inner-product index downstream.

## Config

Every tunable constant is in `src/config.py` and overridable at runtime
via `RANKER_*` environment variables. Weights are validated to sum to 1.0
on startup.

## Performance (measured, CPU-only, 1 core, 3.9 GB RAM)

| Stage                          | Time   |
|-------------------------------|--------|
| JD parse                      | < 0.1s |
| Stream + honeypot filter       | ~35s   |
| Text build (100K)             | ~2s    |
| TF-IDF embed (fallback, 70K)  | ~35s   |
| FAISS build + search          | ~0.5s  |
| Hybrid scoring + aggregate    | ~1s    |
| Reasoning + write CSV         | < 0.1s |
| **Total**                     | **~77s** |

In an environment with HuggingFace access, the sentence-transformers
primary path replaces the TF-IDF embed stage and runs at ~3s/1000 docs
on CPU.
