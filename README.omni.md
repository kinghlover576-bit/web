# OmniVerse (Apex AI Web Nexus)

Monorepo scaffold for the OmniVerse ambient neuro-symbolic web intelligence MVP.

## Quickstart

Requirements: Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
pytest -q
uvicorn api.main:app --reload
```

Then visit: `http://127.0.0.1:8000/healthz`.

## Structure

```
ingestion/        # future: crawlers
processing/       # future: LLM extraction & swarms
index/            # future: ES/Milvus ETL
search/           # future: retrieval & ranking
api/              # FastAPI endpoints (MVP)
orchestration/    # future: Ray/K8s jobs
infra/            # future: Terraform/Helm
ops/              # future: observability & runbooks
tests/            # pytest
docs/             # blueprints & ADRs
```

## Licensing

Apache-2.0.