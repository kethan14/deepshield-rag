# DeepShield-RAG Backend

The DeepShield-RAG backend is a Python API for the project's retrieval-augmented
generation security experiments, guardrail comparisons, and evidence capture.

The current foundation provides a versioned FastAPI application, a health endpoint,
automated tests, strict type checking, coverage enforcement, and reproducible dependency
locking. RAG, guardrail, and experiment endpoints will be added incrementally.

## Requirements

- Python 3.11.9
- uv 0.12.17

All commands below are run from the repository root.

## Install the locked environment

```powershell
uv sync --project backend --locked --extra dev
```

This creates an isolated environment from `backend/uv.lock` and includes the development
tools required for testing and quality checks.

## Run the API

```powershell
uv run --project backend --no-sync uvicorn deepshield_api.main:app --app-dir backend/src --reload
```

The local service is available at:

- Health endpoint: `http://127.0.0.1:8000/api/v1/health`
- Swagger documentation: `http://127.0.0.1:8000/api/docs`
- ReDoc documentation: `http://127.0.0.1:8000/api/redoc`
- OpenAPI schema: `http://127.0.0.1:8000/api/openapi.json`

Stop the development server with `Ctrl+C`.

## Run backend quality checks

Check formatting:

```powershell
uv run --project backend --no-sync ruff format --check --config backend/pyproject.toml backend
```

Run lint rules:

```powershell
uv run --project backend --no-sync ruff check --config backend/pyproject.toml backend
```

Run strict type checking:

```powershell
uv run --project backend --no-sync mypy --config-file backend/pyproject.toml backend/src backend/tests
```

Run tests with coverage:

```powershell
uv run --project backend --no-sync pytest backend --cov=deepshield_api --cov-report=term-missing
```

The configured minimum backend coverage is 80%.

## Project structure

```text
backend/
├── pyproject.toml
├── uv.lock
├── src/
│   └── deepshield_api/
│       ├── __init__.py
│       └── main.py
└── tests/
    └── test_health.py
```

## Updating dependencies

1. Edit the direct dependency version in `backend/pyproject.toml`.
2. Regenerate the lock file:

   ```powershell
   uv lock --project backend --python 3.11
   ```

3. Synchronize the environment:

   ```powershell
   uv sync --project backend --locked --extra dev
   ```

4. Run all backend quality checks.
5. Commit both `pyproject.toml` and `uv.lock` together.

Do not manually edit `uv.lock`.

## Scope and safety

DeepShield-RAG is an academic research prototype. Its F-1 policy demonstration uses
public official documents and must not be treated as legal advice. Security experiments
will use fictional records, synthetic secrets, and controlled mock tools rather than
real personal information or external actions.
