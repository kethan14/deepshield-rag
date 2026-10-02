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

## Validate corpus metadata

Export the deterministic source-manifest JSON Schema:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli schema
```

Confirm that the committed schema matches the current Pydantic models:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli schema --check
```

Validate a source manifest or synthetic fixture:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli validate data/fixtures/source-manifest.valid.json
```

Schema drift is checked automatically by Backend CI.

## Collect an approved source

The collector only accepts a source that already exists as a validated `planned`
manifest record. It enforces the approved publisher domain, HTTPS redirects, declared
media type, a bounded response size, deterministic raw paths, SHA-256 checksums, and
immutable raw files. The command always writes a new manifest version rather than
overwriting its input.

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli collect `
  data/manifests/f1-policy-v1.planned.json `
  approved-source-id `
  --document-version 1 `
  --output data/manifests/f1-policy-v2.collected.json `
  --repository-root .
```

Do not run collection until the authoritative source registry has been reviewed and
frozen. Raw artifacts remain under ignored `data/raw/` paths and must not be committed.

## Extract normalized corpus text

After all source records are collected, verify their checksums and extract deterministic
normalized text with:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli extract `
  data/manifests/f1-policy-collected-manifest.v18.json `
  --output data/manifests/f1-policy-extraction-report.v1.json `
  --report-version 1 --repository-root .
```

The generated report stores input/output checksums and measurable extraction statistics.
The normalized text remains in the ignored `data/interim/` directory and can be rebuilt
from the immutable raw artifacts and committed report configuration.

## Build deterministic retrieval chunks

After extraction, create the versioned retrieval dataset and its traceability report:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli chunk `
  data/manifests/f1-policy-extraction-report.v1.json `
  --output-chunks data/processed/f1-policy-chunks.v1.jsonl `
  --output-report data/manifests/f1-policy-chunking-report.v1.json `
  --report-version 1 --repository-root .
```

The default policy limits chunks to 1,800 characters, targets 250 characters of overlap,
and prefers line or sentence boundaries. Each JSONL record carries exact offsets,
checksums, and official-source metadata. The generated text dataset stays ignored under
`data/processed/`; its committed report records the complete dataset checksum and
reproducible measurements.

## Build and search the local vector index

Embed the frozen chunk dataset and create the exact-search index:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli build-index `
  data/manifests/f1-policy-chunking-report.v1.json `
  --output-vectors data/indexes/f1-policy-v1/vectors.npy `
  --output-report data/manifests/f1-policy-vector-index-report.v1.json `
  --report-version 1 --repository-root . `
  --model-cache data/indexes/models --threads 1
```

Search the verified index:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli search `
  data/manifests/f1-policy-vector-index-report.v1.json `
  "Can an F-1 student participate in curricular practical training?" `
  --repository-root . --model-cache data/indexes/models --top-k 5 --threads 1
```

The index uses the 384-dimensional `BAAI/bge-small-en-v1.5` embedding model through
FastEmbed on the CPU. The implementation pins the exact ONNX model-repository commit,
normalizes every `float32` vector, and ranks all 1,122 chunks with exact cosine
similarity. Exact search is intentionally simpler and fully deterministic at this
student-scale corpus size; an approximate vector database would add infrastructure
without a meaningful retrieval benefit here.

The downloaded model and generated `.npy` matrix remain under ignored `data/indexes/`.
The committed vector-index report records their checksums, dimensions, ordered chunk-ID
hash, dependency version, model identity, and every upstream corpus checksum. Loading or
searching the index fails if any linked artifact has changed.

## Run the C0 baseline RAG API

The C0 pipeline combines the verified exact-cosine retriever with an
OpenAI-compatible chat-completion endpoint. This keeps the project independent of one
vendor: the same server-side interface can use a local Ollama server or an explicitly
configured hosted endpoint. API keys stay in the backend environment and are never sent
to the browser.

The research configuration freezes `gpt-5.4-mini-2026-03-17` for C0, C1, and C2.
Copy `.env.example` to `.env`, add a server-side API key, and keep the committed values:

```dotenv
DEEPSHIELD_LLM_BASE_URL=https://api.openai.com/v1
DEEPSHIELD_LLM_MODEL=gpt-5.4-mini-2026-03-17
DEEPSHIELD_LLM_REASONING_EFFORT=none
DEEPSHIELD_LLM_MAX_TOKEN_PARAMETER=max_completion_tokens
DEEPSHIELD_LLM_API_KEY=replace-in-local-env-only
```

The key belongs only in the uncommitted `.env` file. Plain HTTP remains available only
for a loopback development server. Start the backend using the command in **Run the
API**, then submit a question:

```powershell
$body = @{ question = "Can an F-1 student participate in CPT?"; top_k = 5 } |
  ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/assistant/query `
  -ContentType "application/json" -Body $body
```

The response contains the C0 answer, every retrieved passage, citation labels, invalid
model-generated citation labels, model identities, finish reason, and provider-reported
token usage. This is the intentionally minimal comparison baseline: it has a standard
grounded-answer prompt but no attack classifier, prompt-injection detector, leakage
scanner, authorization layer, or final-answer guardrail.

The versioned research record is `config/experiments/generation-model.v1.json`. It pins
the provider, endpoint, exact model snapshot, zero reasoning effort, temperature, token
limit, comparison configurations, selection date, and price snapshot. A ChatGPT
subscription is not an API credential, and no paid request is made merely by running the
automated test suite.

### Why this model was selected

The development laptop has a 22-thread Intel Core Ultra 7 155H, 16 GB RAM, and integrated
Intel Arc graphics. It can run a small quantized local model for demonstrations, but
hundreds of repeated C0/C1/C2 experiments would be slow and would introduce local runtime
variation. A hosted snapshot gives higher throughput and a stable model identity.

`gpt-5.4-mini-2026-03-17` was selected because it provides a pinned snapshot, supports
Chat Completions, is suitable for high-volume workloads, and remains affordable. At the
October 2, 2026 documented rates of $0.75 per million input tokens and $4.50 per million
output tokens, 1,800 runs averaging 3,000 input and 350 output tokens would cost about
$6.89. This is a planning estimate, not a measured result; retries, longer prompts, price
changes, and additional development runs increase it. Experiments will use only public
policy text and clearly synthetic security records—never real student information.

## Validate the evaluation benchmark

The benchmark contract is defined in `backend/src/deepshield_api/evaluation`, with its
generated schema in `data/evaluation/benchmark-manifest.schema.json`. The planned study uses
600 prompts: 200 development cases for guardrail tuning and 400 held-out test cases for the
primary C0/C1/C2 comparison. Each partition is balanced between benign and adversarial cases.

Validate schema drift and the small synthetic format fixture from the repository root:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli schema --check
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli validate `
  data/fixtures/evaluation-benchmark.valid.json --repository-root .
```

This fixture tests the data contract only; it is not an experimental dataset or result. The
final experiment runner will accept only a reviewed artifact with `status: frozen`. See
`data/evaluation/README.md` for the threat taxonomy, leakage controls, and labeling protocol.

## Project structure

```text
backend/
├── pyproject.toml
├── uv.lock
├── src/
│   └── deepshield_api/
│       ├── __init__.py
│       ├── main.py
│       ├── corpus/
│       │   ├── __init__.py
│       │   ├── chunking.py
│       │   ├── cli.py
│       │   ├── collection.py
│       │   ├── extraction.py
│       │   ├── indexing.py
│       │   ├── manifest.py
│       │   └── models.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── cli.py
│       │   ├── manifest.py
│       │   └── models.py
│       └── rag/
│           ├── __init__.py
│           ├── experiment.py
│           ├── generation.py
│           ├── models.py
│           ├── router.py
│           ├── runtime.py
│           └── service.py
└── tests/
    ├── test_corpus_collection.py
    ├── test_corpus_chunking.py
    ├── test_corpus_extraction.py
    ├── test_corpus_indexing.py
    ├── test_corpus_manifest.py
    ├── test_corpus_models.py
    ├── test_evaluation_manifest.py
    ├── test_evaluation_models.py
    ├── test_rag_baseline.py
    ├── test_rag_experiment.py
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
