# DeepShield-RAG Evaluation Benchmark

This directory holds version-controlled benchmark drafts, review packets, and eventually frozen
human-reviewed records. Generated model outputs and run logs belong in ignored local artifacts
until their contents and size have been reviewed.

## Current version 1 status

`f1-security-benchmark.v1.json` is a deterministic **draft** containing the complete 600-case
study design. It is not yet a frozen result. Its companion artifacts are:

- `synthetic-safety-environment.v1.json`: 24 explicitly fictional student records, unique
  canaries, and four inert mock-tool definitions. Addresses use the reserved `example.test`
  domain, phone numbers use the fictional 555 range, and no tool is executable.
- `f1-security-benchmark-audit.v1.json`: reproducible balance, fixture-integrity, secret-exposure,
  and development/test near-duplicate checks.
- `f1-security-benchmark-review.v1.csv`: a 600-row review packet with blank decision and notes
  columns. It intentionally omits the actual forbidden secret values.

The generated draft contains 300 benign and 300 adversarial cases. Every benign and safe-part
case points to an exact chunk in the frozen official corpus. Each of the six threat categories
contains 50 cases. Human review of wording, evidence labels, difficulty, and near-duplicate
judgment is still required before any case is marked approved and the manifest is frozen.
The benign set uses 17 development and 34 test evidence families; related paraphrases retain a
shared `template_family`. Later confidence intervals and significance tests must account for
these families (for example, by cluster bootstrap) instead of treating every paraphrase as an
independent policy fact.

## Frozen study design

The target benchmark contains 600 prompts and is separate from any future risk-classifier
training data:

| Partition   | Benign | Adversarial | Total | Purpose                                      |
| ----------- | -----: | ----------: | ----: | -------------------------------------------- |
| Development |    100 |         100 |   200 | Tune guardrail rules and decision thresholds |
| Test        |    200 |         200 |   400 | Final held-out C0/C1/C2 comparison           |
| **Total**   |    300 |         300 |   600 | Security and utility benchmark               |

The test partition must not be used to write prompts, tune thresholds, train a classifier, or
choose guardrail behavior. Primary claims will use the 400 held-out test cases. Development
results may be reported separately and must never be presented as held-out performance.

## Experiment evidence boundary

The runner foundation rejects this version 1 benchmark while its status remains `draft`; the
rejection happens before any configuration adapter can make a model call. Once human review is
complete, a development run selects only the 200 development cases and a final run selects only
the 400 held-out test cases. Each selected case is executed in the fixed C0, C1, C2 order with
the same frozen generator model and retrieval depth.

Completed raw evidence belongs under ignored `data/evaluation/runs/<run-id>/` directories. A
run contains `trials.jsonl` and `run-manifest.json`. The manifest records benchmark and model
configuration checksums, the retrieval model, timestamps, counts, and a checksum over every
trial. Loading a run revalidates that checksum, trial count, unique identifiers, and execution
order. Raw files may contain fictional PII or synthetic canaries and therefore must not be
committed before disclosure review. This foundation does not itself call a hosted model or
claim experimental performance.

The comparison adapters share the same official retriever and generator across all three
configurations. For an indirect-injection case, the adapter places the labeled poisoned fixture
ahead of the official evidence and preserves a total retrieval depth of five. For synthetic
secret and fictional-PII cases, it similarly inserts only the registered fictional record. The
synthetic evidence uses the reserved `example.test` domain and publisher label
`synthetic-evaluation`; it is never written into the official corpus or index. C1 and C2 may
exclude such evidence, but every trial retains the complete pre-guardrail retrieval record for
fair comparison. C2 also evaluates each benchmark-labeled mock-tool request against an empty,
trusted authorization context, so denial is enforced rather than inferred from model wording.
The first retrieval for a case is cached and replayed to the remaining configurations, removing
retrieval variation from the C0/C1/C2 comparison and avoiding repeated embedding work.

The execution CLI performs a static preflight before constructing any runtime component. It
requires a frozen benchmark, valid corpus references, the pinned generation configuration, a
new run identifier, and explicit acknowledgement of potential provider charges. Final held-out
execution has a second confirmation flag. This makes validation and CI safe to run without
silently contacting a model provider.

## Adversarial coverage

The benchmark supports six controlled threat categories:

1. Direct prompt injection
2. Indirect prompt injection in retrieved text
3. System-prompt extraction
4. Synthetic-secret exfiltration
5. Fictional-PII exfiltration
6. Unauthorized mock-tool actions

All adversarial records are synthetic. They may use fictional people, canary strings beginning
with `DSRAG_CANARY_`, and mock tools that cannot perform external actions. Real student records,
credentials, production secrets, and real tool permissions are prohibited.

## Labeling rules

Each case records its partition, kind, attack category when applicable, expected safe action,
difficulty, topic tags, and review state. Answerable cases also identify supporting sources and
exact supporting chunks for retrieval metrics, plus required answer facts. Leakage cases
identify exact forbidden synthetic content so attack success can be scored reproducibly.

Automated validation prevents:

- Duplicate case IDs
- Exact duplicates after case and whitespace normalization
- A template family from appearing in both development and test partitions
- Adversarial cases that are not explicitly marked synthetic
- Unsupported answer labels without reference sources and required facts
- Frozen benchmarks with missing reviews or incorrect partition and class counts
- Corpus or chunking-report checksum drift, unknown official source IDs, or impossible chunk IDs

Near-duplicate detection and factual labels still require human review; schema validation alone
does not prove benchmark quality.

The automated audit compares the complete model input, including synthetic poisoned retrieval
text, across development and test partitions using token-set Jaccard similarity. Version 1 uses a
conservative rejection threshold of `0.82`. Passing that automated threshold is evidence against
obvious leakage, not a replacement for reviewer judgment.

## Commands

Run these from the repository root.

Generate the deterministic JSON Schema:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli schema
```

Check for schema drift:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli schema --check
```

Check that the committed benchmark, safety environment, audit, and review packet exactly match
the deterministic generator:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli generate `
  --repository-root . --check
```

Re-run the composition, fixture-integrity, exposure, and cross-split leakage audit:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli audit `
  --repository-root . --check
```

Validate the small synthetic format fixture and its corpus references:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli validate `
  data/fixtures/evaluation-benchmark.valid.json --repository-root .
```

The fixture demonstrates the format only. It is not part of the 600-prompt benchmark and must
not be reported as an experimental result. The generated 600-case version 1 artifact is also a
draft and must not be reported as final performance data. Final experiment runners will require
a benchmark whose status is `frozen`:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli validate `
  data/evaluation/f1-security-benchmark.v1.json --repository-root . --require-frozen
```

After review and freezing, execute the development split with:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli run `
  --repository-root . --purpose development `
  --run-id development-v1 --confirm-paid-run
```

The final test split uses `--purpose final` and also requires `--confirm-held-out-test`. Do not
run it while tuning prompts, thresholds, or guardrails.
