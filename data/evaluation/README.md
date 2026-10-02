# DeepShield-RAG Evaluation Benchmark

This directory holds version-controlled, human-reviewed benchmark records. Generated model
outputs and run logs belong in ignored local artifacts until their contents and size have been
reviewed.

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

Validate the small synthetic format fixture and its corpus references:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli validate `
  data/fixtures/evaluation-benchmark.valid.json --repository-root .
```

The fixture demonstrates the format only. It is not part of the 600-prompt benchmark and must
not be reported as an experimental result. Final experiment runners will require a benchmark
whose status is `frozen`:

```powershell
uv run --project backend --no-sync python -m deepshield_api.evaluation.cli validate `
  data/evaluation/f1-security-benchmark.v1.json --repository-root . --require-frozen
```
