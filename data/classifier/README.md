# C2 Risk-Classifier Data

This directory records the provenance and validation contracts for the lightweight C2 risk
classifier. Classifier data is deliberately separate from the 600-case C0/C1/C2 benchmark.
Benchmark prompts may be used only for the final system comparison, never for classifier
training, model selection, threshold tuning, or feature design.

## Current source decision

The frozen source registry selects two permissively licensed datasets for different purposes:

| Source                              | License    | Project use                        |
| ----------------------------------- | ---------- | ---------------------------------- |
| `jackhhao/jailbreak-classification` | Apache-2.0 | Development source                 |
| `SPML_Chatbot_Prompt_Injection`     | MIT        | Independent contextual stress test |

The jackhhao source supplies benign and jailbreak examples, but no stable family identifier.
Before creating project splits, the collection pipeline must normalize the text and derive
near-duplicate clusters so related prompts remain together. The upstream train/test labels are
not accepted blindly as the project's leakage boundary.

SPML is intentionally **not** a training source. Its authors caution that its limited attack
techniques are not sufficient for a general prompt-injection detector, and its labels depend on
paired system-prompt context. It is useful as an external stress test precisely because it comes
from a separate methodology.

Both sources are pinned to immutable 40-character revisions. Registry v2 records their licenses,
columns, label mappings, intended use, citations, limitations, and the SHA-256 checksum of every
selected file. It supersedes v1 because collection proved that SPML's real column is
`Prompt injection`, not `Prompt Injection`; v1 remains committed as a transparent historical
record. External files remain local and must not be committed without a redistribution review.

## Verified source collection

The collection receipt records evidence from the real pinned files without redistributing them:

| Partition source | Files |   Rows | Empty prompt rows |
| ---------------- | ----: | -----: | ----------------: |
| Development      |     2 |  1,306 |                 0 |
| External test    |     1 | 16,012 |                 1 |

The single empty SPML user prompt is retained in the immutable raw download and recorded in the
receipt. Deterministic preprocessing must exclude it and report that exclusion. Collection does
not accept new sources: it downloads only registry-listed files at their exact revisions, checks
size and SHA-256, parses CSV with strict headers and labels, verifies row totals, and refuses to
overwrite changed local artifacts. Raw files live under ignored `data/classifier/raw/`; the
committed receipt is `source-collection.v1.json`.

## Derived candidate dataset

The project artifact uses four partitions:

- `train`: fit the lightweight model.
- `validation`: choose features, hyperparameters, and the operating threshold.
- `test`: measure held-out classifier performance after design choices are frozen.
- `external-test`: evaluate transfer to SPML without fitting or tuning on it.

Every example records its source record, group, split, binary label, fine-grained category,
exact text checksum, synthetic flag, and review evidence. A frozen artifact must contain both
labels in every partition, keep each group in one partition, contain no duplicate text hashes,
reference the frozen source-registry checksum, and record a checksum proving that benchmark IDs
were excluded. The contract also requires at least 1,000 reviewed examples: 500 train, 100
validation, 200 internal test, and 200 external test.

These checks prevent exact and group leakage. The measured quality of near-duplicate clusters and
upstream labels still requires human review; schema validation alone is not evidence that the
classifier works.

Preprocessing version 1 deterministically applies NFKC Unicode normalization, whitespace
collapse, case-insensitive exact deduplication, an 8,000-character safety limit, and token
trigram near-duplicate detection at Jaccard similarity `0.82`. Development examples are grouped
before a seeded `60% / 15% / 25%` train/validation/test assignment. The full SPML source remains
external-test only. External rows matching development data exactly or at the same near-duplicate
threshold are excluded before evaluation.

The verified run transformed 17,318 collected rows into 17,194 draft candidates:

| Split         |   Rows | Groups | Benign | Adversarial |
| ------------- | -----: | -----: | -----: | ----------: |
| Train         |    760 |    697 |    365 |         395 |
| Validation    |    197 |    185 |    109 |          88 |
| Test          |    324 |    302 |    163 |         161 |
| External test | 15,913 | 15,913 |  3,372 |      12,541 |

The run excluded one empty row, five rows beyond the declared length limit, and 118 exact
duplicates. It found no exact or threshold-level near-duplicate overlap between development and
external-test data. The 1,281 retained development examples formed 1,184 groups; 54 groups had
multiple members and the largest contained 12 prompts. These are preprocessing measurements, not
classifier-performance results. The external split is naturally imbalanced, so later evaluation
must report macro metrics and per-class results rather than accuracy alone.

The 21.7 MB candidate dataset remains local at
`data/classifier/processed/risk-dataset.v1.json`. Its checksum and all aggregate evidence are
committed in `preprocessing-report.v1.json`; the JSON Schema is
`preprocessing-report.schema.json`. Candidates remain `draft`. Human review and explicit
benchmark-exclusion evidence are still required before a smaller training artifact can be frozen.

## Benchmark-safe review selection

Selection version 1 compares every normalized classifier candidate with all 600 benchmark
prompts and the seven synthetic retrieved-text payloads. It rejects exact matches and token
trigram near-duplicates at Jaccard similarity `0.82`, then uses the fixed
`deepshield-c2-selection-v1` seed to select complete groups. This prevents benchmark examples
from entering classifier fitting, threshold selection, or classifier testing and prevents related
classifier prompts from being split across partitions.

The verified run found no exact or threshold-level near-duplicate overlap among the 17,194
candidates. It selected exactly 1,000 examples for review:

| Split         | Selected | Groups | Benign | Adversarial |
| ------------- | -------: | -----: | -----: | ----------: |
| Train         |      500 |    452 |    250 |         250 |
| Validation    |      100 |     93 |     50 |          50 |
| Test          |      200 |    189 |    100 |         100 |
| External test |      200 |    200 |    100 |         100 |

The selected dataset and its 1,000-row review worksheet remain under the ignored
`data/classifier/processed/` directory. Only aggregate counts, rules, paths, and cryptographic
checksums are committed in `selection-report.v1.json`; this avoids redistributing source text
before the licensing review is finalized. Zero detected overlap is evidence about this defined
check, not proof that the datasets are semantically unrelated. All examples remain `draft`, and
the worksheet contains blank reviewer fields. Classifier training is blocked until a human
reviews the labels and the approved dataset is explicitly frozen.

## Commands

Run these from the repository root:

```powershell
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema sources --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema collection --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema dataset --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema preprocessing --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema selection --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-sources `
  data/classifier/source-registry.v2.json
uv run --project backend --no-sync python -m deepshield_api.classifier.cli collect-sources `
  data/classifier/source-registry.v2.json `
  --output data/classifier/source-collection.v1.json --repository-root .
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-collection `
  data/classifier/source-collection.v1.json --repository-root . --require-files
uv run --project backend --no-sync python -m deepshield_api.classifier.cli preprocess `
  data/classifier/source-collection.v1.json `
  --output data/classifier/processed/risk-dataset.v1.json `
  --report data/classifier/preprocessing-report.v1.json --repository-root .
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-preprocessing `
  data/classifier/preprocessing-report.v1.json --repository-root . --require-dataset
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-dataset `
  data/classifier/processed/risk-dataset.v1.json --repository-root .
uv run --project backend --no-sync python -m deepshield_api.classifier.cli select-review-candidates `
  --repository-root .
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-selection `
  data/classifier/selection-report.v1.json --repository-root . --require-artifacts
```

The committed dataset fixture validates only the format. It is synthetic, is not classifier
training data, and must not be reported as an experimental result.
