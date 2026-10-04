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

## Planned derived dataset

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

These checks prevent exact and group leakage. Near-duplicate clustering and label quality still
require measured review; schema validation alone is not evidence that the classifier works.

## Commands

Run these from the repository root:

```powershell
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema sources --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema collection --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli schema dataset --check
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-sources `
  data/classifier/source-registry.v2.json
uv run --project backend --no-sync python -m deepshield_api.classifier.cli collect-sources `
  data/classifier/source-registry.v2.json `
  --output data/classifier/source-collection.v1.json --repository-root .
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-collection `
  data/classifier/source-collection.v1.json --repository-root . --require-files
uv run --project backend --no-sync python -m deepshield_api.classifier.cli validate-dataset `
  data/fixtures/risk-dataset.valid.json --repository-root .
```

The committed dataset fixture validates only the format. It is synthetic, is not classifier
training data, and must not be reported as an experimental result.
