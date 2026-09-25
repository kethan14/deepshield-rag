# DeepShield-RAG Data

This directory contains the source records, synthetic fixtures, evaluation datasets,
and locally generated corpus artifacts used by DeepShield-RAG.

The demonstration corpus is limited to public F-1 student policy material from
authoritative U.S. government sources. The system is an academic research prototype
and does not provide legal advice.

## Planned structure

```text
data/
├── README.md
├── manifests/   # Source schema and later approved collection metadata
├── fixtures/    # Small, explicitly synthetic records used by automated tests
├── evaluation/  # Version-controlled benchmark prompts and reviewed labels
├── incoming/    # Temporary browser downloads; generated locally
├── raw/         # Downloaded source files; generated locally
├── interim/     # Extracted and normalized text; generated locally
├── processed/   # Final chunks and metadata; generated locally
└── indexes/     # Embedding and vector-database files; generated locally
```

Only manifests, synthetic fixtures, and approved evaluation records belong in Git.
Raw downloads and generated corpus artifacts remain local unless their redistribution
rights and repository size have been reviewed.

## Source requirements

A document may enter the main corpus only when it:

1. Comes from an authoritative U.S. government source, such as USCIS, DHS, ICE/SEVP,
   Study in the States, or eCFR.
2. Has a stable public URL and an identifiable publishing agency.
3. Is relevant to the defined F-1 policy-assistant scope.
4. Has its retrieval date, content type, checksum, and source metadata recorded.
5. Passes file-type, size, and extraction validation.
6. Contains no intentionally collected personal student records.

Blogs, discussion forums, search-result summaries, and model-generated explanations
must not be treated as authoritative corpus sources.

## Required source metadata

Each collected source will have a manifest record containing at least:

- Stable source identifier
- Document title
- Publishing agency
- Canonical URL
- Source type and media type
- Retrieval timestamp
- Published or last-updated date when available
- Scope tags
- Local raw-file path
- SHA-256 content checksum
- Collection status
- Rights or redistribution notes

The checksum makes an unnoticed source change detectable. A changed document must be
recorded as a new version rather than silently replacing the earlier record.

## Data lifecycle

```text
Official source registry
        ↓
Validated download
        ↓
Immutable raw file and checksum
        ↓
Text extraction and normalization
        ↓
Traceable chunks with source metadata
        ↓
Embeddings and vector index
        ↓
RAG retrieval and controlled evaluation
```

Raw files are treated as immutable inputs. Interim text, processed chunks, embeddings,
and indexes must be reproducible from the recorded source manifest and project code.

## Privacy and experiment safety

DeepShield-RAG must not collect or store real student records, private university data,
credentials, access tokens, or production secrets.

Security experiments will use only:

- Fictional student profiles
- Synthetic identifiers and canary secrets
- Controlled adversarial documents
- Mock tools that cannot perform real external actions

Synthetic records must be clearly labeled. They must not be mixed with factual policy
documents or presented as real people.

## Ground truth and evaluation boundaries

Benchmark answers must be supported by reviewed passages from the frozen official
corpus. Model-generated answers cannot serve as ground truth without human review.

Training examples for the lightweight risk classifier must remain separate from the
final validation and test prompts. Duplicate or near-duplicate prompts must not cross
those splits. Final benchmark prompts must remain frozen while the three system
configurations are compared.

## Reproducibility

Collection and preprocessing will be performed by versioned scripts rather than
unrecorded manual copying. Each run will record its configuration, source-manifest
version, timestamps, and generated-file checksums.

If automated collection is not permitted or is unreliable for a source, the document
will be downloaded manually and registered through the same validation process.
Website access rules, terms, and redistribution restrictions must be respected.

The automated collector refuses unapproved redirect domains, non-HTTPS final URLs,
unexpected media types, empty responses, oversized responses, and attempts to replace
an existing raw artifact with different bytes. It writes a new manifest version and
never overwrites the approved input manifest.

When an approved government site blocks scripted collection, save its public document
through a normal browser into `data/incoming/`, then register it with `import-file`.
The importer rejects symbolic links, oversized files, empty files, invalid UTF-8 HTML,
and bytes that do not match the declared HTML or PDF type. It applies the same immutable
raw-path, checksum, timestamp, and new-manifest rules as automated collection.

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli import-file `
  data/manifests/input-manifest.json source-id --document-version 1 `
  --input data/incoming/source.html --output data/manifests/output-manifest.json `
  --repository-root .
```

## Text extraction and normalization

The extraction command accepts only a fully collected manifest. Before reading a raw
artifact, it verifies that the file remains inside `data/raw/` and that its SHA-256
checksum matches the collected manifest. HTML extraction removes scripts, styles,
navigation, forms, and other non-content elements before selecting the primary page
content. PDF extraction processes every unencrypted page. Both paths apply deterministic
Unicode and whitespace normalization.

Normalized UTF-8 text is written immutably under `data/interim/`. A versioned extraction
report records every input and output checksum, extraction method, path, character count,
and line count without committing the extracted document text itself.

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli extract `
  data/manifests/f1-policy-collected-manifest.v18.json `
  --output data/manifests/f1-policy-extraction-report.v1.json `
  --report-version 1 --repository-root .
```

## Manifest validation

`manifests/source-manifest.schema.json` is generated deterministically from the
Pydantic models in `backend/src/deepshield_api/corpus`. Do not edit it manually.

Generate the schema from the repository root:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli schema
```

Check for schema drift:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli schema --check
```

Validate the committed synthetic fixture:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli validate data/fixtures/source-manifest.valid.json
```

The fixture demonstrates the file format only. Its synthetic URL is not an approved
corpus source and must never be used as factual policy evidence.

Validate the frozen version 1 F-1 source registry:

```powershell
uv run --project backend --no-sync python -m deepshield_api.corpus.cli validate data/manifests/f1-policy-source-manifest.v1.json
```

The version 1 registry contains planned metadata only. It does not prove that a document has
been downloaded or reviewed, and it must not be treated as a collected corpus manifest until
each record has retrieval metadata, a raw path, and a SHA-256 checksum.
