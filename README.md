# WholeDoc Bench

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/built%20with-uv-purple)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

WholeDoc Bench measures how well a document parser turns PDFs, Office files and images into
Markdown and layout JSON. It scores the output against answer keys that a second person checked,
on seven tracks: text, reading order, structure, tables, barcodes, charts and visual grounding.

It runs cloud APIs, LLMs and local tools the same way. Every run saves the raw responses, so a
result can be scored again offline, and the leaderboard shows what each cloud run cost.

## Paper

[WholeDoc Bench: Continuity, Fidelity and Machine-Readable Content in Document Parsing](paper.pdf)
describes how the documents were collected and annotated, how each track is scored, and the
results of 13 parsers in 19 configurations. The paper is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Leaderboard

The leaderboard is published with the first dataset release. You can build your own from any
set of results with `wholedoc-bench report`.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and, for this example, `pdftotext` from Poppler.

```bash
# Install
uv sync

# Download the public sample, pinned to its tag
uvx --from huggingface_hub hf download nutrientdocs/wholedoc-bench-sample --repo-type dataset --revision v1 --local-dir wholedoc-bench-sample

# Run a parser on it
uv run wholedoc-bench run --provider pdftotext --corpus wholedoc-bench-sample --out runs/pdftotext

# Score the outputs against the answer keys
uv run wholedoc-bench score --corpus wholedoc-bench-sample --predictions runs/pdftotext --results results

# Build the leaderboard: printed, and written to LEADERBOARD.md
uv run wholedoc-bench report --results results
```

`pdftotext` reads only the text layer of a PDF. It returns nothing for 3 of the 4 scans in the
sample, so the leaderboard marks it `FAIL` on the coverage gate.

## What it measures

Each document is scored only on the tracks and sub-metrics its answer key supports. A track
score is the weighted mean of its sub-metrics. The overall score is the mean of the track scores.

| Track | What it checks | Sub-metrics | Reads |
|---|---|---|---|
| **T1** Text fidelity | The words, sentences and diacritics are all there | `word_bag_f1`, `sentence_integrity`, `orthography_diacritics`, `block_edit_similarity` | Markdown |
| **T2** Reading order | Text comes in the order a person reads it | `order_tests`, `block_order_similarity` | Markdown |
| **T3** Structure and formatting | Headings, paragraphs, captions, footnotes, bold and italic, page breaks and fonts | `heading_tree`, `block_integrity`, `relations`, `text_styles`, `page_breaks`, `typography` | Markdown and layout JSON |
| **T4** Tables | Cells, headers, rows and whole tables | `teds`, `header_matched_rows`, `table_integrity`, `cell_neighbour_tests` | Markdown and layout JSON |
| **T5** Machine-readable content | Barcode payloads, and machine-readable zones (MRZ) with their text and check digits | `barcode_payloads`, `mrz_identified`, `mrz_text`, `mrz_validity` | Markdown and layout JSON |
| **T6** Charts | Data points, series, chart type and title | `data_points`, `series`, `chart_type`, `chart_title` | Markdown or layout JSON |
| **T7** Visual grounding | Element boxes, element classes and table cell boxes on the page | `element_boxes`, `element_classes`, `table_cell_boxes` | Layout JSON |

**Coverage gate.** A parser must return a valid result for at least 95% of the documents to be
ranked first. A parser below that is listed after the others and marked `FAIL`.

**Cost.** The leaderboard prices each cloud run per 1,000 pages, from the usage in its saved
responses and the dated list prices in [`prices/`](prices). Cost never changes the ranking.

## Supported parsers

| Parser | `--provider` | Install | Settings |
|---|---|---|---|
| Nutrient Data Extraction API | `nutrient` | | `mode`: `structure`, `understand` (default) or `agentic` |
| AWS Textract | `textract` | `--extra textract` | none |
| Azure AI Document Intelligence | `azure-di` | | `features`: `barcodes`, `formulas`, `ocrHighResolution`, `styleFont` (all on by default) |
| Reducto | `reducto` | | [`configs/reducto/`](configs/reducto) |
| Extend | `extend` | | [`configs/extend/`](configs/extend) |
| LlamaParse | `llamaparse` | | [`configs/llamaparse/`](configs/llamaparse): `fast`, `cost_effective`, `agentic` (default), `agentic_plus` |
| Databricks `ai_parse_document` | `databricks` | | none |
| Anthropic Claude | `anthropic` | `--extra anthropic` | [`configs/anthropic/`](configs/anthropic): Claude Opus 5.5 and Claude Fable 5.1 |
| OpenAI GPT | `openai` | `--extra openai` | [`configs/openai/`](configs/openai): GPT-6.1 Sol and GPT-6 Luna |
| Docling | `docling` | `--extra parsers` | `chart_extraction` (on by default) |
| MarkItDown | `markitdown` | `--extra parsers` | none |
| PyMuPDF | `pymupdf` | `--extra parsers` | none |
| pdftotext | `pdftotext` | Poppler on your system | none |

Pass settings with `--config FILE`, a JSON object. A parser without settings accepts only `{}`.

<details>
<summary>Notes on some parsers</summary>

- **Claude and GPT** get each page with ParseBench's layout prompt. After a page that ends with a
  paragraph, list item or table, the next request quotes that element, so the model can mark
  what continues across the break. Claude Fable 5.1 needs an organization with 30-day data
  retention.
- **Docling** with `chart_extraction` reads each bar, line and pie chart with Granite Vision and
  writes its data as a table. The first run downloads about 8 GB, and without a GPU each chart
  takes minutes.
- **Azure** bills every feature except `barcodes` per page on top of the layout model, and
  applies them to PDF and image files only.
- **Databricks** writes page images to a Unity Catalog volume to measure its boxes, and the
  adapter deletes them after each document.

</details>

## Dataset

The public sample is on Hugging Face as
[`nutrientdocs/wholedoc-bench-sample`](https://huggingface.co/datasets/nutrientdocs/wholedoc-bench-sample):
10 documents (28 pages) from the benchmark, with their answer keys. It is a release folder:

```
documents.jsonl       one row per document: its files, checksums, tracks and supported metrics
documents/            the documents
groundtruth/          one answer key per document
SHA256SUMS            a checksum for every file
release.json          the release tag and counts
```

Download the whole release, pinned to its tag, as in the [quick start](#quick-start). The rows
alone are not enough to score. Then pass the folder with `--corpus`. It does not need to be
inside this checkout. Every command checks the release first and stops if a file is missing or
changed.

The row and prediction formats are in [`schemas/`](schemas).

## Usage

### Run a cloud parser

```bash
cp -n .env.example .env && chmod 600 .env    # then fill the keys you need
uv run --env-file .env wholedoc-bench run --provider nutrient --config nutrient.json --corpus release --out runs/nutrient-agentic
uv run --env-file .env --extra anthropic wholedoc-bench run --provider anthropic --config configs/anthropic/fable-5-1-high.json --corpus release --out runs/claude-fable-5-1-high
uv run wholedoc-bench score --corpus release --predictions runs/nutrient-agentic --results results
```

Here `nutrient.json` holds `{"mode": "agentic"}`. Use a new `--out` folder for each parser and
configuration, and a new `--results` folder for each evaluation. Existing results are never
overwritten.

- **Resume offline.** Repeat `run` with `--from-raw` to convert the saved responses again
  without calling the parser, for example after an adapter fix.
- **Leave a document out** with `--exclude-document ID`. Use the same selection for `run` and
  `score`.
- **Failures say where they happened:** before the parser was called (`input`), while calling it
  (`fetch`), or after it answered (`normalize`, an adapter bug). `report` refuses to rank results
  with `normalize` failures.

### Score your own parser

A parser without an adapter here is scored from its output files.

1. Write one prediction per document as `<id>.json`, one per row of `documents.jsonl`, in a
   folder that holds nothing else. [`schemas/prediction.schema.json`](schemas/prediction.schema.json)
   describes the format, and
   [`schemas/examples/prediction-minimal.json`](schemas/examples/prediction-minimal.json) is the
   smallest valid file.
2. Write a claims file. Its fields are shown as claims, never as verified facts:

   ```json
   {
     "parser": "My Parser",
     "version": "2.0",
     "settings": {"mode": "accurate"},
     "ran_at": "2026-10-18",
     "ran_where": "a laptop, offline",
     "training_statement": "Not trained or tuned on this release."
   }
   ```

3. Import, score and report:

   ```bash
   uv run wholedoc-bench import --corpus release --predictions my-outputs --out imports/my-parser --name my-parser --claims claims.json
   uv run wholedoc-bench score --corpus release --predictions imports/my-parser --results results
   uv run wholedoc-bench report --results results
   ```

The import checks every file first and refuses the whole import if one is wrong. Use
`--missing-is-failure` to score a missing document as a failed parse. Imported rows are marked
`(imported)` and are ranked with wholedoc-bench runs only with `report --allow-mixed-origins`.

### Compare results

Results from different releases or setups are never ranked together. To compare parsers, score
them all again with the same checkout and Python environment: even a Python patch update or an
adapter edit changes what a result is compared on. `report --prices-as-of DATE` prices runs at
the rates of another day.

## Configuration

### API keys

Copy [`.env.example`](.env.example) to `.env`, fill only what you use, and pass it with
`uv run --env-file .env`. Keep keys out of `--config` files: the effective configuration is
saved in plain text with the predictions and results.

| Parser | Variables |
|---|---|
| Nutrient | `NUTRIENT_API_KEY` |
| AWS Textract | the AWS SDK credential chain (`AWS_PROFILE` or access keys) and `AWS_DEFAULT_REGION` |
| Azure | `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`, `AZURE_DOCUMENT_INTELLIGENCE_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Reducto | `REDUCTO_API_KEY` |
| Extend | `EXTEND_API_KEY`, optional `EXTEND_WORKSPACE_ID` |
| LlamaParse | `LLAMA_CLOUD_API_KEY` |
| Databricks | `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (or `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET`), `DATABRICKS_VOLUME_PATH` |

A missing key stops a run before any output changes. Offline replay and scoring need no key.

### CLI reference

| Command | What it does |
|---|---|
| `wholedoc-bench run` | Runs a parser on every document and saves its predictions and raw responses |
| `wholedoc-bench score` | Scores saved predictions against the answer keys |
| `wholedoc-bench import` | Imports predictions made outside wholedoc-bench so they can be scored |
| `wholedoc-bench report` | Builds the leaderboard from scored results |

Add `--help` to any command for its options.

## Citation

```bibtex
@misc{wholedocbench2026,
  title  = {WholeDoc Bench: Continuity, Fidelity and Machine-Readable Content in Document Parsing},
  author = {Moldovan, Denis and Sookun, Shrevin and Mogage, Nicolae and Pizur, Matus and
            Mitrea, Costinel and Houssin, Olivier and Alby, Etienne and Carlier, Coralie and
            Mart{\'\i}n, Daniel},
  year   = {2026},
  url    = {https://github.com/PSPDFKit-labs/wholedoc-bench}
}
```

## License

WholeDoc Bench is licensed under the [Apache License 2.0](LICENSE). The paper, `paper.pdf`, is
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
