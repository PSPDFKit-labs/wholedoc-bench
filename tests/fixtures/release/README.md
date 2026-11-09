---
configs:
- config_name: default
  data_files:
  - split: test
    path: documents.jsonl
---

# release-test v1.0

A document-parsing benchmark: 2 documents and their answer keys. `documents.jsonl` has one
row per document; `document_path` and `groundtruth_path` name its files in `documents/` and
`groundtruth/`, with their SHA-256 checksums.

- **Every key was approved by a second person,** for exactly the files published and for
  exactly the sub-metrics in its `supported_metrics`. A document is scored only on those.
- **Any parser may be run on these documents,** including cloud services.
- **How each key was made** is in `groundtruth_source`; `limits` says what a key does not cover.
- **Checksums:** `SHA256SUMS` lists every file; `release.json` names the source commit.

Download this release with its files, pinned to its tag (`datasets.load_dataset` alone loads
the rows, with paths rather than keys):

```python
from huggingface_hub import snapshot_download

folder = snapshot_download(repo_id="<owner>/<dataset>", repo_type="dataset", revision="v1.0")
```

Score a parser with the benchmark's scorer, pointing `--corpus` at that folder.

## How the keys were made

| groundtruth_source | documents |
|---|---|
| generator | 2 |

## Tracks

| track | documents |
|---|---|
| T1 | 2 |

## Licences

| licence | documents |
|---|---|
| CC-BY-4.0 | 2 |
