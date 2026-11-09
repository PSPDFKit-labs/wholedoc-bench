# Release and prediction schemas

- [release-row.schema.json](release-row.schema.json): one line of a public release's
  `documents.jsonl`: every release column, with its type, and nothing else.
- [prediction.schema.json](prediction.schema.json): one prediction file, the format
  `wholedoc-bench import` accepts. [examples/prediction-minimal.json](examples/prediction-minimal.json)
  is the smallest valid one. `wholedoc-bench import` also runs the `Prediction` model itself.

Generated from `wholedoc_bench.corpus.release` and `wholedoc_bench.schema`, using JSON Schema draft
2020-12:

```bash
uv run python tools/generate_schemas.py
```

The schemas check structure, field types and syntax. A release row that passes its schema can
still be refused when the release is loaded. Loading also checks `SHA256SUMS`, each row's paths
and checksums, that `tracks` match `supported_metrics`, unique ids, the ledger and each answer
key.
