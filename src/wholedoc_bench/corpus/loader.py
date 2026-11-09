import json
from dataclasses import dataclass
from pathlib import Path

from wholedoc_bench.corpus.checks import CorpusError, check_groundtruth
from wholedoc_bench.corpus.release import is_release, read_release
from wholedoc_bench.schema import SCHEMA_VERSION, Groundtruth, Prediction


class SchemaVersionError(ValueError):
    pass


def load_groundtruth(path: Path) -> Groundtruth:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{path}: schema_version is {version!r}, this scorer reads {SCHEMA_VERSION!r}"
        )
    truth = Groundtruth.model_validate(data)
    try:
        check_groundtruth(truth)
    except CorpusError as e:
        raise CorpusError(f"{path}: {e}") from e
    return truth


def load_corpus(corpus_dir: Path) -> list[Groundtruth]:
    if (corpus_dir / "corpus.json").exists() or is_release(corpus_dir):
        raise ValueError(
            "working corpora and releases require load_corpus_context() and its "
            "metric_support when scoring"
        )
    return _load_legacy(corpus_dir)


def _load_release(
    corpus_dir: Path, *, include_drafts: bool, exclude: frozenset[str]
) -> "LoadedCorpus":
    if include_drafts:
        raise ValueError("a public release holds accepted documents only; drop --include-drafts")
    release = read_release(corpus_dir)
    ids = {document.row.id for document in release.documents}
    if unknown := exclude - ids:
        raise ValueError(f"unknown excluded document ids: {', '.join(sorted(unknown))}")
    selected: list[LoadedDocument] = []
    for document in sorted(release.documents, key=lambda d: d.row.id):
        row = document.row
        if row.id in exclude:
            continue
        identity = {
            "id": row.id,
            "status": "accepted",
            "document_sha256": row.document_sha256,
            "key_sha256": row.groundtruth_sha256,
            "supported_metrics": sorted(row.supported_metrics),
        }
        metrics = frozenset(row.supported_metrics)
        selected.append(LoadedDocument(document.truth, document.source, metrics, True, identity))
    if not selected:
        raise ValueError("no documents selected from the release")
    info = release.info
    identity = {
        "format_version": info.format_version,
        "name": info.corpus,
        "header_sha256": info.header_sha256,
        "manifest_sha256": info.manifest_sha256,
        "include_drafts": False,
        "excluded_ids": sorted(exclude),
        "documents": [d.identity for d in selected],
        "release": {
            "tag": info.tag,
            "source_commit": info.source_commit,
            "release_sha256": release.release_sha256,
            "sums_sha256": release.sums_sha256,
        },
    }
    return LoadedCorpus(tuple(selected), identity)


def _load_legacy(corpus_dir: Path) -> list[Groundtruth]:
    paths = sorted(corpus_dir.glob("*/groundtruth.json"))
    if not paths:
        raise FileNotFoundError(f"{corpus_dir}: no */groundtruth.json files")
    corpus = [load_groundtruth(p) for p in paths]
    seen: dict[str, Path] = {}
    for path, truth in zip(paths, corpus, strict=True):
        doc_id = truth.document.id
        if doc_id in seen:
            raise ValueError(
                f"document id {doc_id!r} is used twice: {seen[doc_id].parent} and {path.parent}"
            )
        seen[doc_id] = path
    return corpus


@dataclass(frozen=True)
class LoadedDocument:
    truth: Groundtruth
    source: Path | None
    supported_metrics: frozenset[str] | None
    cloud_allowed: bool
    identity: dict


@dataclass(frozen=True)
class LoadedCorpus:
    documents: tuple[LoadedDocument, ...]
    identity: dict | None = None

    @property
    def is_manifest(self) -> bool:
        return self.identity is not None

    @property
    def sources(self) -> dict[str, str]:
        return {d.truth.document.id: d.identity["document_sha256"] for d in self.documents}

    @property
    def metric_support(self) -> dict[str, frozenset[str]] | None:
        if not self.is_manifest:
            return None
        return {d.truth.document.id: d.supported_metrics or frozenset() for d in self.documents}


def load_corpus_context(
    corpus_dir: Path, *, include_drafts: bool = False, exclude: frozenset[str] = frozenset()
) -> LoadedCorpus:
    if is_release(corpus_dir):
        return _load_release(corpus_dir, include_drafts=include_drafts, exclude=exclude)
    if (corpus_dir / "corpus.json").exists():
        raise ValueError(f"{corpus_dir} is a working corpus, not a release: score a release folder")
    if include_drafts or exclude:
        raise ValueError("status and exclusion options require a release")
    documents = []
    for truth in _load_legacy(corpus_dir):
        found = sorted((corpus_dir / truth.document.id).glob("document.*"))
        documents.append(LoadedDocument(truth, found[0] if found else None, None, True, {}))
    return LoadedCorpus(tuple(documents))


def prediction_path(predictions_dir: Path, document_id: str) -> Path:
    return predictions_dir / f"{document_id}.json"


def load_prediction(path: Path) -> Prediction:
    return Prediction.model_validate_json(path.read_text(encoding="utf-8"))


def save_prediction(prediction: Prediction, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prediction.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8")
