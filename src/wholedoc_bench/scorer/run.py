import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.aggregate import corpus_track_score, overall
from wholedoc_bench.scorer.base import Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.gates import GateResult, coverage
from wholedoc_bench.scorer.tracks import ALL_TRACKS

SPEC_VERSION = "1"


@dataclass(frozen=True)
class ParseFailure:
    reason: str


@dataclass(frozen=True)
class DocumentResult:
    document_id: str
    failed: bool
    error: str | None
    sub_scores: dict[str, dict[str, float | None]]
    track_scores: dict[str, float | None]
    details: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    statuses: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class CorpusResult:
    documents: list[DocumentResult]
    track_scores: dict[str, float | None]
    overall: float | None
    coverage: GateResult
    not_implemented: list[str] = field(default_factory=list)
    eligible_documents: dict[str, int] = field(default_factory=dict)


def score_document(
    truth: Groundtruth,
    pred: Prediction | ParseFailure | None,
    tracks: Sequence[Track] = ALL_TRACKS,
    not_implemented: set[str] | None = None,
    *,
    supported_metrics: frozenset[str] | None = None,
) -> DocumentResult:
    from wholedoc_bench.corpus.checks import check_font_annotations, check_style_annotations

    check_style_annotations(truth)
    check_font_annotations(truth)
    if supported_metrics is not None:
        known = {f"{t.id}.{m.name}" for t in tracks for m, _ in t.parts}
        if unknown := supported_metrics - known:
            raise ValueError(f"unknown supported metrics: {', '.join(sorted(unknown))}")
    error = _failure_reason(pred)
    ctx = DocContext(truth, pred) if error is None and isinstance(pred, Prediction) else None
    sub_scores: dict[str, dict[str, float | None]] = {}
    track_scores: dict[str, float | None] = {}
    details: dict[str, dict[str, dict[str, Any]]] = {}
    statuses: dict[str, dict[str, str]] = {}
    for track in tracks:
        result = track.evaluate(truth, ctx, supported_metrics)
        statuses[track.id] = result.statuses
        sub_scores[track.id] = result.sub_scores
        track_scores[track.id] = result.score
        if result.details:
            details[track.id] = result.details
        if not_implemented is not None:
            not_implemented.update(f"{track.id}.{name}" for name in result.not_implemented)
    return DocumentResult(
        document_id=truth.document.id,
        failed=error is not None,
        error=error,
        sub_scores=sub_scores,
        track_scores=track_scores,
        details=details,
        statuses=statuses,
    )


def score_corpus(
    items: Iterable[tuple[Groundtruth, Prediction | ParseFailure | None]],
    tracks: Sequence[Track] = ALL_TRACKS,
    *,
    supported_metrics: Mapping[str, frozenset[str]] | None = None,
) -> CorpusResult:
    not_implemented: set[str] = set()
    pairs = list(items)
    if supported_metrics is not None:
        ids = [truth.document.id for truth, _ in pairs]
        if len(ids) != len(set(ids)) or set(ids) != set(supported_metrics):
            raise ValueError("metric support must exactly match the unique document ids")
    documents = [
        score_document(
            t,
            p,
            tracks,
            not_implemented,
            supported_metrics=(
                supported_metrics[t.document.id] if supported_metrics is not None else None
            ),
        )
        for t, p in pairs
    ]
    return aggregate(documents, tracks, not_implemented)


def aggregate(
    documents: Sequence[DocumentResult],
    tracks: Sequence[Track] = ALL_TRACKS,
    not_implemented: Iterable[str] = (),
) -> CorpusResult:
    documents = list(documents)
    if not documents:
        raise ValueError("no documents to score")
    per_track = {
        track.id: corpus_track_score(d.track_scores[track.id] for d in documents)
        for track in tracks
    }
    ok = sum(not d.failed for d in documents)
    return CorpusResult(
        documents=documents,
        track_scores=per_track,
        overall=overall(per_track),
        coverage=coverage(ok, len(documents)),
        not_implemented=sorted(set(not_implemented)),
        eligible_documents={
            **{t.id: sum(d.track_scores[t.id] is not None for d in documents) for t in tracks},
            **{
                f"{t.id}.{m.name}": sum(d.sub_scores[t.id][m.name] is not None for d in documents)
                for t in tracks
                for m, _ in t.parts
            },
        },
    )


def write_scores_jsonl(
    result: CorpusResult,
    path: Path,
    tracks: Sequence[Track] = ALL_TRACKS,
    *,
    evaluation_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in result.documents:
            for track in tracks:
                for metric, weight in track.parts:
                    row = {
                        "document_id": doc.document_id,
                        "failed": doc.failed,
                        "error": doc.error,
                        "track": track.id,
                        "sub_metric": metric.name,
                        "source": metric.source.value,
                        "weight": weight,
                        "score": doc.sub_scores[track.id][metric.name],
                        "status": doc.statuses[track.id][metric.name],
                    }
                    if payload := doc.details.get(track.id, {}).get(metric.name):
                        row["details"] = payload
                    if evaluation_id is not None:
                        row["evaluation_id"] = evaluation_id
                    f.write(json.dumps(row) + "\n")


def _failure_reason(pred: Prediction | ParseFailure | None) -> str | None:
    if pred is None:
        return "no prediction"
    if isinstance(pred, ParseFailure):
        return pred.reason
    if not pred.markdown.strip() and not pred.layout:
        return "empty output"
    return None
