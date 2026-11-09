from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from wholedoc_bench.schema import Groundtruth
from wholedoc_bench.scorer.context import DocContext


class ScoringError(RuntimeError): ...


class Source(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"
    EITHER = "either"
    BOTH = "both"


class SubMetric(ABC):
    name: ClassVar[str]
    source: ClassVar[Source]
    implemented: ClassVar[bool] = True

    def applies(self, truth: Groundtruth) -> bool:
        return True

    @abstractmethod
    def score(self, ctx: DocContext) -> float: ...

    def details(self, truth: Groundtruth, ctx: DocContext | None, status: str) -> dict[str, Any]:
        return {}


@dataclass(frozen=True)
class TrackResult:
    sub_scores: dict[str, float | None]
    score: float | None
    not_implemented: tuple[str, ...] = ()
    details: dict[str, dict[str, Any]] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Track:
    id: str
    name: str
    parts: tuple[tuple[SubMetric, float], ...]

    def __post_init__(self) -> None:
        names = [m.name for m, _ in self.parts]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.id}: sub-metric names must be unique, got {names}")
        if any(w <= 0 for _, w in self.parts):
            raise ValueError(f"{self.id}: weights must be positive")

    def evaluate(
        self,
        truth: Groundtruth,
        ctx: DocContext | None,
        supported_metrics: frozenset[str] | None = None,
    ) -> TrackResult:
        sub_scores: dict[str, float | None] = {}
        not_implemented: list[str] = []
        details: dict[str, dict[str, Any]] = {}
        statuses: dict[str, str] = {}
        for metric, _ in self.parts:
            if (
                supported_metrics is not None
                and f"{self.id}.{metric.name}" not in supported_metrics
            ):
                sub_scores[metric.name] = None
                statuses[metric.name] = "unsupported"
                continue
            applicable = (
                metric.applies(truth)
                if metric.implemented or supported_metrics is not None
                else False
            )
            if supported_metrics is not None and not applicable:
                raise ValueError(f"{self.id}.{metric.name}: supported metric has nothing to score")
            if not metric.implemented:
                sub_scores[metric.name] = None
                not_implemented.append(metric.name)
                statuses[metric.name] = "not_implemented"
            elif not applicable:
                sub_scores[metric.name] = None
            elif ctx is None:
                sub_scores[metric.name] = 0.0
            else:
                sub_scores[metric.name] = self._run(metric, ctx)
            if metric.implemented:
                status = (
                    "not_applicable"
                    if sub_scores[metric.name] is None
                    else "parse_failed"
                    if ctx is None
                    else "scored"
                )
                statuses[metric.name] = status
                try:
                    payload = metric.details(truth, ctx, status)
                except Exception as error:
                    raise ScoringError(f"{self.id}.{metric.name} details: {error}") from error
                if payload:
                    details[metric.name] = payload
        return TrackResult(
            sub_scores, self.combine(sub_scores), tuple(not_implemented), details, statuses
        )

    def _run(self, metric: SubMetric, ctx: DocContext) -> float:
        where = f"{self.id}.{metric.name} on {ctx.truth.document.id}"
        try:
            value = metric.score(ctx)
        except Exception as e:
            raise ScoringError(f"{where}: {type(e).__name__}: {e}") from e
        if not 0.0 <= value <= 1.0:
            raise ScoringError(f"{where}: score {value} is outside 0 to 1")
        return value

    def combine(self, sub_scores: Mapping[str, float | None]) -> float | None:
        scored = [(s, w) for m, w in self.parts if (s := sub_scores.get(m.name)) is not None]
        if not scored:
            return None
        return sum(s * w for s, w in scored) / sum(w for _, w in scored)
