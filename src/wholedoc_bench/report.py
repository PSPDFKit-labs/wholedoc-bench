import json
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from wholedoc_bench.prices import STALE_AFTER, Cost, PriceBook, is_stale, price
from wholedoc_bench.provenance import comparison_key, validate_comparison
from wholedoc_bench.providers import PROVIDERS
from wholedoc_bench.scorer.run import SPEC_VERSION, CorpusResult
from wholedoc_bench.scorer.tracks import ALL_TRACKS


@dataclass(frozen=True)
class Gate:
    value: float
    passed: bool


@dataclass(frozen=True)
class Summary:
    parser: str
    parser_version: str
    spec_version: str
    documents: int
    parsed: int
    track_scores: dict[str, float | None]
    overall: float | None
    coverage: Gate
    provider: str | None = None
    provider_config: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    eligible_documents: dict[str, int] = field(default_factory=dict)
    failure_stages: dict[str, int] = field(default_factory=dict)
    origin: Literal["run", "import"] = "run"
    usage: dict[str, Any] | None = None


ORIGINS = ("run", "import")


@dataclass(frozen=True)
class Prices:
    book: PriceBook
    as_of: date


def count_failures(attempts: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        if attempt.get("status") == "failed":
            stage = attempt.get("stage") or "unclassified"
            counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def check_rankable(summaries: list[Summary]) -> None:
    blocked = []
    for summary in summaries:
        if summary.evaluation is None:
            continue
        stages = count_failures(summary.evaluation.get("attempts", {}).values())
        found = {s: n for s, n in stages.items() if s in ("normalize", "unclassified")}
        if found:
            blocked.append(f"{summary.parser}: " + ", ".join(f"{n} {s}" for s, n in found.items()))
    if blocked:
        raise ValueError(
            "results with adapter or unclassified failures can't be ranked; fix the adapter and "
            "replay with `run --from-raw`, or run again:\n  " + "\n  ".join(blocked)
        )


def summarize(
    parser: str,
    result: CorpusResult,
    parser_version: str,
    *,
    provider: str | None = None,
    provider_config: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    failure_stages: dict[str, int] | None = None,
    origin: Literal["run", "import"] = "run",
    usage: dict[str, Any] | None = None,
) -> Summary:
    return Summary(
        parser=parser,
        provider=provider,
        provider_config=provider_config,
        comparison=comparison,
        evaluation=evaluation,
        eligible_documents=dict(result.eligible_documents),
        failure_stages=failure_stages or {},
        parser_version=parser_version,
        spec_version=SPEC_VERSION,
        documents=len(result.documents),
        parsed=sum(not d.failed for d in result.documents),
        track_scores=dict(result.track_scores),
        overall=result.overall,
        coverage=Gate(result.coverage.value, result.coverage.passed),
        origin=origin,
        usage=usage,
    )


def write_summary(summary: Summary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(summary)
    if data["origin"] == "run":
        del data["origin"]
    if data["usage"] is None:
        del data["usage"]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_summary(path: Path) -> Summary:
    return _summary(_summary_data(path), path)


def read_summaries(results_dir: Path) -> list[Summary]:
    summaries: list[Summary] = []
    stale: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        data = _summary_data(path)
        if data["spec_version"] != SPEC_VERSION:
            stale.append(f"{data.get('parser') or path.stem}: spec {data['spec_version']}")
        else:
            summaries.append(_summary(data, path))
    _refuse_stale(stale)
    return summaries


def check_comparable(summaries: list[Summary]) -> None:
    _refuse_stale(
        [f"{s.parser}: spec {s.spec_version}" for s in summaries if s.spec_version != SPEC_VERSION]
    )
    identities = []
    for summary in summaries:
        if summary.comparison is not None:
            validate_comparison(summary.comparison, summary.documents, summary.spec_version)
            identities.append(comparison_key(summary.comparison))
    if identities and (len(identities) != len(summaries) or len(set(identities)) != 1):
        raise ValueError(
            "results have different or missing comparison identities; "
            "re-score one scope with one scorer"
        )
    if summaries and not identities:
        warnings.warn(
            "legacy summaries have no exact comparison identity; "
            "document count alone does not prove comparability",
            stacklevel=2,
        )
    if len({s.documents for s in summaries}) > 1:
        raise ValueError(
            "results scored on corpora of different sizes; score every parser on one corpus:\n  "
            + "\n  ".join(
                f"{s.parser}: {s.documents} document{'' if s.documents == 1 else 's'}"
                for s in summaries
            )
        )


def check_origins(summaries: list[Summary], *, allow_mixed: bool = False) -> None:
    if len({s.origin for s in summaries}) > 1 and not allow_mixed:
        raise ValueError(
            "results mix wholedoc-bench runs with imported predictions; rank them separately, "
            "or pass --allow-mixed-origins to show them together with labels:\n  "
            + "\n  ".join(f"{s.parser}: {s.origin}" for s in summaries)
        )


def parser_label(summary: Summary) -> str:
    return f"{summary.parser} (imported)" if summary.origin == "import" else summary.parser


def _refuse_stale(stale: list[str]) -> None:
    if stale:
        raise ValueError(
            "results scored under another spec version; re-score these parsers:\n  "
            + "\n  ".join(stale)
            + f"\n  (this scorer: spec {SPEC_VERSION})"
        )


def _summary_data(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"{path.name}: not a parser summary ({e})") from e
    if not isinstance(data, dict) or not isinstance(data.get("spec_version"), str):
        raise ValueError(f"{path.name}: not a parser summary (no spec_version)")
    return data


def _summary(data: dict[str, Any], path: Path) -> Summary:
    try:
        if data.get("comparison") is not None:
            validate_comparison(data["comparison"], data["documents"], data["spec_version"])
        if data.get("origin", "run") not in ORIGINS:
            raise ValueError(f"unknown origin {data['origin']!r}")
        if data.get("usage") is not None:
            _check_usage(data["usage"])
        return Summary(**{**data, "coverage": Gate(**data["coverage"])})
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path.name}: not a parser summary of spec {SPEC_VERSION} ({e})") from e


def _check_usage(usage: Any) -> None:
    counts = isinstance(usage, dict) and set(usage) == {"documents", "pages", "meters"}
    if not counts or not all(
        isinstance(usage[k], int) and not isinstance(usage[k], bool) for k in ("documents", "pages")
    ):
        raise ValueError("usage needs documents, pages and meters")
    meters = usage["meters"]
    if not isinstance(meters, dict) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in meters.values()
    ):
        raise ValueError("usage meters must be numbers")


def ranked(summaries: list[Summary]) -> list[Summary]:
    return sorted(summaries, key=lambda s: (not passes_gates(s), -(s.overall or 0.0), s.parser))


def passes_gates(summary: Summary) -> bool:
    return summary.coverage.passed


def coverage_cell(summary: Summary) -> str:
    verdict = "pass" if summary.coverage.passed else "FAIL"
    return f"{summary.parsed}/{summary.documents} ({summary.coverage.value:.0%}, {verdict})"


def render_text(summaries: list[Summary], prices: Prices | None = None) -> str:
    header, rows, notes = _table(summaries, prices)
    widths = [max(len(line[i]) for line in [header, *rows]) for i in range(len(header))]
    coverage = len(ALL_TRACKS) + 2

    def line(cells: list[str]) -> str:
        first, *rest = cells
        aligned = [first.ljust(widths[0])]
        for i, cell in enumerate(rest, start=1):
            aligned.append(cell.ljust(widths[i]) if i == coverage else cell.rjust(widths[i]))
        return "  ".join(aligned).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(header), rule, *map(line, rows), "", _track_names(), *notes])


def render_markdown(summaries: list[Summary], prices: Prices | None = None) -> str:
    header, rows, notes = _table(summaries, prices)
    first = summaries[0]
    lines = [
        "# Leaderboard",
        "",
        (
            f"Spec {first.spec_version}, {first.documents} documents. "
            "Built by `wholedoc-bench report`."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "|---|"
        + "---:|" * (len(ALL_TRACKS) + 1)
        + "---|"
        + "---:|" * (len(header) - len(ALL_TRACKS) - 3),
        *("| " + " | ".join(row) + " |" for row in rows),
        "",
        _track_names(),
    ]
    for note in notes:
        lines += ["", note]
    return "\n".join(lines) + "\n"


COST_HEADER = "$/1k pages"


def _table(
    summaries: list[Summary], prices: Prices | None
) -> tuple[list[str], list[list[str]], list[str]]:
    track_ids = [t.id for t in ALL_TRACKS]
    order = ranked(summaries)
    costs = None
    if prices is not None and any(s.usage is not None for s in summaries):
        costs = [_cost(s, prices) for s in order]
    header = ["Parser", *track_ids, "Overall", "Coverage (G3)", *([COST_HEADER] if costs else [])]
    rows = [
        [
            parser_label(s),
            *(_score(s.track_scores.get(t)) for t in track_ids),
            _score(s.overall),
            coverage_cell(s),
            *([costs[i][0]] if costs else []),
        ]
        for i, s in enumerate(order)
    ]
    notes = [_IMPORTED_NOTE] if any(s.origin == "import" for s in summaries) else []
    if costs and prices is not None:
        notes += _cost_notes(order, [cost for _, cost in costs], prices)
    return header, rows, notes


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _cost(summary: Summary, prices: Prices) -> tuple[str, Cost | None]:
    usage = summary.usage
    if usage is None:
        provider = PROVIDERS.get(summary.provider or "")
        local = provider is not None and not provider.sends_documents
        return ("local" if local else "n/a"), None
    if not usage["pages"] or summary.provider is None:
        return "n/a", None
    config = summary.provider_config or {}
    cost = price(prices.book, summary.provider, config, usage["meters"], prices.as_of)
    if cost.usd is None:
        return "no price", cost
    per_1k = cost.usd / usage["pages"] * 1000
    return f"{'~' if cost.estimated else ''}{per_1k:.2f}", cost


def _cost_notes(order: list[Summary], costs: list[Cost | None], prices: Prices) -> list[str]:
    as_of, book = prices.as_of.isoformat(), prices.book.sha256[:12]
    notes = [
        f"{COST_HEADER}: US dollars per 1,000 pages at the list prices in force on {as_of} "
        f"(price book {book}), over the pages of the documents whose saved responses report "
        "their usage. It is not part of the ranking."
    ]
    if any(c is not None and c.estimated for c in costs):
        notes.append(
            "~ estimated: a rate the cost uses is marked as an estimate in the price book, as "
            "the responses cannot show what the provider billed."
        )
    partial = [
        f"{parser_label(s)} ({s.usage['documents']}/{s.documents})"
        for s in order
        if s.usage is not None and s.usage["documents"] < s.documents
    ]
    if partial:
        notes.append("Usage from fewer documents than scored: " + ", ".join(partial) + ".")
    missing = [
        f"{parser_label(s)} ({', '.join(c.missing)})"
        for s, c in zip(order, costs, strict=True)
        if c is not None and c.missing
    ]
    if missing:
        notes.append("No price in the book for: " + ", ".join(missing) + ".")
    stale = sorted(
        {
            f"{s.provider} {r.meter} (checked {r.checked.isoformat()})"
            for s, c in zip(order, costs, strict=True)
            if c is not None
            for r in c.rates
            if is_stale(r, prices.as_of)
        }
    )
    if stale:
        notes.append(
            f"Prices not checked in the {STALE_AFTER.days} days before {as_of}: "
            + ", ".join(stale)
            + "."
        )
    return notes


_IMPORTED_NOTE = (
    "(imported): predictions supplied from outside wholedoc-bench and imported with "
    "`wholedoc-bench import`. The scores are computed by this scorer; how the predictions were "
    "made is claimed, not verified."
)


def _track_names() -> str:
    return "Tracks: " + "; ".join(f"{t.id} {t.name}" for t in ALL_TRACKS) + "."
