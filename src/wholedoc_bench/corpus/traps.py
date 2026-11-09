import html
import unicodedata
from collections.abc import Iterable, Sequence
from urllib.parse import unquote

from wholedoc_bench.schema import Chart, Groundtruth, Planted
from wholedoc_bench.scorer.utils.prose import letter_digit_runs, phrase_count
from wholedoc_bench.scorer.utils.tracks.table_links import plain_number

HIDDEN_KINDS = frozenset({"hidden", "redacted"})


def hidden_traps(truth: Groundtruth) -> list[Planted]:
    return [p for p in truth.planted if p.kind in HIDDEN_KINDS]


def trap_phrase(trap: Planted) -> list[str]:
    return _folded_runs(trap.printed)


def found_traps(traps: Sequence[Planted], texts: Iterable[str]) -> list[Planted]:
    pieces = [_folded_runs(form) for text in texts for form in (text, _decoded(text))]
    return [trap for trap in traps if phrase_count(trap_phrase(trap), pieces)]


def chart_texts(chart: Chart | None) -> list[str]:
    if chart is None:
        return []
    names = [series.name for series in chart.series]
    labels = dict.fromkeys(point.label for series in chart.series for point in series.points)
    rows = [
        [label]
        + [
            plain_number(point.value)
            for series in chart.series
            for point in series.points
            if point.label == label
        ]
        for label in labels
    ]
    return [chart.title or "", *names, *(text for row in rows for text in row)]


def _decoded(text: str) -> str:
    return unquote(html.unescape(text))


def _folded_runs(text: str) -> list[str]:
    return letter_digit_runs(_without_accents(text.casefold()))


def _without_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))
