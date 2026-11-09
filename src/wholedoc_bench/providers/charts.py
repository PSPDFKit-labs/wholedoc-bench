import re
from collections.abc import Iterable

from wholedoc_bench.schema import ChartKind

_SUMMARY_KINDS = {
    "bar": ChartKind.BAR,
    "column": ChartKind.BAR,
    "line": ChartKind.LINE,
    "area": ChartKind.AREA,
    "pie": ChartKind.PIE,
    "scatter": ChartKind.SCATTER,
    "bubble": ChartKind.BUBBLE,
    "histogram": ChartKind.HISTOGRAM,
    "box": ChartKind.BOX,
    "box and whisker": ChartKind.BOX,
    "violin": ChartKind.VIOLIN,
    "heatmap": ChartKind.HEATMAP,
    "heat map": ChartKind.HEATMAP,
    "treemap": ChartKind.TREEMAP,
    "funnel": ChartKind.FUNNEL,
    "radar": ChartKind.RADAR,
    "spider": ChartKind.RADAR,
    "candlestick": ChartKind.CANDLESTICK,
    "step": ChartKind.STEP,
    "step line": ChartKind.STEP,
    "stem": ChartKind.STEM,
    "combo": ChartKind.COMBO,
    "combination": ChartKind.COMBO,
    "density": ChartKind.DENSITY,
    "error bar": ChartKind.ERRORBAR,
    "errorbar": ChartKind.ERRORBAR,
    "bullet": ChartKind.BULLET,
    "marimekko": ChartKind.MARIMEKKO,
    "mosaic": ChartKind.MARIMEKKO,
    "pyramid": ChartKind.PYRAMID,
    "population pyramid": ChartKind.PYRAMID,
    "span": ChartKind.SPAN,
    "range": ChartKind.SPAN,
    "radial bar": ChartKind.RADIAL_BAR,
    "radial column": ChartKind.RADIAL_BAR,
    "parallel coordinates": ChartKind.PARALLEL,
    "stream": ChartKind.STREAM,
    "donut": ChartKind.DONUT,
    "doughnut": ChartKind.DONUT,
    "rose": ChartKind.ROSE,
    "nightingale rose": ChartKind.ROSE,
    "polar area": ChartKind.ROSE,
    "sunburst": ChartKind.SUNBURST,
    "circle packing": ChartKind.CIRCLE_PACKING,
    "circular packing": ChartKind.CIRCLE_PACKING,
    "proportional area": ChartKind.PROPORTIONAL_AREA,
    "dot matrix": ChartKind.DOT_MATRIX,
    "gantt": ChartKind.GANTT,
    "waterfall": ChartKind.WATERFALL,
}
_SUMMARY_WORDS = {
    "histogram": ChartKind.HISTOGRAM,
    "heatmap": ChartKind.HEATMAP,
    "treemap": ChartKind.TREEMAP,
    "streamgraph": ChartKind.STREAM,
    "sunburst": ChartKind.SUNBURST,
}


def _longest_first(names: Iterable[str]) -> str:
    return "|".join(sorted(map(re.escape, names), key=len, reverse=True))


_SUMMARY_KIND = re.compile(
    rf"\b(?:({_longest_first(_SUMMARY_KINDS)}) (?:chart|plot|graph|diagram)"
    rf"|({_longest_first(_SUMMARY_WORDS)}))\b",
    re.IGNORECASE,
)


def named_chart_kind(description: str) -> ChartKind | None:
    found = _SUMMARY_KIND.search(description)
    if found is None:
        return None
    named, word = found.groups()
    return _SUMMARY_KINDS[named.lower()] if named else _SUMMARY_WORDS[word.lower()]
