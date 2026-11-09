import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property, partial

from wholedoc_bench.schema import (
    ChartSeries,
    Element,
    ElementClass,
    Groundtruth,
    LayoutItem,
    ValueAxis,
)
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.blocks import (
    Cells,
    Grid,
    MarkdownTable,
    markdown_text,
    normalize_text,
    without_quote_markers,
)
from wholedoc_bench.scorer.utils.boxes import MIN_IOU, box_iou
from wholedoc_bench.scorer.utils.pairing import reaches
from wholedoc_bench.scorer.utils.tracks.table_links import (
    AnchorKey,
    AnchorSource,
    TableLink,
    cell_tokens,
    chart_anchor_text,
    linked_tables,
    text_tokens,
    token_similarity,
)

UNICODE_MINUS = "−"
MINUS = "-"
PLUS = "+"
NEGATIVE_OPEN = "("
NEGATIVE_CLOSE = ")"
PERCENT_SIGN = "%"
APPROXIMATION_MARKS = ("~", "≈")
CURRENCY_MARKS = ("USD", "EUR", "GBP", "$", "€", "£", "¥")
SCALE_FACTORS = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "mn": 1e6,
    "bn": 1e9,
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
}
THOUSANDS_SEPARATOR = ","
DECIMAL_POINT = "."
_THOUSANDS_GROUPED = re.compile(r"-?\d+(?:,\d{3})+(?:\.\d+)?")
_DECIMAL_COMMA = re.compile(r"-?\d+,(?!\d{3}$)\d+")
_TRAILING_GROUP = re.compile(r"\s*\([^()]*\)$")
VALUE_TOLERANCE = 0.01
AXIS_SPAN_TOLERANCE = 0.005
IOU_DECIMALS = 9


def chart_key(text: str) -> str:
    return normalize_text(text, fold_case=True)


def series_matches(predicted_name: str, reference_name: str) -> bool:
    predicted = chart_key(predicted_name)
    return chart_key(reference_name) in (predicted, _TRAILING_GROUP.sub("", predicted))


def compares_series(truth_chart: Element | None) -> bool:
    return truth_chart is None or truth_chart.chart is None or len(truth_chart.chart.series) != 1


def value_tolerance(reference: float, axis: ValueAxis | None) -> float:
    axis_part = 0.0 if axis is None else AXIS_SPAN_TOLERANCE * (axis.max - axis.min)
    return max(VALUE_TOLERANCE * abs(reference), axis_part)


def within_tolerance(reference: float, value: float, axis: ValueAxis | None) -> bool:
    difference = abs(value - reference)
    limit = value_tolerance(reference, axis)
    return difference <= limit or math.isclose(difference, limit)


def _without_whitespace(text: str) -> str:
    return "".join(text.split())


def _without_approximation_mark(text: str) -> str:
    for mark in APPROXIMATION_MARKS:
        if text.startswith(mark):
            return text.removeprefix(mark)
    return text


def _sign_and_magnitude(text: str) -> tuple[int, str]:
    text = text.replace(UNICODE_MINUS, MINUS)
    if text.startswith(NEGATIVE_OPEN) and text.endswith(NEGATIVE_CLOSE):
        return -1, text[1:-1]
    if text.startswith(MINUS):
        return -1, text.removeprefix(MINUS)
    return 1, text


def _without_currency(text: str) -> str:
    for mark in CURRENCY_MARKS:
        if text.startswith(mark):
            return text.removeprefix(mark)
    for mark in CURRENCY_MARKS:
        if text.endswith(mark):
            return text.removesuffix(mark)
    return text


def _scale_split(text: str) -> tuple[str, float | None]:
    lowered = text.casefold()
    for suffix, factor in SCALE_FACTORS.items():
        if lowered.endswith(suffix):
            return text[: -len(suffix)], factor
    return text, None


def _separators_resolved(text: str) -> str:
    if _THOUSANDS_GROUPED.fullmatch(text):
        return text.replace(THOUSANDS_SEPARATOR, "")
    if _DECIMAL_COMMA.fullmatch(text):
        return text.replace(THOUSANDS_SEPARATOR, DECIMAL_POINT)
    return text


def _finite_number(text: str) -> float | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def cell_values(text: str) -> tuple[float, ...]:
    exact = _without_approximation_mark(_without_whitespace(text))
    sign, magnitude = _sign_and_magnitude(exact)
    unitless = _without_currency(magnitude).removesuffix(PERCENT_SIGN)
    if sign < 0 and unitless.startswith((MINUS, PLUS)):
        return ()
    digits, factor = _scale_split(unitless)
    number = _finite_number(_separators_resolved(digits))
    if number is None:
        return ()
    written = sign * number
    return (written,) if factor is None else (written, written * factor)


class LabelSide(StrEnum):
    ROWS = "rows"
    COLUMNS = "columns"


@dataclass(frozen=True)
class PredictedPoint:
    series: str
    label: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class OrientedTable:
    table: MarkdownTable
    row_headers: Cells
    column_headers: Cells
    value_rows: Grid
    label_side: LabelSide

    @property
    def labels(self) -> Cells:
        return self.row_headers if self.label_side is LabelSide.ROWS else self.column_headers

    @property
    def series_names(self) -> Cells:
        return self.column_headers if self.label_side is LabelSide.ROWS else self.row_headers

    @cached_property
    def points(self) -> tuple[PredictedPoint, ...]:
        return tuple(
            PredictedPoint(*self._series_and_label(r, c), values)
            for r, row in enumerate(self.value_rows)
            for c, cell in enumerate(row)
            if (values := cell_values(cell))
        )

    def _series_and_label(self, row: int, column: int) -> tuple[str, str]:
        label_index, series_index = (
            (row, column) if self.label_side is LabelSide.ROWS else (column, row)
        )
        return (
            _cell_or_empty(self.series_names, series_index),
            _cell_or_empty(self.labels, label_index),
        )


def _cell_or_empty(cells: Cells, index: int) -> str:
    return cells[index] if index < len(cells) else ""


def _header_and_body(table: MarkdownTable) -> tuple[Cells, Grid]:
    if table.header:
        return table.header[-1], table.body
    return table.body[0], table.body[1:]


def _truth_label_keys(chart: Element | None) -> frozenset[str]:
    if chart is None or chart.chart is None:
        return frozenset()
    return frozenset(chart_key(p.label) for s in chart.chart.series for p in s.points)


def _shared_label_count(headers: Iterable[str], labels: frozenset[str]) -> int:
    return sum(chart_key(header) in labels for header in headers)


def orient_table(table: MarkdownTable, truth_chart: Element | None) -> OrientedTable:
    header, body = _header_and_body(table)
    row_headers = tuple(_cell_or_empty(row, 0) for row in body)
    column_headers = header[1:]
    labels = _truth_label_keys(truth_chart)
    column_side_wins = _shared_label_count(column_headers, labels) > _shared_label_count(
        row_headers, labels
    )
    return OrientedTable(
        table=table,
        row_headers=row_headers,
        column_headers=column_headers,
        value_rows=tuple(row[1:] for row in body),
        label_side=LabelSide.COLUMNS if column_side_wins else LabelSide.ROWS,
    )


def series_points(series: Iterable[ChartSeries]) -> tuple[PredictedPoint, ...]:
    return tuple(PredictedPoint(s.name, p.label, (p.value,)) for s in series for p in s.points)


def _same_point(
    point: PredictedPoint,
    reference: PredictedPoint,
    compare_series: bool,
    values_match: Callable[[float, float], bool],
) -> bool:
    return (
        chart_key(point.label) == chart_key(reference.label)
        and (not compare_series or series_matches(point.series, reference.series))
        and any(
            values_match(expected, value) for expected in reference.values for value in point.values
        )
    )


def point_matches(
    point: PredictedPoint,
    reference: PredictedPoint,
    axis: ValueAxis | None,
    compare_series: bool,
) -> bool:
    return _same_point(point, reference, compare_series, partial(within_tolerance, axis=axis))


def maximum_matching[C, R](
    references: Sequence[R], candidates: Sequence[C], matches: Callable[[C, R], bool]
) -> dict[int, int]:
    options = [
        [index for index, candidate in enumerate(candidates) if matches(candidate, reference)]
        for reference in references
    ]
    owners: dict[int, int] = {}

    def assign(reference: int, visited: set[int]) -> bool:
        for candidate in options[reference]:
            if candidate in visited:
                continue
            visited.add(candidate)
            if candidate not in owners or assign(owners[candidate], visited):
                owners[candidate] = reference
                return True
        return False

    for reference in range(len(references)):
        assign(reference, set())
    return owners


def _unrepeated_table_points(
    tables: Iterable[OrientedTable],
    json_points: Sequence[PredictedPoint],
    truth: Element | None,
) -> tuple[PredictedPoint, ...]:
    compare_series = compares_series(truth)

    def repeats(json_point: PredictedPoint, point: PredictedPoint) -> bool:
        return _same_point(point, json_point, compare_series, math.isclose)

    table_points = [point for table in tables for point in table.points]
    absorbed = set(maximum_matching(table_points, json_points, repeats).values())
    return tuple(point for index, point in enumerate(table_points) if index not in absorbed)


@dataclass(frozen=True)
class ChartGroup:
    item: LayoutItem | None
    truth: Element | None
    json_series: tuple[ChartSeries, ...]
    tables: tuple[OrientedTable, ...]
    points: tuple[PredictedPoint, ...]


def _chart_group(
    item: LayoutItem | None,
    truth: Element | None,
    json_series: tuple[ChartSeries, ...],
    tables: tuple[MarkdownTable, ...],
) -> ChartGroup:
    oriented = tuple(orient_table(table, truth) for table in tables)
    json_points = series_points(json_series)
    points = (*json_points, *_unrepeated_table_points(oriented, json_points, truth))
    return ChartGroup(item, truth, json_series, oriented, points)


@dataclass(frozen=True)
class _GroupOwner:
    item: LayoutItem | None
    truth: Element | None

    @property
    def anchor_keys(self) -> frozenset[AnchorKey]:
        keys: list[AnchorKey] = []
        if self.item is not None:
            keys.append((AnchorSource.LAYOUT, self.item.id))
        if self.truth is not None:
            keys.append((AnchorSource.GROUNDTRUTH, self.truth.id))
        return frozenset(keys)

    @property
    def json_series(self) -> tuple[ChartSeries, ...]:
        if self.item is None or self.item.chart is None:
            return ()
        return tuple(self.item.chart.series)


def truth_charts(truth: Groundtruth) -> tuple[Element, ...]:
    return tuple(e for e in truth.elements if e.chart is not None)


def _chart_iou(chart: Element, item: LayoutItem) -> float | None:
    if chart.page != item.page or chart.box is None or item.box is None:
        return None
    return round(box_iou(chart.box, item.box), IOU_DECIMALS)


def _best_pairs(candidates: Iterable[tuple[float, int, int]]) -> dict[int, int]:
    pairs: dict[int, int] = {}
    for _, chart_index, item_index in sorted(candidates):
        if item_index not in pairs and chart_index not in pairs.values():
            pairs[item_index] = chart_index
    return pairs


def _box_pairs(charts: Sequence[Element], items: Sequence[LayoutItem]) -> dict[int, int]:
    return _best_pairs(
        (-iou, chart_index, item_index)
        for chart_index, chart in enumerate(charts)
        for item_index, item in enumerate(items)
        if (iou := _chart_iou(chart, item)) is not None and reaches(iou, MIN_IOU)
    )


def _item_tokens(item: LayoutItem, links: Iterable[TableLink]) -> Counter[str]:
    key = (AnchorSource.LAYOUT, item.id)
    rows = (row for link in links if link.anchor_key == key for row in link.table.rows)
    return text_tokens(chart_anchor_text(item)) + cell_tokens(rows)


def _content_pairs(
    charts: Sequence[Element],
    items: Sequence[LayoutItem],
    links: Sequence[TableLink],
    box_pairs: dict[int, int],
) -> dict[int, int]:
    chart_tokens = {
        index: text_tokens(chart_anchor_text(chart))
        for index, chart in enumerate(charts)
        if index not in box_pairs.values()
    }
    item_tokens = {
        index: _item_tokens(item, links)
        for index, item in enumerate(items)
        if index not in box_pairs
    }
    return _best_pairs(
        (-similarity, chart_index, item_index)
        for item_index, tokens in item_tokens.items()
        for chart_index, truth_tokens in chart_tokens.items()
        if items[item_index].page == charts[chart_index].page
        and (similarity := float(token_similarity(tokens, truth_tokens))) > 0
    )


def _group_owners(
    truth: Groundtruth, layout: Sequence[LayoutItem], links: Sequence[TableLink]
) -> list[_GroupOwner]:
    charts = truth_charts(truth)
    items = [item for item in layout if item.cls is ElementClass.CHART]
    pairs = _box_pairs(charts, items)
    pairs |= _content_pairs(charts, items, links, pairs)
    owners = [
        _GroupOwner(item, charts[pairs[index]] if index in pairs else None)
        for index, item in enumerate(items)
    ]
    unpaired = (chart for index, chart in enumerate(charts) if index not in pairs.values())
    return [*owners, *(_GroupOwner(None, chart) for chart in unpaired)]


def chart_groups(ctx: DocContext) -> tuple[ChartGroup, ...]:
    chart_links = [link for link in linked_tables(ctx) if link.kind is ElementClass.CHART]
    claimed: set[AnchorKey] = set()
    groups: list[ChartGroup] = []
    for owner in _group_owners(ctx.truth, ctx.pred.layout, chart_links):
        keys = owner.anchor_keys - claimed
        claimed |= keys
        tables = tuple(link.table for link in chart_links if link.anchor_key in keys)
        groups.append(_chart_group(owner.item, owner.truth, owner.json_series, tables))
    orphans = (link.table for link in chart_links if link.anchor_key not in claimed)
    groups.extend(_chart_group(None, None, (), (table,)) for table in orphans)
    return tuple(groups)


Span = tuple[int, int]

_NON_BLANK_LINE = re.compile(r"^[^\n]*\S[^\n]*$", re.MULTILINE)
_TRAILING_STOPS = (":", ".")


def _without_trailing_stop(text: str) -> str:
    return text[:-1] if text.endswith(_TRAILING_STOPS) else text


_TITLE_STRIPPING_STEPS = (_without_trailing_stop,)


def _title_forms(text: str) -> Iterator[str]:
    form = normalize_text(text)
    yield form
    for step in _TITLE_STRIPPING_STEPS:
        form = step(form)
        yield form


def title_keys(text: str) -> frozenset[str]:
    return frozenset(form.casefold() for form in _title_forms(text))


def _non_blank_lines(markdown: str) -> tuple[Span, ...]:
    return tuple(line.span() for line in _NON_BLANK_LINE.finditer(markdown))


def _overlaps_any_table(span: Span, tables: Iterable[MarkdownTable]) -> bool:
    start, end = span
    return any(start < table.end and table.start < end for table in tables)


def _first_text(
    markdown: str, segments: Iterable[Span], tables: Sequence[MarkdownTable]
) -> str | None:
    for segment in segments:
        if _overlaps_any_table(segment, tables):
            return None
        text = markdown[segment[0] : segment[1]].strip()
        if without_quote_markers(text).strip():
            return text
    return None


def _segments_before(lines: Sequence[Span], table: MarkdownTable) -> Iterator[Span]:
    for start, end in reversed(lines):
        if start < table.start:
            yield start, min(end, table.start)


def _segments_after(lines: Sequence[Span], table: MarkdownTable) -> Iterator[Span]:
    for start, end in lines:
        if end > table.end:
            yield max(start, table.end), end


def touching_lines(
    markdown: str, table: MarkdownTable, markdown_tables: Sequence[MarkdownTable]
) -> tuple[str, ...]:
    lines = _non_blank_lines(markdown)
    found = (
        _first_text(markdown, _segments_before(lines, table), markdown_tables),
        _first_text(markdown, _segments_after(lines, table), markdown_tables),
    )
    return tuple(markdown_text(text) for text in found if text is not None)
