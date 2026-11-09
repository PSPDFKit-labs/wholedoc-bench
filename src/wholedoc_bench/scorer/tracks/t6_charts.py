from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial

from wholedoc_bench.schema import ChartKind, Element, Groundtruth, LayoutItem
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.blocks import MarkdownTable
from wholedoc_bench.scorer.utils.tracks.t6_charts.charts import (
    ChartGroup,
    PredictedPoint,
    chart_groups,
    chart_key,
    compares_series,
    maximum_matching,
    point_matches,
    series_matches,
    series_points,
    title_keys,
    touching_lines,
    truth_charts,
)
from wholedoc_bench.scorer.utils.tracks.table_links import linked_tables


def _matched_count[C, R](
    references: Sequence[R], candidates: Sequence[C], matches: Callable[[C, R], bool]
) -> int:
    return len(maximum_matching(references, candidates, matches))


def _truth_points(chart: Element) -> tuple[PredictedPoint, ...]:
    return () if chart.chart is None else series_points(chart.chart.series)


def _group_hits(group: ChartGroup) -> int:
    if group.truth is None or group.truth.chart is None:
        return 0
    matches = partial(
        point_matches,
        axis=group.truth.chart.value_axis,
        compare_series=compares_series(group.truth),
    )
    return _matched_count(_truth_points(group.truth), group.points, matches)


class DataPoints(SubMetric):
    name = "data_points"
    source = Source.EITHER

    def applies(self, truth: Groundtruth) -> bool:
        return any(_truth_points(chart) for chart in truth_charts(truth))

    def score(self, ctx: DocContext) -> float:
        truth_count = sum(len(_truth_points(chart)) for chart in truth_charts(ctx.truth))
        groups = chart_groups(ctx)
        predicted_count = sum(len(group.points) for group in groups)
        hits = sum(map(_group_hits, groups))
        return 2 * hits / (truth_count + predicted_count)


def _truth_series_names(chart: Element) -> tuple[str, ...]:
    if chart.chart is None:
        return ()
    return tuple(s.name for s in chart.chart.series if chart_key(s.name))


def _unique_names(candidates: Iterable[str]) -> list[str]:
    names: list[str] = []
    for name in candidates:
        if chart_key(name) and chart_key(name) not in map(chart_key, names):
            names.append(name)
    return names


def _table_names(group: ChartGroup) -> Iterator[str]:
    for table in group.tables:
        yield from _unique_names(table.series_names)


def _either_way_match(first: str, second: str) -> bool:
    return series_matches(first, second) or series_matches(second, first)


def _group_names(group: ChartGroup) -> tuple[tuple[str, ...], ...]:
    if group.truth is not None and not _truth_series_names(group.truth):
        return ()
    json_names = _unique_names(s.name for s in group.json_series)
    json_keys = set(map(chart_key, json_names))
    table_names = [name for name in _table_names(group) if chart_key(name) not in json_keys]
    partners = maximum_matching(table_names, json_names, _either_way_match)
    paired = tuple(
        (json_name, table_names[partners[index]]) if index in partners else (json_name,)
        for index, json_name in enumerate(json_names)
    )
    partnered = set(partners.values())
    unpaired = _unique_names(
        name for index, name in enumerate(table_names) if index not in partnered
    )
    return paired + tuple((name,) for name in unpaired)


def _hits_truth_name(members: tuple[str, ...], truth_name: str) -> bool:
    return any(series_matches(member, truth_name) for member in members)


def _group_name_hits(group: ChartGroup) -> int:
    if group.truth is None:
        return 0
    return _matched_count(_truth_series_names(group.truth), _group_names(group), _hits_truth_name)


class Series(SubMetric):
    name = "series"
    source = Source.EITHER

    def applies(self, truth: Groundtruth) -> bool:
        return any(_truth_series_names(chart) for chart in truth_charts(truth))

    def score(self, ctx: DocContext) -> float:
        truth_count = sum(len(_truth_series_names(chart)) for chart in truth_charts(ctx.truth))
        groups = chart_groups(ctx)
        predicted_count = sum(len(_group_names(group)) for group in groups)
        if predicted_count == 0:
            return 0.0
        hits = sum(map(_group_name_hits, groups))
        return 2 * hits / (truth_count + predicted_count)


def _kind(holder: Element | LayoutItem | None) -> ChartKind | None:
    return None if holder is None or holder.chart is None else holder.chart.kind


def _has_matched_kind(group: ChartGroup) -> bool:
    return _kind(group.truth) is not None and _kind(group.truth) is _kind(group.item)


class ChartType(SubMetric):
    name = "chart_type"
    source = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        return bool(truth_charts(truth))

    def score(self, ctx: DocContext) -> float:
        right = sum(map(_has_matched_kind, chart_groups(ctx)))
        return right / len(truth_charts(ctx.truth))


def _title(holder: Element | LayoutItem | None) -> str | None:
    return None if holder is None or holder.chart is None else holder.chart.title


def _title_candidates(
    group: ChartGroup, markdown: str, tables: Sequence[MarkdownTable]
) -> Iterator[str]:
    if (title := _title(group.item)) is not None:
        yield title
    if group.item is not None:
        yield from group.item.text.splitlines()
    for oriented in group.tables:
        yield from touching_lines(markdown, oriented.table, tables)


def _title_recovered(group: ChartGroup, markdown: str, tables: Sequence[MarkdownTable]) -> bool:
    title = _title(group.truth)
    if title is None:
        return False
    expected = chart_key(title)
    candidates = _title_candidates(group, markdown, tables)
    return any(expected in title_keys(candidate) for candidate in candidates)


class ChartTitle(SubMetric):
    name = "chart_title"
    source = Source.EITHER

    def applies(self, truth: Groundtruth) -> bool:
        return any(_title(chart) is not None for chart in truth_charts(truth))

    def score(self, ctx: DocContext) -> float:
        titled = sum(_title(chart) is not None for chart in truth_charts(ctx.truth))
        tables = tuple(link.table for link in linked_tables(ctx))
        recovered = sum(
            _title_recovered(group, ctx.pred.markdown, tables) for group in chart_groups(ctx)
        )
        return recovered / titled


TRACK = Track(
    id="T6",
    name="Charts",
    parts=(
        (DataPoints(), 1.0),
        (Series(), 0.5),
        (ChartType(), 0.2),
        (ChartTitle(), 0.2),
    ),
)
