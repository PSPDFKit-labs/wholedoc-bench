from collections.abc import Mapping, Sequence

import pytest

from tests.builders import (
    BOTTOM_RIGHT,
    CHART_BOX,
    PROFIT,
    REGIONS,
    REVENUE,
    TITLE,
    TOP_LEFT,
    ChartSeriesData,
    as_prediction,
    cells_text,
    chart_element,
    chart_grid,
    chart_item,
    chart_tables,
    chart_text,
    groundtruth,
    html_table,
    item_series,
    json_item_with_linked_table,
    json_only,
    layout_item,
    layout_over,
    markdown_of,
    markdown_only,
    named_series,
    one_chart,
    para,
    pipe_table,
    prediction,
    quoted,
    series_pipe,
    table_element,
    truth_with_series,
    two_charts,
    two_labels,
    two_named,
)
from wholedoc_bench.schema import (
    Box,
    ChartKind,
    ElementClass,
    Groundtruth,
    LayoutItem,
    Prediction,
)
from wholedoc_bench.scorer.base import SubMetric
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t6_charts import TRACK, ChartTitle, ChartType, DataPoints, Series
from wholedoc_bench.scorer.utils.tracks.t4_tables.table_matching import output_tables
from wholedoc_bench.scorer.utils.tracks.table_links import linked_tables

BAR, LINE, AREA, PIE, SCATTER, DONUT = (
    ChartKind.BAR,
    ChartKind.LINE,
    ChartKind.AREA,
    ChartKind.PIE,
    ChartKind.SCATTER,
    ChartKind.DONUT,
)
BOX, TL, BR = CHART_BOX, TOP_LEFT, BOTTOM_RIGHT
UPPER: Box = (0.0, 0.0, 0.5, 0.5)
MIDDLE: Box = (0.0, 0.125, 0.5, 0.5)
LOWER: Box = (0.0, 0.25, 0.5, 0.5)
SPREAD_FACTORS = (0.6, 0.8, 1.0, 1.2, 1.4)
EXTRA_WORDS = ("Revenue", "Profit", "Costs", "West", "East", "Total")
OTHER_TITLE = "Units by region"

Q1_Q2: ChartSeriesData = {"Revenue": [("Q1", 10.0), ("Q2", 20.0)]}
CONFLICTING: ChartSeriesData = {"Revenue": [("Q1", 15.0), ("Q2", 30.0), ("Q3", 25.0), ("Q4", 35.0)]}
COSTS: ChartSeriesData = {"Costs": [("Rent", 5.0), ("Staff", 7.0)]}
UNNAMED: ChartSeriesData = {"": REVENUE["Revenue"]}
NORTH: ChartSeriesData = {"North": REGIONS["North"]}
NORTH_WEST: ChartSeriesData = {"North": REGIONS["North"], "West": REGIONS["South"]}
IN_MILLIONS: ChartSeriesData = {f"{name} ($m)": points for name, points in REGIONS.items()}
WITH_EU: ChartSeriesData = {f"{name} (EU)": points for name, points in REGIONS.items()}
LARGE: ChartSeriesData = {"Revenue": [("Q1", 1234567.0), ("Q2", 2345678.0)]}
TINY: ChartSeriesData = {"Revenue": [("Q1", 0.00001), ("Q2", 0.00002)]}
MILLIONS: ChartSeriesData = {"Revenue": [("Q1", 1000000.0), ("Q2", 2000000.0)]}
MORE_MILLIONS: ChartSeriesData = {"Revenue": [("Q1", 4000000.0), ("Q2", 5000000.0)]}
AXIS_TRUTH: ChartSeriesData = {"Revenue": [("Q1", 2.0), ("Q2", 500.0)]}
AXIS_TABLE: ChartSeriesData = {"Revenue": [("Q1", 6.0), ("Q2", 500.0)]}
FORECAST: ChartSeriesData = {
    "Revenue": [("2023", 10.0), ("2024", 12.0)],
    "Revenue (forecast)": [("2023", 11.0), ("2024", 14.0)],
}
FORECAST_FIRST: ChartSeriesData = dict(reversed(FORECAST.items()))
FORECAST_UNITS: ChartSeriesData = {
    "Revenue (forecast)": FORECAST["Revenue (forecast)"],
    "Revenue ($m)": FORECAST["Revenue"],
}
FORECAST_PLAIN: ChartSeriesData = {
    "Revenue": FORECAST["Revenue"],
    "Revenue (forecast) (EU)": FORECAST["Revenue (forecast)"],
}
LEVEL: ChartSeriesData = {"Revenue": [("2023", 10.0)], "Revenue (forecast)": [("2023", 10.0)]}
LEVEL_FIRST: ChartSeriesData = dict(reversed(LEVEL.items()))
LEVEL_TABLE: ChartSeriesData = {
    "Revenue (forecast)": LEVEL["Revenue"],
    "Revenue (forecast) (EU)": LEVEL["Revenue"],
}
PROFIT_MARGIN: ChartSeriesData = {**PROFIT, "Margin": [("Q1", 1.0), ("Q2", 2.0)]}
HALF_YEAR: ChartSeriesData = {"Revenue": [("Q1", 10.0), ("Q2", 12.0)]}
UNITS: ChartSeriesData = {"Units": [("North", 3.0), ("South", 4.0)]}


def t6(metric: str, truth: Groundtruth, pred: Prediction) -> float | None:
    return TRACK.evaluate(truth, DocContext(truth, pred)).sub_scores[metric]


def f1(hits: int, truth_count: int, predicted_count: int) -> float:
    return 2 * hits / (truth_count + predicted_count)


def pipe(header: list[str], *rows: list[str]) -> str:
    return pipe_table(header, list(rows))


def md(*blocks: str) -> Prediction:
    return markdown_only(*blocks)


def tab(series: ChartSeriesData = REVENUE, transpose: bool = False) -> str:
    return series_pipe(series, transpose=transpose)


def linked(json: ChartSeriesData, table: ChartSeriesData, copies: int = 1) -> Prediction:
    return json_item_with_linked_table(json, table, copies=copies)


def charts(*specs: tuple[ChartKind | None, Box], pages: int | None = None) -> Groundtruth:
    elements = [chart_element(f"c{n}", kind=k, box=b) for n, (k, b) in enumerate(specs, 1)]
    return groundtruth(elements, page_count=pages)


def items(
    *specs: tuple[ChartKind | None, Box | None], extra: Sequence[LayoutItem] = ()
) -> Prediction:
    layout = [chart_item(f"p{n}", kind, box=box) for n, (kind, box) in enumerate(specs, 1)]
    return prediction(layout=[*layout, *extra])


def json_charts(*series: tuple[ChartSeriesData, Box]) -> Prediction:
    layout = [chart_item(f"p{n}", BAR, box=b, series=s) for n, (s, b) in enumerate(series, 1)]
    return prediction(layout=layout)


def on_chart_item(*tables: str, text: str = chart_text(), **fields) -> Prediction:
    return prediction(markdown_of(*tables), [chart_item("p1", BAR, text=text, **fields)])


def with_unmatched_table(series: ChartSeriesData) -> Prediction:
    unmatched = chart_item("p2", None, box=BR, text=chart_text(COSTS))
    return prediction(markdown_of(tab(COSTS)), [chart_item("p1", BAR, series=series), unmatched])


def own(truth: Groundtruth) -> tuple[Groundtruth, Prediction, float]:
    return truth, as_prediction(truth), 1.0


def edited(series: ChartSeriesData, row: int, column: int, cell: str) -> Prediction:
    header, rows = chart_grid(series)
    rows[row][column] = cell
    return md(pipe_table(header, rows))


def headers(*names: str) -> Prediction:
    header, rows = chart_grid(REGIONS)
    return md(pipe_table([header[0], *names], rows))


def renamed(names: dict[str, str], data: ChartSeriesData = REGIONS) -> ChartSeriesData:
    return {names.get(name, name): points for name, points in data.items()}


def spread(series: ChartSeriesData) -> ChartSeriesData:
    points = next(iter(series.values()))
    return {f"guess {n}": [(lb, v * f) for lb, v in points] for n, f in enumerate(SPREAD_FACTORS)}


def with_data_free_chart(series: ChartSeriesData) -> Groundtruth:
    free = chart_element("c2", "Costs by item: Rent Staff", box=BR)
    return groundtruth([chart_element("c1", kind=BAR, box=TL, series=series), free])


def on_next_page(series: ChartSeriesData) -> tuple[Groundtruth, Prediction]:
    truth = groundtruth([chart_element("c1", kind=BAR, box=BOX, series=series)], page_count=2)
    return truth, prediction(layout=[chart_item("p1", BAR, page=1, series=series)])


def own_two(first: ChartSeriesData, second: ChartSeriesData, **axis) -> tuple:
    c2 = chart_element("c2", kind=LINE, box=BR, series=second, **axis)
    return own(groundtruth([chart_element("c1", kind=BAR, box=TL, series=first), c2]))


def cases(*tables: tuple[Groundtruth | None, Mapping[str, tuple]]) -> tuple[list[tuple], list[str]]:
    rows = [(truth, *row) if truth else row for truth, table in tables for row in table.values()]
    return rows, [name for _, table in tables for name in table]


NO_DATA = groundtruth([chart_element("c1", "Revenue Q1 10")])
UNTITLED_CHARTS = groundtruth(
    [
        chart_element("c1", kind=BAR, box=TL, series=HALF_YEAR),
        chart_element("c2", kind=LINE, box=BR),
        chart_element("c3", text=TITLE),
    ]
)
TITLED_SECOND = groundtruth(
    [chart_element("c1", kind=BAR, box=TL), chart_element("c2", kind=BAR, title=TITLE, box=BR)]
)
APPLIES = [
    (ChartType(), groundtruth([para("t1", "x"), *charts((BAR, BOX)).elements]), True),
    (ChartType(), charts((None, BOX)), False),
    (DataPoints(), one_chart(), True),
    (DataPoints(), charts((BAR, BOX)), False),
    (DataPoints(), NO_DATA, False),
    (Series(), one_chart(REGIONS), True),
    (Series(), two_charts(UNNAMED, PROFIT), True),
    (Series(), one_chart(UNNAMED), False),
    (Series(), charts((BAR, BOX)), False),
    (ChartTitle(), TITLED_SECOND, True),
    (ChartTitle(), UNTITLED_CHARTS, False),
]


@pytest.mark.parametrize(("metric", "truth", "expected"), APPLIES)
def test_applies_reads_the_groundtruth(
    metric: SubMetric, truth: Groundtruth, expected: bool
) -> None:
    assert metric.applies(truth) is expected


KIND_SPOTS = [(BAR, 0, TL), (PIE, 0, BR), (LINE, 1, BOX), (AREA, 2, TL), (SCATTER, 2, BR)]
ALL_KINDS = groundtruth(
    [para("t1", "Quarterly results")]
    + [chart_element(f"c{n}", kind=k, page=p, box=b) for n, (k, p, b) in enumerate(KIND_SPOTS)],
    page_count=3,
)
BAR_PIE = charts((BAR, TL), (PIE, BR))
BAR_PIE_2 = charts((BAR, TL), (PIE, BR), pages=2)
EXTRAS = [chart_item("x1", AREA, box=(0.0, 0.75, 0.25, 0.25)), chart_item("x2", SCATTER, page=1)]
EXTRAS += [chart_item("x3", None, box=None)]
FIGURE = LayoutItem(id="f", cls=ElementClass.FIGURE, page=0, box=BOX)
TWIN_TRUTH = groundtruth([chart_element("c1", kind=BAR, box=UPPER, text=TITLE)])
DECIMAL_TIE = charts((BAR, (0, 0, 0.5, 0.1)), (LINE, (0, 0.05, 0.5, 0.1)))


def twin(twin_kind: ChartKind, overlap: ChartKind) -> Prediction:
    layout = [
        chart_item("t", twin_kind, box=LOWER, text=TITLE),
        chart_item("o", overlap, box=UPPER),
    ]
    return prediction(layout=layout)


def half(height: float, width: float = 0.5) -> tuple[Groundtruth, Prediction]:
    return charts((BAR, (0, 0, width, height))), items((BAR, (0, 0, width, height / 2)))


CHART_TYPE_ON_BAR = {
    "no box": (items((BAR, None)), 0),
    "no chart data": (items((None, BOX)), 0),
    "other page": (prediction(layout=[chart_item("p1", BAR, page=1)]), 0),
    "figure takes no match": (prediction(layout=[FIGURE, chart_item("p2", BAR)]), 1),
    "tie to earlier item": (items((BAR, BOX), (LINE, BOX)), 1),
    "tie skips later item": (items((LINE, BOX), (BAR, BOX)), 0),
    "markdown only": (Prediction(markdown="Figure 1: revenue.\n\n| Q1 | 10 |"), 0),
}
CHART_TYPE_ON_UPPER = {
    "overlap of one half": (items((BAR, (0, 0, 0.5, 0.25))), 1),
    "just under one half": (items((BAR, (0, 0, 0.5, 0.2421875))), 0),
    "under one half by float rounding only": (items((BAR, (0, 0, 0.5, 0.2499999996))), 1),
    "overlap beats kind": (items((BAR, MIDDLE), (LINE, UPPER)), 0),
}
CHART_TYPE_OTHER = {
    "own prediction": own(ALL_KINDS),
    "other order": (BAR_PIE, items((PIE, BR), (BAR, TL)), 1),
    "position, not kind": (BAR_PIE, items((PIE, TL), (BAR, BR)), 0),
    "donut is not a pie": (BAR_PIE, items((BAR, TL), (DONUT, BR)), 0.5),
    "chart not predicted": (BAR_PIE, items((BAR, TL)), 0.5),
    "extras, all right": (BAR_PIE_2, items((BAR, TL), (PIE, BR), extra=EXTRAS), 1),
    "text twin, right kind": (TWIN_TRUTH, twin(LINE, BAR), 1),
    "half on decimal boxes": (*half(0.4), 1),
    "data-free chart takes no item": (charts((None, UPPER), (BAR, MIDDLE)), items((BAR, UPPER)), 1),
    "one item, one chart": (charts((BAR, UPPER), (BAR, LOWER)), items((BAR, MIDDLE)), 0.5),
    "highest overlap first": (charts((BAR, UPPER), (LINE, MIDDLE)), items((LINE, MIDDLE)), 0.5),
    "tie to earlier chart": (charts((BAR, UPPER), (LINE, LOWER)), items((BAR, MIDDLE)), 0.5),
    "tie not to later chart": (charts((BAR, UPPER), (LINE, LOWER)), items((LINE, MIDDLE)), 0),
    "tie on decimal boxes": (DECIMAL_TIE, items((BAR, (0, 0, 0.5, 0.2))), 0.5),
    "page-sized chart": (charts((BAR, TL), (BAR, BR)), items((BAR, (0, 0, 1, 1))), 0),
}
CHART_TYPE, CHART_TYPE_IDS = cases(
    (charts((BAR, BOX)), CHART_TYPE_ON_BAR),
    (charts((BAR, UPPER)), CHART_TYPE_ON_UPPER),
    (None, CHART_TYPE_OTHER),
)


@pytest.mark.parametrize(("truth", "pred", "expected"), CHART_TYPE, ids=CHART_TYPE_IDS)
def test_chart_type(truth: Groundtruth, pred: Prediction, expected: float) -> None:
    assert t6("chart_type", truth, pred) == pytest.approx(expected)


def literal(*values: str) -> str:
    return pipe(["Quarter", "Revenue"], *([f"Q{n}", value] for n, value in enumerate(values, 1)))


LARGE_TABLE, TINY_TABLE = literal("1234567", "2345678"), literal("0.00001", "0.00002")
MILLIONS_TABLE, MORE_MILLIONS_TABLE = literal("1000000", "2000000"), literal("4000000", "5000000")
REVENUE_GRID, REGIONS_GRID = chart_grid(REVENUE), chart_grid(REGIONS)
HEADER_AS_ROW = [REVENUE_GRID[0], *REVENUE_GRID[1]]
LABELS, VALUES = [lb for lb, _ in REVENUE["Revenue"]], [f"{v:g}" for _, v in REVENUE["Revenue"]]
SQUARE = pipe_table(["", *LABELS], [[label, *VALUES] for label in LABELS])
MULTI_HEADER = (
    "<table><thead><tr><th>Group</th><th>Sales</th><th>Sales</th></tr>"
    "<tr><th>Quarter</th><th>North</th><th>South</th></tr></thead><tbody>"
    "<tr><td>Q1</td><td>10</td><td>30</td></tr><tr><td>Q2</td><td>20</td><td>40</td></tr>"
    "</tbody></table>"
)
CALLED_TABLE = prediction(
    markdown_of(tab()), [layout_item("t", ElementClass.TABLE, text=chart_text())]
)
NEAR = {"Revenue": [(label, value * 1.005) for label, value in REVENUE["Revenue"]]}
ONE_OFF = {"Revenue": [*REVENUE["Revenue"][:2], ("Q3", 99.0), REVENUE["Revenue"][3]]}
SWAPPED = {"North": REGIONS["South"], "South": REGIONS["North"]}
SWAPPED_COLUMN = {**NORTH, "South": [("Q1", 40.0), ("Q2", 30.0)]}
REPEATED_LABELS = [("Q1", 10.0), ("q1", 15.0), ("Q1", 99.0), *REVENUE["Revenue"][1:]]
NORMALIZED = pipe(
    ["Quarter", "**north**", "SOUTH"], ["<i>q1</i>", "10", "30"], [" Q2 ", "20", "40"]
)
ESTIMATES = pipe(["Label", "North", "South"], ["Q1 (est)", "10", "30"], ["Q2 (est)", "20", "40"])
SQUARE_REGIONS = pipe(["", "Q1", "Q2"], ["Q1", "10", "30"], ["Q2", "20", "40"])
TWO_JSON = two_charts(Q1_Q2, PROFIT)
BAR_AND_PIE = groundtruth(
    [chart_element("c1", kind=BAR, box=TL, series=REVENUE), chart_element("c2", kind=PIE, box=BR)]
)
COSTS_TABLE = pipe(["Item", "Costs"], ["Rent", "5"], ["Staff", "7"])
THREE_QUARTERS = one_chart({"Revenue": [("Q1", 10.0), ("Q2", 20.0), ("Q3", 30.0)]})
MORE_LABELS = pipe(["", "Q1", "Q2", "Q3"], ["Q1", "10", "20", "30"])
TIE = pipe(["", "Q1", "Q2"], ["Q1", "10", "20"], ["Q2", "0", "0"])
NINES = {"Profit": [("Q1", 9.0), ("Q2", 9.0)]}
SCALED_JSON: ChartSeriesData = {"Revenue": [("Q1", 2010000.0)]}
SCALED_TABLE = pipe(["Label", "Revenue"], ["Q1", "2.01m"])
SCALED_ITEM = chart_item("p1", BAR, series=SCALED_JSON, md_start=0, md_end=len(SCALED_TABLE))
ENTITY: ChartSeriesData = {"Revenue &amp;": [("<b>Q1</b>", 10.0)]}
DECODED: ChartSeriesData = {"Revenue &": [("Q1", 10.0)]}
LABEL_PAIRS = [("Q1", "**Q1**"), ("Q1", "Q 1")]
NAME_PAIRS = [("A", "*A*"), ("", "South")]
ONE_LABEL_TWO_SERIES = truth_with_series(
    named_series("North", [("Q1", 1.0)]), named_series("South", [("Q1", 2.0)])
)
REVENUE_POINTS, CONFLICTING_POINTS = REVENUE["Revenue"], CONFLICTING["Revenue"]

DATA_POINTS_ON_REVENUE = {
    "perfect pipe": (md(tab()), 1),
    "transposed": (md(tab(transpose=True)), 1),
    "no output": (md("No tables in this output."), 0),
    "one wrong value": (edited(REVENUE, 2, 1, "99"), f1(3, 4, 4)),
    "wrong label": (edited(REVENUE, 0, 0, "Q5"), f1(3, 4, 4)),
    "missing rows": (md(pipe(REVENUE_GRID[0], REVENUE_GRID[1][0])), f1(1, 4, 1)),
    "single series header": (md(tab({"Something else": REVENUE_POINTS})), 1),
    "pipe with empty header": (md(pipe_table(None, HEADER_AS_ROW)), 1),
    "linked repeats json": (linked(REVENUE, REVENUE), 1),
    "linked within tolerance": (linked(REVENUE, NEAR), f1(4, 4, 8)),
    "linked one conflict": (linked(REVENUE, ONE_OFF), f1(4, 4, 5)),
    "header next to unnamed json": (linked(UNNAMED, REVENUE), 1),
    "two linked copies": (linked(REVENUE, REVENUE, 2), f1(4, 4, 8)),
    "unmatched predicted chart": (json_charts((REVENUE, BOX), (PROFIT, BR)), f1(4, 4, 6)),
    "table of unmatched item": (with_unmatched_table(REVENUE), f1(4, 4, 6)),
    "right then wrong table": (md(tab(), tab(CONFLICTING)), f1(4, 4, 8)),
    "spread table": (md(tab(spread(REVENUE))), f1(4, 4, 20)),
    "labels on both sides": (md(SQUARE), f1(4, 4, 16)),
    "layout calls it a table": (CALLED_TABLE, 0),
    "empty predicted series": (item_series(("Revenue", REVENUE_POINTS), ("", [])), 1),
    "two unnamed series": (
        item_series(("", REVENUE_POINTS), ("", CONFLICTING_POINTS)),
        f1(4, 4, 8),
    ),
    "repeated labels": (item_series(("Revenue", REPEATED_LABELS)), f1(4, 4, 6)),
    "item listed twice": (prediction(layout=[chart_item("p1", BAR, series=REVENUE)] * 2), 2 / 3),
}
DATA_POINTS_ON_REGIONS = {
    "multi-series transposed": (md(tab(REGIONS, transpose=True)), 1),
    "multi-series html": (md(html_table(*REGIONS_GRID)), 1),
    "wrong series header": (headers("South", "North"), 0),
    "swapped json column": (json_only(SWAPPED_COLUMN), f1(2, 4, 4)),
    "normalized names and labels": (md(NORMALIZED), 1),
    "header unit": (headers("North ($m)", "South ($m)"), 1),
    "two trailing groups": (headers("North ($m) (2024)", "South ($m) (2024)"), 0),
    "leading group": (headers("($m) North", "($m) South"), 0),
    "label with trailing unit": (md(ESTIMATES), 0),
    "last header row names": (md(MULTI_HEADER), 1),
    "linked wrong header": (linked(REGIONS, SWAPPED), f1(4, 4, 8)),
    "linked unit headers": (linked(REGIONS, IN_MILLIONS), 1),
    "square multi-series": (md(SQUARE_REGIONS), 0),
    "two series of one name": (
        item_series(("North", NORTH["North"]), ("North", SWAPPED["North"])),
        0.5,
    ),
}
DATA_POINTS_OTHER = {
    "own json prediction": own_two(REVENUE, REGIONS, value_axis=(0.0, 50.0)),
    "chart text shares no word": (one_chart(text="Figure 1"), md(tab()), 1),
    "tiny then large": (two_charts(LARGE, TINY), md(TINY_TABLE, LARGE_TABLE), 1),
    "millions reversed": (
        two_charts(MILLIONS, MORE_MILLIONS),
        md(MORE_MILLIONS_TABLE, MILLIONS_TABLE),
        1,
    ),
    "name with parentheses": (one_chart(WITH_EU), md(tab(WITH_EU)), 1),
    "trailing group series first": (one_chart(LEVEL), json_only(LEVEL_FIRST), 1),
    "table repeats across groups": (one_chart(LEVEL), linked(LEVEL_FIRST, LEVEL_TABLE), 1),
    "name without its unit": (one_chart(WITH_EU), md(tab(REGIONS)), 0),
    "more labels are labels": (THREE_QUARTERS, md(MORE_LABELS), 1),
    "orientation tie": (one_chart(Q1_Q2), md(TIE), f1(1, 2, 4)),
    "table repeats a scaled json value": (
        one_chart(SCALED_JSON),
        prediction(markdown_of(SCALED_TABLE), [SCALED_ITEM]),
        1,
    ),
    "json names stay plain": (one_chart(ENTITY), json_only(ENTITY), 1),
    "decoded json names": (one_chart(ENTITY), json_only(DECODED), 0),
    "repeat inside the axis tolerance": (
        one_chart(AXIS_TRUTH, value_axis=(0, 1000)),
        linked(AXIS_TRUTH, AXIS_TABLE),
        f1(2, 2, 3),
    ),
    "own json groups": (TWO_JSON, json_charts((Q1_Q2, TL), (PROFIT, BR)), 1),
    "swapped json groups": (TWO_JSON, json_charts((PROFIT, TL), (Q1_Q2, BR)), 0),
    "own tables without layout": (two_charts(REVENUE, COSTS), md(tab(COSTS), tab()), 1),
    "f1 over the document": (
        two_charts(REVENUE, PROFIT),
        json_charts((REVENUE, TL), (NINES, BR)),
        2 / 3,
    ),
    "chart not predicted": (*on_next_page(REVENUE), 0),
    "points on a seriesless chart": (BAR_AND_PIE, json_charts((REVENUE, TL), (PROFIT, BR)), 0.8),
    "table of a data-free chart": (with_data_free_chart(REVENUE), md(tab(), COSTS_TABLE), 0.8),
    **{f"labels {a!r} {b!r}": own(two_labels(a, b)) for a, b in LABEL_PAIRS},
    "one label in two series": own(ONE_LABEL_TWO_SERIES),
    **{f"names {a!r} {b!r}": own(two_named(a, b)) for a, b in NAME_PAIRS},
}
DATA_POINTS, DATA_POINTS_IDS = cases(
    (one_chart(), DATA_POINTS_ON_REVENUE),
    (one_chart(REGIONS), DATA_POINTS_ON_REGIONS),
    (None, DATA_POINTS_OTHER),
)


@pytest.mark.parametrize(("truth", "pred", "expected"), DATA_POINTS, ids=DATA_POINTS_IDS)
def test_data_points(truth: Groundtruth, pred: Prediction, expected: float) -> None:
    assert t6("data_points", truth, pred) == pytest.approx(expected)


HUNDRED = one_chart({"Revenue": [("Q1", 100.0)]})


@pytest.mark.parametrize(("json_value", "table_value"), [(101.2, 100.3), (100.3, 101.2)])
def test_different_json_and_table_values_both_count(json_value: float, table_value: float) -> None:
    pred = linked({"Revenue": [("Q1", json_value)]}, {"Revenue": [("Q1", table_value)]})
    assert t6("data_points", HUNDRED, pred) == pytest.approx(f1(1, 1, 2))


def over_table(item: LayoutItem) -> Prediction:
    table = tab()
    spanned = item.model_copy(update={"md_start": 0, "md_end": len(table)})
    return prediction(markdown_of(table), [spanned])


BELOW_OVERLAP: Box = (0.125, 0.125, 0.5, 0.12)
LAYOUT_VARIANTS = {
    "chart labelled figure": over_table(FIGURE),
    "chart box below the overlap": over_table(chart_item("p1", None, box=BELOW_OVERLAP)),
    "chart without a box": over_table(chart_item("p1", None, box=None)),
}


@pytest.mark.parametrize("pred", LAYOUT_VARIANTS.values(), ids=LAYOUT_VARIANTS)
def test_layout_that_misses_the_chart_keeps_the_table_credit(pred: Prediction) -> None:
    assert t6("data_points", one_chart(), pred) == 1.0


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("chart_first", [False, True])
def test_chart_next_to_a_same_numbers_table_scores_once_per_track(
    chart_first: bool, layout: bool
) -> None:
    table = table_element("t1", 0, *chart_grid(REVENUE, "Quarter"))
    chart = chart_element("c1", kind=BAR, box=BOX, series=REVENUE)
    truth = groundtruth([chart, table] if chart_first else [table, chart])
    block = pipe_table(*chart_grid(REVENUE, "Quarter"))
    second = len(block) + len("\n\n")
    spans = [
        layout_item("t", ElementClass.TABLE, md_start=0, md_end=len(block)),
        chart_item("p1", None, md_start=second, md_end=second + len(block)),
    ]
    ctx = DocContext(truth, prediction(markdown_of(block, block), spans if layout else []))
    assert t6("data_points", truth, ctx.pred) == 1.0
    output, charts = (
        [t.index for t in output_tables(linked_tables(ctx))],
        [t.index for t in chart_tables(ctx)],
    )
    assert len(output) == len(charts) == 1
    assert sorted(output + charts) == [0, 1]
    assert not layout or (output, charts) == ([0], [1])


NORTH_TWICE_GRID = (
    ["Label", "North ($m)", "North ($m)"],
    [[lb, f"{v:g}", f"{v:g}"] for lb, v in NORTH["North"]],
)
NORTH_TWICE = pipe_table(*NORTH_TWICE_GRID)
ABSORBED_TWICE = on_chart_item(NORTH_TWICE, text=cells_text(*NORTH_TWICE_GRID), series=NORTH)
WIDE_HEADER = [*REGIONS_GRID[0], *EXTRA_WORDS]
WIDE_ROWS = [[*row, *(["1"] * len(EXTRA_WORDS))] for row in REGIONS_GRID[1]]
SIDES = ["Q1", "Q2", "North", "South"]
BOTH_SIDES = pipe_table(["", *SIDES], [[side, "1", "2", "3", "4"] for side in SIDES])
COPIED_ROWS = [["North", "1", "2"], ["South", "3", "4"]]
AS_LABELS = pipe(["Label", "A", "B"], ["Q1", "10", "30"], ["Q2", "20", "40"], *COPIED_ROWS)
UNDER_TABLE = pipe_table(REGIONS_GRID[0], [*REGIONS_GRID[1], *COPIED_ROWS])
ORPHAN = pipe(["Item", "Costs", "Budget"], ["Rent", "5", "6"], ["Staff", "7", "8"])
AMERICA = one_chart(renamed({"North": "North America"}))
SPACED_TABLE = md(tab(renamed({"North": "NORTH   AMERICA", "South": "south"})))
NORTH_IN_MILLIONS = {"North ($m)": REGIONS["North"]}
TWO_NORTHS = one_chart({"North": REGIONS["North"], "North ($m)": REGIONS["South"]})
WITH_SOUTH = {**PROFIT, "South": REGIONS["South"]}
REGIONS_TEXT = chart_text(REGIONS)


def groups(first: ChartSeriesData, second: ChartSeriesData) -> Prediction:
    return json_charts((first, TL), (second, BR))


SERIES_ON_REGIONS = {
    "perfect multi-series pipe": (md(tab(REGIONS)), 1),
    "no output": (md("No tables in this output."), 0),
    "transposed": (md(tab(REGIONS, transpose=True)), 1),
    "half right": (md(tab(NORTH_WEST)), 0.5),
    "name twice in a group": (on_chart_item(tab(REGIONS), tab(REGIONS), text=REGIONS_TEXT), 1),
    "extra name in a second table": (
        on_chart_item(tab(REGIONS), tab(NORTH_WEST), text=REGIONS_TEXT),
        f1(2, 2, 3),
    ),
    "json names differ in case": (json_only({**REGIONS, "NORTH": REGIONS["North"]}), 1),
    "json name in linked table": (linked(REGIONS, REGIONS), 1),
    "json name, unit in table": (linked(REGIONS, IN_MILLIONS), 1),
    "json unit, table without": (linked(IN_MILLIONS, REGIONS), 1),
    "absorbs one table name": (linked(REGIONS, IN_MILLIONS, 2), f1(2, 2, 4)),
    "json name in two tables": (linked(REGIONS, REGIONS, 2), 1),
    "table name json lacks": (linked(REGIONS, NORTH_WEST), f1(2, 2, 3)),
    "one name, one match": (
        json_only({"North ($m)": NORTH["North"], "North (%)": SWAPPED["North"]}),
        0.5,
    ),
    "empty json name": (json_only({"": REVENUE_POINTS, **REGIONS}), 1),
    "empty table header": (md(tab({**NORTH, "": REGIONS["South"]})), f1(1, 2, 1)),
    "same name in two groups": (json_charts((REGIONS, BOX), (NORTH, BR)), f1(2, 2, 3)),
    "every word as a header": (md(pipe_table(WIDE_HEADER, WIDE_ROWS)), f1(2, 2, 8)),
    "names on both sides": (md(BOTH_SIDES), f1(2, 2, 4)),
    "names copied as labels": (md(AS_LABELS), 0),
    "copied labels add nothing": (md(UNDER_TABLE), 1),
}
SERIES_OTHER = {
    "own json prediction": own_two(REVENUE, REGIONS),
    "perfect single-series pipe": (one_chart(REVENUE), md(tab(REVENUE)), 1),
    "json trailing group first": (one_chart(FORECAST), json_only(FORECAST_FIRST), 1),
    "table trailing group first": (one_chart(FORECAST), md(tab(FORECAST_FIRST)), 1),
    "table names pair across groups": (
        one_chart(FORECAST),
        linked(FORECAST_UNITS, FORECAST_PLAIN),
        1,
    ),
    "json without names": (one_chart(REVENUE), json_only(UNNAMED), 0),
    "case and spaces in a table": (AMERICA, SPACED_TABLE, 1),
    "table name has truth unit": (
        one_chart(NORTH_IN_MILLIONS),
        linked(NORTH, NORTH_IN_MILLIONS),
        1,
    ),
    "absorbed name twice in a table": (one_chart(NORTH), ABSORBED_TWICE, 1),
    "name without the truth unit": (one_chart(IN_MILLIONS), json_only(REGIONS), 0),
    "one name, not two hits": (TWO_NORTHS, json_only(NORTH_IN_MILLIONS), f1(1, 2, 1)),
    "unnamed truth series": (two_charts(UNNAMED, PROFIT), groups(UNNAMED, PROFIT), 1),
    "header of a nameless chart": (
        two_charts(UNNAMED, PROFIT),
        md(tab(), tab(PROFIT_MARGIN)),
        2 / 3,
    ),
    "names in own groups": (two_charts(REGIONS, PROFIT), groups(REGIONS, PROFIT), 1),
    "names in swapped groups": (two_charts(REGIONS, PROFIT), groups(PROFIT, REGIONS), 0),
    "one name in the wrong group": (two_charts(REGIONS, PROFIT), groups(NORTH, WITH_SOUTH), 2 / 3),
    "wrong single-series name": (one_chart(REVENUE), md(tab({"Sales": REVENUE_POINTS})), 0),
    "orphan table names": (with_data_free_chart(REGIONS), md(tab(REGIONS), ORPHAN), f1(2, 2, 4)),
    "json names stay plain": (one_chart(ENTITY), json_only(ENTITY), 1),
    "decoded json names": (one_chart(ENTITY), json_only(DECODED), 0),
}
SERIES, SERIES_IDS = cases((one_chart(REGIONS), SERIES_ON_REGIONS), (None, SERIES_OTHER))


@pytest.mark.parametrize(("truth", "pred", "expected"), SERIES, ids=SERIES_IDS)
def test_series(truth: Groundtruth, pred: Prediction, expected: float) -> None:
    assert t6("series", truth, pred) == pytest.approx(expected)


REVENUE_TABLE, UNITS_TABLE = tab(HALF_YEAR), tab(UNITS)
REVENUE_HTML = html_table(*chart_grid(HALF_YEAR))
OTHER_TABLE = pipe(["Office", "Staff"], ["Paris", "40"], ["Berlin", "55"])
CRLF_TABLE = REVENUE_TABLE.replace("\n", "\r\n")
INDENTED_TABLE = REVENUE_TABLE.replace("\n", "\n   ")
GAPS = ("\n", "\n   \n\t\n")


def titled(title: str | None = TITLE) -> Groundtruth:
    return groundtruth([chart_element("c1", kind=BAR, box=BOX, title=title, series=HALF_YEAR)])


def above(line: str, table: str = REVENUE_TABLE) -> str:
    return f"{line}\n\n{table}\n"


def below(line: str, table: str = REVENUE_TABLE) -> str:
    return f"Intro text.\n\n{table}\n\n{line}\n"


FORMS = [f"# {TITLE}", f"## **{TITLE}**", f"*{TITLE}*", f"- {TITLE}", f"<b>{TITLE}</b>"]
FORMS += [TITLE.upper(), "Revenue   by  quarter"]
EDGES = [
    f"{TITLE}:",
    f"{TITLE}.",
    f"**{TITLE}**:",
    f"**{TITLE}:**",
    f"### {TITLE}:",
]
EDGES += [f"<h3>{TITLE}</h3>", f"<p>{TITLE}</p>", f"<p><strong>{TITLE}</strong></p>"]
EDGES += [f"### {TITLE} ###"]
MARKERS = ("1. Revenue", "# of employees")
MARKER_LINES = ("{}", "### {}")
REPEATED_BODY = "\n\n".join([f"# {TITLE}", TITLE, f"**{TITLE}**", f"- {TITLE}", TITLE]) + "\n"
AROUND_TABLE = [REVENUE_TABLE, "## Staff", "## Offices"]
HEADINGS = (
    "\n\n".join(["# Annual report", f"## {TITLE}", "## Outlook", "## Risks", *AROUND_TABLE]) + "\n"
)
TOUCHING = "\n\n".join(["# Annual report", "## Outlook", f"## {TITLE}", *AROUND_TABLE]) + "\n"
ELSEWHERE = f"## {TITLE}\n\nThe company grew.\n\n## Results\n\n{REVENUE_TABLE}\n\n"
ELSEWHERE += f"Notes follow.\n\n{TITLE}\n"
SENTENCE = f"{TITLE} rose in every region.\n\n{REVENUE_TABLE}\n\nSee {TITLE} above.\n"
SAME_LINE = [
    f"Intro text.\n\n{before}{REVENUE_HTML}\n\nOutro text.\n" for before in (f"**{TITLE}** ",)
]
SAME_LINE += [f"Intro text.\n\n{REVENUE_HTML} **{TITLE}**\n\nOutro text.\n"]
CRLF = [
    f"### {TITLE}\r\n\r\n{CRLF_TABLE}\r\n",
    f"Intro text.\r\n\r\n{CRLF_TABLE}\r\n \r\n**{TITLE}**\r\n",
]
EDGE_OF_TEXT = [f"{REVENUE_TABLE}\n\n**{TITLE}**", f"### {TITLE}\n{REVENUE_TABLE}"]
QUOTED = [quoted(above(f"### {TITLE}")), quoted(below(f"**{TITLE}**"))]
PRINTED = {
    TITLE: [above(line) for line in FORMS]
    + [above(line) for line in EDGES]
    + [above(EDGES[0], REVENUE_HTML), below(EDGES[0], REVENUE_HTML)]
    + [f"### {TITLE}{gap}{REVENUE_TABLE}\n" for gap in GAPS]
    + [f"Intro text.\n\n{REVENUE_TABLE}\n{gap}**{TITLE}**\n" for gap in GAPS]
    + [TOUCHING, *CRLF, *EDGE_OF_TEXT, *SAME_LINE, *QUOTED],
    "R&D spend": [above("### R&amp;D spend")],
    f"Figure 3: {TITLE}": [above(f"### Figure 3: {TITLE}")],
    **{
        p: [
            above(line.format(r"\# of employees" if p == "# of employees" and line == "{}" else p))
            for line in MARKER_LINES
        ]
        for p in MARKERS
    },
    "Sales in the U.S.": [above("### Sales in the U.S.")],
}
NOT_PRINTED = {
    TITLE: [
        f"{TITLE}\nSource: company filings\n\n{REVENUE_TABLE}\n",
        below(f"Values in $m.\n**{TITLE}**"),
        quoted(below(f"Values in $m.\n**{TITLE}**")),
    ]
    + [ELSEWHERE, SENTENCE, above(f"### Figure 3: {TITLE}"), REPEATED_BODY, HEADINGS]
    + [
        f"{TITLE}\n\nSee below: {REVENUE_HTML}\n",
        f"Intro text.\n\n{REVENUE_HTML} note\n{TITLE}\n",
    ],
    "Figure 2. Revenue": [above(line) for line in ("Figure 3. Revenue", "Figure Revenue")],
    "Revenue 2020 - 2024": [above("### Revenue 2020 2024")],
    "1. Revenue": [above("### Revenue")],
    "# of employees": [above("### of employees")],
    "Revenue": [above("### 1. Revenue"), above("- # Revenue"), above("1234567890. Revenue")],
}
NOT_PRINTED[TITLE] += [
    above(f"1. {TITLE}"),
    above(f"2) {TITLE}"),
    f"1. **{TITLE}**\n\n   {INDENTED_TABLE}\n",
]
NOT_PRINTED["Revenue"] += [above("123456789. Revenue")]
NOT_PRINTED["# of employees"] += [above("# of employees")]
TITLE_LINES = [(title, text, 1.0) for title, lines in PRINTED.items() for text in lines]
TITLE_LINES += [(title, text, 0.0) for title, lines in NOT_PRINTED.items() for text in lines]


@pytest.mark.parametrize(("printed", "markdown", "expected"), TITLE_LINES)
def test_chart_title_from_markdown_lines(printed: str, markdown: str, expected: float) -> None:
    assert t6("chart_title", titled(printed), prediction(markdown)) == expected


def spanning(
    markdown: str, block: str, title: str | None = None, item_id: str = "p1", **fields
) -> LayoutItem:
    start = markdown.index(block)
    return chart_item(
        item_id, BAR, title=title, md_start=start, md_end=start + len(block), **fields
    )


def with_span(markdown: str, block: str, title: str | None = None, **fields) -> Prediction:
    return prediction(markdown, [spanning(markdown, block, title, **fields)])


def json_title(title: str | None, **fields) -> Prediction:
    return prediction(layout=[chart_item("p1", BAR, title=title, **fields)])


def titled_pair(second_title: str | None) -> Groundtruth:
    first = chart_element("c1", kind=BAR, title=TITLE, box=TL, series=HALF_YEAR)
    return groundtruth(
        [first, chart_element("c2", kind=BAR, title=second_title, box=BR, series=UNITS)]
    )


def next_to_other_table(markdown: str) -> Prediction:
    other = layout_over("t1", ElementClass.TABLE, markdown, OTHER_TABLE)
    chart = spanning(markdown, REVENUE_TABLE)
    order = (
        [other, chart]
        if markdown.index(OTHER_TABLE) < markdown.index(REVENUE_TABLE)
        else [chart, other]
    )
    return prediction(markdown, order)


TWO_TITLED, HALF_TITLED = titled_pair(OTHER_TITLE), titled_pair(None)
HEADED = above(f"### {TITLE}")
OTHER_ABOVE = f"### {TITLE}\n\n{OTHER_TABLE}\n\n{REVENUE_TABLE}\n"
OTHER_BELOW = f"Intro text.\n\n{REVENUE_TABLE}\n\n{OTHER_TABLE}\n\n**{TITLE}**\n"
GROUP_BLOCK = (
    f"{tab({'Revenue': [('Q1', 10.0)]})}\n\n### {TITLE}\n\n{tab({'Revenue': [('Q2', 12.0)]})}"
)
GROUP = f"Intro text.\n\n{GROUP_BLOCK}\n\nOutro text.\n"
TWO_TABLES = f"### {TITLE}\n\n{REVENUE_TABLE}\n\nSee the note.\n\n### Other\n\n{UNITS_TABLE}\n"
UNTITLED_SECOND = f"### {TITLE}\n\n{REVENUE_TABLE}\n\nSee the note.\n\n{UNITS_TABLE}\n"
WRONG = f"### {OTHER_TITLE}\n\n{REVENUE_TABLE}\n\nSee the note.\n\n"
WRONG += f"### {TITLE}\n\n{UNITS_TABLE}\n\nEnd.\n"
BACK_TO_BACK = f"{REVENUE_TABLE}\n\n**{TITLE}**\n\n### {OTHER_TITLE}\n\n{UNITS_TABLE}\n"
SAME_NUMBERS_TABLE = table_element(
    "t1", header=["Quarter", "Revenue"], rows=[["Q1", "10"], ["Q2", "12"]]
)
OTHER_CHART_TITLES = [
    chart_item("p1", BAR, title=OTHER_TITLE, box=TL),
    chart_item("p2", BAR, title=TITLE, box=BR),
]
UNMATCHED_TITLE = groundtruth([chart_element("c1", kind=BAR, title=TITLE, box=BR)])
TAGGED = groundtruth([chart_element("c1", kind=BAR, box=BOX, title="<b>Title</b>")])

CHART_TITLE_ON_TITLED = {
    "json title without series": (json_title("revenue BY Quarter"), 1),
    "markdown title, json-linked": (with_span(HEADED, REVENUE_TABLE, series=HALF_YEAR), 1),
    "unprinted json prefix": (json_title(f"Figure 3: {TITLE}"), 0),
    "wrong json, right line": (with_span(HEADED, REVENUE_TABLE, OTHER_TITLE), 1),
    "right json, wrong line": (with_span(above(f"### {OTHER_TITLE}"), REVENUE_TABLE, TITLE), 1),
    "printed heading marker in item text": (
        prediction(f"# {TITLE}\n", [chart_item("p1", BAR, text=f"# {TITLE}")]),
        0,
    ),
    "line of item text": (json_title(None, text=f"{TITLE}\n{chart_text(HALF_YEAR)}"), 1),
    "other line in item text": (json_title(None, text=f"{OTHER_TITLE}\n{TITLE} rose"), 0),
    "repeated body, json chart": (
        prediction(REPEATED_BODY, [chart_item("p1", BAR, series=HALF_YEAR)]),
        0,
    ),
    "title above another table": (next_to_other_table(OTHER_ABOVE), 0),
    "title below another table": (next_to_other_table(OTHER_BELOW), 0),
    "second table of a group": (with_span(GROUP, GROUP_BLOCK), 1),
}
CHART_TITLE_OTHER = {
    "own json prediction": own(TWO_TITLED),
    "json title, unmatched item": (UNMATCHED_TITLE, json_title(TITLE, box=TL), 0),
    "json title of another chart": (TWO_TITLED, prediction(layout=OTHER_CHART_TITLES), 0),
    "one of two titles": (TWO_TITLED, prediction(TWO_TABLES), 0.5),
    "untitled charts do not count": (HALF_TITLED, prediction(UNTITLED_SECOND), 1),
    "titles by the wrong tables": (TWO_TITLED, prediction(WRONG), 0),
    "same numbers as a table": (
        groundtruth([SAME_NUMBERS_TABLE, *titled().elements]),
        prediction(above(f"## {TITLE}")),
        1,
    ),
    "back to back tables": (TWO_TITLED, prediction(BACK_TO_BACK), 1),
    "other figure number in json": (
        titled("Figure 2. Revenue"),
        json_title("Figure 3. Revenue", series=HALF_YEAR),
        0,
    ),
    "json marker": (titled(MARKERS[0]), json_title(MARKERS[0], series=HALF_YEAR), 1),
    "json title keeps printed tags": (TAGGED, json_title("<b>Title</b>"), 1),
    "json title without its printed tags": (TAGGED, json_title("Title"), 0),
}
CHART_TITLE, CHART_TITLE_IDS = cases((titled(), CHART_TITLE_ON_TITLED), (None, CHART_TITLE_OTHER))


@pytest.mark.parametrize(("truth", "pred", "expected"), CHART_TITLE, ids=CHART_TITLE_IDS)
def test_chart_title(truth: Groundtruth, pred: Prediction, expected: float) -> None:
    assert t6("chart_title", truth, pred) == pytest.approx(expected)
