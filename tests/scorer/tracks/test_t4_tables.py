import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import pairwise

import pytest
from rapidfuzz.distance import Levenshtein

from tests.builders import (
    CHART_TEXT,
    HEADER,
    ROWS,
    Row,
    Rows,
    chart_element,
    chart_pipe,
    chart_tables,
    context,
    continues_in,
    grid,
    groundtruth,
    header_rows_table,
    html_row,
    html_table,
    indexes,
    layout_item,
    layout_over,
    logical_tables,
    markdown_of,
    one_table,
    only_table,
    output_table_anchors,
    para,
    perfect_html_markdown,
    perfect_pipe_markdown,
    pipe_table,
    split_truth,
    stays_apart,
    table_element,
    truth,
    two_different_tables,
)
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.schema import (
    CellSpan,
    Element,
    ElementClass,
    Groundtruth,
    LayoutItem,
    Prediction,
)
from wholedoc_bench.scorer.base import SubMetric
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t4_tables import (
    TRACK,
    CellNeighbourTests,
    HeaderMatchedRows,
    NeighbourRule,
    NeighbourRuleKind,
    TableIntegrity,
    TableTree,
    Teds,
    table_neighbour_rules,
    teds,
    tree_edit_distance,
)
from wholedoc_bench.scorer.utils.blocks import Grid
from wholedoc_bench.scorer.utils.tracks.t4_tables.table_matching import output_tables
from wholedoc_bench.scorer.utils.tracks.table_links import linked_tables, parse_truth_tables

TEDS = Teds()
ROW_MATCH = HeaderMatchedRows()
INTEGRITY = TableIntegrity()
NEIGHBOURS = CellNeighbourTests()
SUB_METRICS = (TEDS, ROW_MATCH, INTEGRITY, NEIGHBOURS)
ALL_AT_ONE = {metric.name: 1.0 for metric in SUB_METRICS}
PERFECT_FORMS = (perfect_pipe_markdown, perfect_html_markdown)
UNDER, LEFT, BELOW, NOT_BELOW = NeighbourRuleKind

Truth = Groundtruth | str
GridText = Rows | str | None
Table = tuple[Row | None, GridText]
Layout = list[LayoutItem] | Callable[[str], list[LayoutItem]]
CASE = ("truth", "markdown", "expected")


def sub_metric_id(value: object) -> str | None:
    return value.name if isinstance(value, SubMetric) else None


def cells(text: str) -> list[list[str]]:
    return [row.split("|") for row in text.split(" / ")]


def parsed(text: str) -> tuple[list[str], list[list[str]]]:
    header, *rows = cells(text)
    return header, rows


def rows_of(rows: GridText) -> Rows | None:
    return cells(rows) if isinstance(rows, str) else rows


def truth_of(truth: Truth) -> Groundtruth:
    return one_table(*parsed(truth)) if isinstance(truth, str) else truth


def pipe(text: str) -> str:
    return pipe_table(*parsed(text))


def html(text: str) -> str:
    return html_table(*parsed(text))


def as_markdown(output: str) -> str:
    is_table_text = bool(output) and "\n" not in output and not output.startswith("<")
    return pipe(output) if is_table_text else output


def scored(metric: SubMetric, truth: Truth, markdown: str, layout=(), relations=()) -> float:
    return metric.score(context(truth_of(truth), markdown, layout, relations))


def lone(header: Row | None, rows: GridText) -> Groundtruth:
    return groundtruth([table_element("t1", 0, header, rows_of(rows))])


def chained(header: Row | None, *pieces: GridText) -> Groundtruth:
    ids = "abcd"[: len(pieces)]
    elements = [
        table_element(piece_id, page, header if page == 0 else None, rows_of(rows))
        for page, (piece_id, rows) in enumerate(zip(ids, pieces, strict=True))
    ]
    return groundtruth(elements, [continues_in(a, b) for a, b in pairwise(ids)])


def pair(first: Table | str, second: Table | str, relation=stays_apart, page=1) -> Groundtruth:
    (h1, r1), (h2, r2) = (parsed(t) if isinstance(t, str) else t for t in (first, second))
    elements = [table_element("a", 0, h1, rows_of(r1)), table_element("b", page, h2, rows_of(r2))]
    return groundtruth(elements, [] if relation is None else [relation("a", "b")])


def joined(truth: Groundtruth, form: Callable[[Row | None, Rows], str] = pipe_table) -> str:
    tables = logical_tables(truth)
    return form(tables[0][0], [row for _, rows in tables for row in rows])


def apart_blocks(truth: Groundtruth) -> list[str]:
    return [pipe_table(header, rows) for header, rows in logical_tables(truth)]


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall)


METRIC = "Metric|Value / Revenue|10 / Cost|7"
METRIC_TABLE = pipe(METRIC)

FRUIT = ["Name", "Qty", "Price"]
FRUIT_FIRST = cells("apple|1|1.50 / pear|2|2.50")
FRUIT_SECOND = cells("plum|3|3.50 / fig|4|4.50")
FRUIT_ROWS = FRUIT_FIRST + FRUIT_SECOND
FRUIT_SPLIT = chained(FRUIT, FRUIT_FIRST, FRUIT_SECOND)
FRUIT_TABLE = pipe_table(FRUIT, FRUIT_ROWS)
NO_TABLE_OUTPUTS = [
    "",
    f"```\n{FRUIT_TABLE}\n```\n",
    pipe_table(["", "", ""], [["", "", ""], ["", "", ""]]),
    "\n".join("\t".join(row) for row in [FRUIT, *FRUIT_ROWS]),
]


DEMO_PREDICTIONS = {
    "html": lambda truth, raw: Prediction(markdown=perfect_html_markdown(truth)),
    "nutrient": lambda truth, raw: NutrientProvider().normalize(raw),
}


@pytest.mark.parametrize("predict", DEMO_PREDICTIONS.values(), ids=DEMO_PREDICTIONS)
def test_demo_predictions_score_one_on_every_sub_metric(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult, predict
) -> None:
    ctx = DocContext(demo_groundtruth, predict(demo_groundtruth, demo_nutrient_raw))
    assert len(only_table(ctx).body) == 40
    result = TRACK.evaluate(demo_groundtruth, ctx)
    assert (result.not_implemented, result.sub_scores) == ((), pytest.approx(ALL_AT_ONE))


def demo_table_ids(pred: Prediction) -> list[str]:
    return [item.id for item in pred.layout if item.cls is ElementClass.TABLE]


DEMO_BREAK = 37


def demo_output(truth: Groundtruth, name: str) -> str:
    [(header, rows)] = logical_tables(truth)
    first, rest = pipe_table(header, rows[:DEMO_BREAK]), rows[DEMO_BREAK:]
    outputs = {
        "data_row_as_header": [first, pipe_table(rest[0], rest[1:])],
        "headerless": [first, pipe_table(None, rest)],
    }
    return markdown_of(*outputs[name])


DEMO_SPLIT_TEDS = 1 - (3 + 3 * 3) / (1 + 41 + 123)


@pytest.mark.parametrize(
    ("metric", "output", "expected"),
    [
        (TEDS, "data_row_as_header", DEMO_SPLIT_TEDS),
        (ROW_MATCH, "data_row_as_header", 37 / 40),
        (NEIGHBOURS, "data_row_as_header", (3 * 2 + 2 * 3) / 24),
    ],
    ids=sub_metric_id,
)
def test_demo_table_output(
    demo_groundtruth: Groundtruth, metric: SubMetric, output: str, expected: float
) -> None:
    markdown = demo_output(demo_groundtruth, output)
    assert scored(metric, demo_groundtruth, markdown) == pytest.approx(expected)


@pytest.mark.parametrize(("keep_links", "expected"), [(True, 1.0), (False, 0.0)])
def test_demo_split_at_the_page_break_joins_only_through_the_nutrient_links(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult, keep_links: bool, expected: float
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    markdown = demo_output(demo_groundtruth, "headerless")
    relations = pred.relations if keep_links else []
    split = Prediction(markdown=markdown, layout=pred.layout, relations=relations)
    ctx = DocContext(demo_groundtruth, split)
    assert list(output_table_anchors(ctx)) == demo_table_ids(pred)
    assert INTEGRITY.score(ctx) == expected


@pytest.mark.parametrize("markdown", NO_TABLE_OUTPUTS)
def test_output_without_a_real_table_scores_zero_on_every_sub_metric(markdown: str) -> None:
    assert [scored(metric, FRUIT_SPLIT, markdown) for metric in SUB_METRICS] == [0.0] * 4


@pytest.mark.parametrize("by_text", [False, True])
def test_output_table_the_layout_marks_as_a_chart_scores_zero_on_every_sub_metric(by_text):
    markdown = markdown_of(FRUIT_TABLE)
    text = " ".join(cell for row in [FRUIT, *FRUIT_ROWS] for cell in row)
    by_span = layout_over("c", ElementClass.CHART, markdown, FRUIT_TABLE)
    chart = layout_item("c", ElementClass.CHART, text=text) if by_text else by_span
    assert [scored(metric, FRUIT_SPLIT, markdown) for metric in SUB_METRICS] == [1.0] * 4
    assert [scored(m, FRUIT_SPLIT, markdown, [chart]) for m in SUB_METRICS] == [0.0] * 4


NAME_QTY = ["Name", "Qty"]
APPLE_PEAR = cells("apple|1 / pear|2")
PAGE_ONE = "Name|Qty / apple|1 / pear|2"
MIXED = groundtruth(
    [*split_truth().elements, table_element("c", 1, None, cells("x1|y1 / x2|y2"))],
    [continues_in("a", "b")],
)
PEN = "Item|Cost / pen|5 / ink|6"
NORTH = "Region|Sales / north|70 / south|80"
APART = pair(PEN, NORTH)
SPLIT_THEN_APART = groundtruth(
    [*FRUIT_SPLIT.elements, table_element("c", 2, *parsed(NORTH))],
    [continues_in("a", "b"), stays_apart("b", "c")],
)
KV_PAGES = chained(["K", "V"], "a1|x1 / a2|x2", "b1|y1", "c1|z1 / c2|z2")
TOTAL_ABOVE = pair("Item|Cost / pen|5 / Total|Sales", NORTH)

S_HEADER, S_ROWS = grid("s", 6, 2)
S_FIRST, S_SECOND = S_ROWS[:4], S_ROWS[4:]
X_HEADER, X_ROWS = grid("x", 3, 2)
Y_ROWS = grid("y", 3, 2)[1]
Z_HEADER, Z_ROWS = grid("z", 4, 2)
S_ONE = pipe_table(S_HEADER, S_ROWS)
S_PAGES = [pipe_table(S_HEADER, S_FIRST), pipe_table(None, S_SECOND)]
S_THIRDS = [pipe_table(S_HEADER, S_ROWS[:2]), pipe_table(None, S_ROWS[2:4])]
S_THIRDS += [pipe_table(None, S_ROWS[4:])]
X_ONE, Y_ONE = pipe_table(X_HEADER, X_ROWS), pipe_table(None, Y_ROWS)
Z_ONE = pipe_table(Z_HEADER, Z_ROWS)
XY_JOINED = pipe_table(X_HEADER, X_ROWS + Y_ROWS)
EXTRA = pipe("Colour|Code / red|r1 / blue|b2")
S_SPLIT = chained(S_HEADER, S_FIRST, S_SECOND)
S_THREE_PAGES = chained(S_HEADER, S_ROWS[:2], S_ROWS[2:4], S_ROWS[4:])
XY_APART = pair((X_HEADER, X_ROWS), (None, Y_ROWS))
XZ = pair((X_HEADER, X_ROWS), (Z_HEADER, Z_ROWS), None, 0)
Z_TABLE = one_table(Z_HEADER, Z_ROWS)
XY_ELEMENTS = [table_element("x", 1, X_HEADER, X_ROWS), table_element("y", 2, None, Y_ROWS)]
REVIEW_RELATIONS = [*S_SPLIT.relations, stays_apart("x", "y")]
REVIEW_SIX = groundtruth([*S_SPLIT.elements, *XY_ELEMENTS], REVIEW_RELATIONS)
Z_PAGE_THREE = table_element("z", 3, Z_HEADER, Z_ROWS)
FOUR_TABLES = groundtruth([*REVIEW_SIX.elements, Z_PAGE_THREE], REVIEW_RELATIONS)
REVIEW_BLOCKS = [*S_PAGES, X_ONE, Y_ONE]

LABELS_X = "Item|2023 / Rent|10 / Staff|20 / Total|30"
LABELS_Y = "Rent|11 / Staff|21 / Total|32"
SAME_LABELS = {
    "stays_apart": pair(LABELS_X, f"Item|2023 / {LABELS_Y}"),
    "three_columns": pair(
        "Item|2023|2022 / Rent|10|9 / Staff|20|19 / Total|30|28",
        "Item|2023|2022 / Rent|10|7 / Staff|20|17 / Total|30|24",
    ),
}
SHARED_ROWS = {
    "two_of_four": pair("Item|2023 / a|1 / b|2 / c|3 / d|4", "Item|2023 / a|1 / b|2 / c|5 / d|6"),
}
SUBSET = "Item|2023 / Rent|10 / Staff|20"
SUPERSET = f"{SUBSET} / Total|30 / Tax|5"
SUBSET_FIRST = pair(SUBSET, SUPERSET)
SUBSET_FIRST_SPLIT = groundtruth(
    [
        table_element("y", 0, *parsed(SUBSET)),
        table_element("a", 1, *parsed(SUBSET)),
        table_element("b", 2, None, cells("Total|30 / Tax|5")),
    ],
    [stays_apart("y", "a"), continues_in("a", "b")],
)
BLANK_ROWS = [["", ""], [" ", "\t"], ["\n", ""]]
PERFECT_TRUTHS = [MIXED, FRUIT_SPLIT, SPLIT_THEN_APART, KV_PAGES, TOTAL_ABOVE, FOUR_TABLES]
PERFECT_TRUTHS += [SAME_LABELS["three_columns"], *SHARED_ROWS.values()]


@pytest.mark.parametrize("truth", PERFECT_TRUTHS)
@pytest.mark.parametrize("perfect", PERFECT_FORMS)
def test_groundtruth_as_a_perfect_prediction_scores_one_where_a_sub_metric_applies(
    truth: Groundtruth, perfect: Callable[[Groundtruth], str]
) -> None:
    for metric in SUB_METRICS:
        if metric.applies(truth):
            assert scored(metric, truth, perfect(truth)) == pytest.approx(1.0), metric.name


NO_TABLES = groundtruth([para("p1", "text"), chart_element("c1", CHART_TEXT)])
BR_TABLE = lone(None, [["<br>"]])
ROW = "Name|Qty / apple|1 / pear|2 / plum|3"
NO_TEXT_HEADERS = [(None, APPLE_PEAR), (["", ""], APPLE_PEAR)]
NO_COUNTED_ROWS = [(NAME_QTY, []), (NAME_QTY, [["", ""], ["", ""]])]
BLANK_AND_BODY = pair((X_HEADER, BLANK_ROWS), (Z_HEADER, Z_ROWS), None, 0)
NO_BODY_ROWS = [NO_TABLES, lone(Z_HEADER, []), lone(X_HEADER, BLANK_ROWS)]


@pytest.mark.parametrize(
    ("metric", "truth", "expected"),
    [
        (TEDS, lone(HEADER, []), True),
        *[(TEDS, truth, False) for truth in [NO_TABLES, lone([], [])]],
        (ROW_MATCH, lone(["Name", ""], "|9"), True),
        *[(ROW_MATCH, lone(h, r), False) for h, r in NO_TEXT_HEADERS + NO_COUNTED_ROWS],
        (ROW_MATCH, NO_TABLES, False),
        *[(INTEGRITY, t, True) for t in [chained(S_HEADER, None, S_ROWS), BLANK_AND_BODY]],
        *[(INTEGRITY, truth, False) for truth in NO_BODY_ROWS],
        (INTEGRITY, chained(X_HEADER, BLANK_ROWS[:1], BLANK_ROWS), False),
        (INTEGRITY, BR_TABLE, True),
        *[(NEIGHBOURS, truth, True) for truth in [APART, FRUIT_SPLIT]],
        (NEIGHBOURS, lone(FRUIT, FRUIT_ROWS), False),
    ],
    ids=sub_metric_id,
)
def test_sub_metric_applies_only_when_the_groundtruth_has_something_to_score(
    metric: SubMetric, truth: Groundtruth, expected: bool
) -> None:
    assert metric.applies(truth) is expected


NODES = 1 + 3 + 6
PROFIT = f"{METRIC} / Profit|3"
TWO_PERFECT = perfect_pipe_markdown(two_different_tables())
ROWSPAN = (
    "<table><tr><th>Region</th><th>Q1</th><th>Q2</th></tr><tr><td rowspan='2'>North</td>"
    "<td>10</td><td>20</td></tr><tr><td>30</td><td>40</td></tr></table>"
)
ESCAPED_PIPE = "| **Term** | Value |\n|---|---|\n| a \\| b | **Total** |\n| c | 7 |\n"
TERMS = one_table(["Term", "Value"], [["a | b", "Total"], ["c", "7"]])
TWO = two_different_tables()
SPLIT_AS_TWO = markdown_of(pipe(PAGE_ONE), pipe("plum|3 / fig|4"))
DOUBLED = "Metric|Value / Revenue|10 / Revenue|10 / Cost|7 / Cost|7"
PROSE = "Use the `<table>` tag for tables."
HTML_PROSE = "In HTML, <table> starts a table."
TAGGED = "Label|Value / <b>x</b>|1"
PROSE_CASES = [(PROSE, METRIC_TABLE), (HTML_PROSE, html(METRIC))]
TEDS_CASES = {
    **{f"prose_{i}": (METRIC, markdown_of(*case), 1.0) for i, case in enumerate(PROSE_CASES)},
    "split_as_two_tables": (split_truth(), SPLIT_AS_TWO, 5 / 8),
    "one_cell_differs": (METRIC, "Metric|Value / Revenu|10 / Cost|7", 1 - (1 / 7) / NODES),
    "one_row_missing": (METRIC, "Metric|Value / Revenue|10", 1 - 3 / NODES),
    "extra_column": (METRIC, "Metric|Value|Note / Revenue|10|up / Cost|7|down", 1 - 3 / 13),
    "numbers": ("Q|R / a|$18m / b|1,234.5", "Q|R / a|18m / b|1234.5", 1 - (1 / 4 + 1 / 7) / NODES),
    "case_is_kept": ("Alpha / Beta", "ALPHA / BETA", 1 - (4 / 5 + 3 / 4) / 5),
    "headerless_truth": (one_table(None, cells("a1|b1 / a2|b2")), "a1|b1 / a2|b2", 1.0),
    "header_as_body_row": (METRIC, html_table(None, cells(METRIC)), 1.0),
    "printed_stars": (
        "**M**|V / Net   income |10 / Cost|7",
        "M|V / Net income|10 / Cost|7",
        1 - (4 / 5) / NODES,
    ),
    "second_table_missing": (TWO, PAGE_ONE, 0.5),
    "escaped_pipe_and_bold": (TERMS, ESCAPED_PIPE, 1.0),
    "perfect_twice": (TWO, markdown_of(TWO_PERFECT, TWO_PERFECT), 1.0),
    "stays_apart_joined": (TWO, joined(TWO), (1 - 6 / 16) / 2),
    "every_row_twice": (METRIC, DOUBLED, 1 - 6 / 16),
    "header_only": (PROFIT, "Metric|Value", 1 - 9 / 13),
    "cell_as_row": (lone(None, "x|x|y"), html_table(None, cells("x / y / x")), 1 / 7),
    "printed_tag_in_code": (TAGGED, "Label|Value / `<b>x</b>`|1", 1.0),
    "printed_tag_dropped": (TAGGED, "Label|Value / x|1", 1 - (7 / 8) / 7),
}


@pytest.mark.parametrize(CASE, TEDS_CASES.values(), ids=TEDS_CASES)
def test_teds(truth: Truth, markdown: str, expected: float) -> None:
    assert scored(TEDS, truth, as_markdown(markdown)) == pytest.approx(expected)


MERGED_NORTH = groundtruth(
    [
        Element.model_validate(
            {
                "id": "t",
                "class": "table",
                "page": 0,
                "header": ["Region", "Q1", "Q2"],
                "rows": [["North", "10", "20"], [None, "30", "40"]],
                "spans": [{"row": 1, "column": 0, "rows": 2}],
            }
        )
    ]
)
REPEATED_NORTH = "Region|Q1|Q2 / North|10|20 / North|30|40"


@pytest.mark.parametrize(
    ("truth", "markdown", "expected"),
    [
        (MERGED_NORTH, ROWSPAN, 1.0),
        (MERGED_NORTH, REPEATED_NORTH, 1 - 2 / 13),
        (REPEATED_NORTH, ROWSPAN, 1 - 2 / 13),
        (REPEATED_NORTH, REPEATED_NORTH, 1.0),
    ],
    ids=["same span", "merged written twice", "two cells written merged", "no spans"],
)
def test_teds_compares_merged_cells_with_their_spans(
    truth: Truth, markdown: str, expected: float
) -> None:
    assert scored(TEDS, truth, as_markdown(markdown)) == pytest.approx(expected)


def test_a_span_on_a_later_piece_of_a_split_table_moves_with_its_rows() -> None:
    head = {
        "id": "a",
        "class": "table",
        "page": 0,
        "header": ["Region", "Q1"],
        "rows": [["South", "5"]],
    }
    tail = {
        "id": "b",
        "class": "table",
        "page": 1,
        "rows": [["North", "10"], [None, "30"]],
        "spans": [{"row": 0, "column": 0, "rows": 2}],
    }
    split = truth(head, tail, relations=[{"kind": "continues_in", "from": "a", "to": "b"}])
    [table] = parse_truth_tables(split)
    assert table.merged == (CellSpan(row=2, column=0, rows=2),)


def chart_and_table(chart_text: str, chart_first: bool, table: str) -> Groundtruth:
    elements = [table_element("t1", 0, *parsed(table)), chart_element("c1", chart_text)]
    return groundtruth(elements[::-1] if chart_first else elements)


CHART = "Quarter|Revenue / Q1|10 / Q2|20"
COSTS = "Quarter|Costs / Q1|5 / Q2|7"
TWIN_CHARTS = markdown_of(chart_pipe(), chart_pipe())
OTHER_NUMBERS = markdown_of(pipe(COSTS), chart_pipe(), pipe("Quarter|Revenue / Q1|30 / Q2|40"))
INEXACT_TEXT = "Revenue Q1 10 Q2 20"
CHART_CASES = {
    "twin_table_first": (chart_and_table(CHART_TEXT, False, CHART), TWIN_CHARTS, [0]),
    "twin_chart_first": (chart_and_table(CHART_TEXT, True, CHART), TWIN_CHARTS, [1]),
    "twin_inexact_chart_first": (chart_and_table(INEXACT_TEXT, True, CHART), TWIN_CHARTS, [0]),
    "other_numbers_chart_first": (chart_and_table(CHART_TEXT, True, COSTS), OTHER_NUMBERS, [0]),
}


@pytest.mark.parametrize(("truth", "markdown", "tables"), CHART_CASES.values(), ids=CHART_CASES)
def test_teds_scores_only_the_output_tables_that_are_not_charts(truth, markdown, tables) -> None:
    ctx = context(truth, markdown)
    charts = [index for index in range(len(linked_tables(ctx))) if index not in tables]
    assert indexes(output_tables(linked_tables(ctx))) == tables
    assert indexes(chart_tables(ctx)) == charts
    assert TEDS.score(ctx) == pytest.approx(1.0)


KIWI_TEXT = "plum|3 / fig|4 / kiwi|5"
KIWI = cells(KIWI_TEXT)
APPLE_ONE = "Name|Qty / apple|1"
APPLE_TWO = "Name|Qty / apple|2"
KIWI_TWO = "Name|Qty / plum|3 / fig|4"
ROW_SPLIT = chained(NAME_QTY, APPLE_PEAR, KIWI)
WIDE = "ID|Name|Qty / 1|apple|5 / 2|pear|6 / 3|plum|7"
LABEL = "|2023|2024 / Revenue|5|6 / Cost|7|8 / Tax|1|2"
TWO_LABEL = "||2023 / EU|Revenue|5 / US|Cost|7"
YEARS = "Year|Value|Value / 2020|1|2"
PAST = "Name|Qty / apple|1|extra / pear|2"
PAST_EMPTY = "Name| / apple|x|extra / pear|y|more"
DUPLICATE = "Name|Qty / apple|1 / apple|1 / pear|2"
LOOSE = "Name|Qty / apple| / apple|1"
DIAGONAL = "A|B|C / a|| / |b| / ||c"
PEAR_20 = "Name|Qty / apple|1 / pear|20 / plum|3"
COLSPAN = '<tr><th>Item</th><th colspan="2">Amount</th></tr>'
PEN_ROW = html_row(["pen", "5", "6"], "td")
YEAR_THEAD = (
    f"<table><thead>{COLSPAN}{html_row(['', '2023', '2024'], 'th')}</thead>"
    f"<tbody>{PEN_ROW}</tbody></table>"
)
FRUIT_PAGES = [table_element("a", 0, *parsed(PAGE_ONE)), table_element("b", 1, *parsed(KIWI_TWO))]
REVENUE_HEADER = {
    "id": "a",
    "class": "table",
    "page": 0,
    "header": [["Region", "Revenue", None], [None, "2025", "2026"]],
    "rows": [["North", "100", "120"], ["South", "90", "95"]],
    "spans": [{"row": 0, "column": 0, "rows": 2}, {"row": 0, "column": 1, "columns": 2}],
}
REVENUE = truth(REVENUE_HEADER)
REVENUE_HEADER_ROWS = [["Region", "Revenue", "Revenue"], ["Region", "2025", "2026"]]
REVENUE_ROWS = [["North", "100", "120"], ["South", "90", "95"]]
REVENUE_BODY = "North|100|120 / South|90|95"
REVENUE_CASES = {
    "two_header_rows": header_rows_table(REVENUE_HEADER_ROWS, REVENUE_ROWS),
    "lower_header_row_as_data": "Region|Revenue|Revenue / |2025|2026 / North|100|120 / South|90|95",
    "upper_header_row_only": "Region|Revenue|Revenue / North|100|120 / South|90|95",
    "lower_header_row_only": "Region|2025|2026 / North|100|120 / South|90|95",
    "year_columns_swapped": "Region|2026|2025 / North|120|100 / South|95|90",
    "labels_joined_in_one_cell": "Region|Revenue 2025|Revenue 2026 / North|100|120 / South|90|95",
    "labels_joined_with_punctuation": f"Region|Revenue (2025)|Revenue (2026) / {REVENUE_BODY}",
}


@pytest.mark.parametrize("markdown", REVENUE_CASES.values(), ids=REVENUE_CASES)
def test_one_label_from_any_header_row_finds_the_column(markdown: str) -> None:
    assert scored(ROW_MATCH, REVENUE, as_markdown(markdown)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "markdown",
    [
        "Region|2025|2026 / North|120|100 / South|95|90",
        "Region|2025 Revenue|2026 Revenue / North|100|120 / South|90|95",
        "Region|Revenue 2026|Revenue 2025 / North|100|120 / South|90|95",
        "Region|Revenue| / |2025|2026 / North|100|120 / South|90|95",
    ],
    ids=[
        "values_under_the_wrong_year",
        "joined_labels_out_of_order",
        "joined_labels_over_the_wrong_values",
        "covered_header_cell_blank",
    ],
)
def test_a_column_with_no_label_of_the_groundtruth_column_misses_its_rows(markdown: str) -> None:
    assert scored(ROW_MATCH, REVENUE, as_markdown(markdown)) == 0.0


TOTALS = truth(
    {
        "id": "t",
        "class": "table",
        "page": 0,
        "header": [["", "", "2025"], ["Product", "Total", "Total"]],
        "rows": [["Pens", "300", "100"]],
    }
)
TOTALS_MOVED_HEADER = [["", "2025", ""], ["Product", "Total", "Total"]]


@pytest.mark.parametrize(
    ("values", "expected"),
    [(["Pens", "100", "300"], 1.0), (["Pens", "300", "100"], 0.0)],
    ids=["column_moved_with_its_values", "headings_moved_without_values"],
)
def test_the_column_sharing_more_labels_is_matched_first(
    values: list[str], expected: float
) -> None:
    markdown = header_rows_table(TOTALS_MOVED_HEADER, [values])
    assert scored(ROW_MATCH, TOTALS, markdown) == expected


REVENUE_TAIL = {**REVENUE_HEADER, "id": "b", "page": 1, "rows": [["East", "80", "85"]]}
REVENUE_SPLIT = truth(
    REVENUE_HEADER, REVENUE_TAIL, relations=[{"kind": "continues_in", "from": "a", "to": "b"}]
)
REVENUE_ALL_ROWS = [*REVENUE_ROWS, ["East", "80", "85"]]


def test_a_split_table_keeps_every_header_row_and_drops_the_repeated_ones() -> None:
    [table] = parse_truth_tables(REVENUE_SPLIT)
    assert table.header == (("Region", "Revenue", "Revenue"), ("Region", "2025", "2026"))
    assert table.rows == tuple(map(tuple, REVENUE_ALL_ROWS))
    assert table.merged == (CellSpan(row=0, column=0, rows=2), CellSpan(row=0, column=1, columns=2))


def test_a_cell_under_a_two_row_header_passes_with_or_without_the_lower_row() -> None:
    rule = NeighbourRule(NeighbourRuleKind.UNDER_HEADER, "80", "Revenue", ("2025",))
    assert rule in table_neighbour_rules(REVENUE_SPLIT)
    lower_row_as_data = pipe_table(
        REVENUE_HEADER_ROWS[0], [["", "2025", "2026"], *REVENUE_ALL_ROWS]
    )
    joined = pipe_table(["Region", "Revenue 2025", "Revenue 2026"], REVENUE_ALL_ROWS)
    for markdown in (
        header_rows_table(REVENUE_HEADER_ROWS, REVENUE_ALL_ROWS),
        lower_row_as_data,
        joined,
    ):
        assert scored(NEIGHBOURS, REVENUE_SPLIT, markdown) == pytest.approx(1.0)


RESULTS_HEAD = {
    "id": "a",
    "class": "table",
    "page": 0,
    "header": [["Results", None], ["Name", "Score"]],
    "rows": [["alice", "10"]],
    "spans": [{"row": 0, "column": 0, "columns": 2}],
}
RESULTS_TAIL = {"id": "b", "class": "table", "page": 1, "rows": [["bob", "20"]]}
RESULTS_SPLIT = truth(
    RESULTS_HEAD, RESULTS_TAIL, relations=[{"kind": "continues_in", "from": "a", "to": "b"}]
)
RESULTS_HEADER_ROWS = [["Results", "Results"], ["Name", "Score"]]


@pytest.mark.parametrize(
    ("header_rows", "rows", "expected"),
    [
        (RESULTS_HEADER_ROWS, [["alice", "10"], ["bob", "20"]], 1.0),
        (RESULTS_HEADER_ROWS, [["alice", "10"], ["20", "bob"]], 0.0),
        (RESULTS_HEADER_ROWS[:1], [["alice", "10"], ["bob", "20"]], 1.0),
        ([["Results Name", "Results Score"]], [["alice", "10"], ["20", "bob"]], 0.0),
    ],
    ids=[
        "right_columns",
        "row_swapped_under_shared_label",
        "lower_header_row_left_out",
        "row_swapped_under_joined_labels",
    ],
)
def test_a_cell_is_under_the_lowest_header_label_the_output_has(
    header_rows: list[list[str]], rows: list[list[str]], expected: float
) -> None:
    markdown = header_rows_table(header_rows, rows)
    assert scored(NEIGHBOURS, RESULTS_SPLIT, markdown) == pytest.approx(expected)


ROW_MATCH_CASES = {
    "data_row_as_header": (ROW_SPLIT, markdown_of(pipe(PAGE_ONE), pipe(KIWI_TEXT)), 2 / 5),
    "header_repeated": (ROW_SPLIT, markdown_of(pipe(PAGE_ONE), pipe_table(NAME_QTY, KIWI)), 1.0),
    "columns_reordered": (WIDE, "Qty|ID|Name / 5|1|apple / 6|2|pear / 7|3|plum", 1.0),
    "row_moved_right_html": (WIDE, html("ID|Name|Qty / 1|apple|5 / |2|pear|6 / 3|plum|7"), 2 / 3),
    "short_row": (WIDE, html("ID|Name|Qty / 1|apple|5 / 2|pear / 3|plum|7"), 2 / 3),
    "first_row_as_header": (ROW, html_table(None, cells(ROW)), 1.0),
    "header_printed_once": (
        "Name|Qty / Name|Qty / pear|2",
        html_table(None, cells("Name|Qty / pear|2")),
        0.5,
    ),
    "colspan_header": ("Item|Amount|Amount / pen|5|6", f"<table>{COLSPAN}{PEN_ROW}</table>", 1.0),
    "second_header_row": ("Item|2023|2024 / pen|5|6", YEAR_THEAD, 1.0),
    "colspan_over_row": ("Item|Amount / pen|5", YEAR_THEAD, 1.0),
    "kth_swapped": (YEARS, "Year|Value|Value / 2020|2|1", 0.0),
    "kth_moved": (YEARS, "Value|Year|Value / 1|2020|2", 1.0),
    "no_kth_column": (YEARS, "Year|Value|Other / 2020|1|2", 0.0),
    "casefold": ("S|Q / Straße|1 / Main  Road|2", "**S**|<b>q</b> / STRASSE|1 / main road|2", 1),
    "numbers": ("I|A / a|$18m / b|1,234.5 / c|18.2%", "I|A / a|18m / b|1234.5 / c|18.2%", 1 / 3),
    "empty_truth_cell": ("Name|Qty / apple| / pear|2", "Name|Qty / apple|9 / pear|2", 1.0),
    "empty_header_wrong": ("Name| / apple|x / pear|y", "Name|Other / apple|z / pear|y", 1 / 2),
    "empty_header_row_missing": ("Name| / apple|1 / |9", "Name| / apple|1", 1 / 2),
    "named_label_column": (LABEL, html(f"Segment{LABEL}"), 1.0),
    "label_column_moved": (LABEL, "2023|2024|Segment / 5|6|Revenue / 7|8|Cost / 1|2|Tax", 1.0),
    "label_column_dropped": (LABEL, "2023|2024 / 5|6 / 7|8 / 1|2", 0.0),
    "two_labels_last": (TWO_LABEL, "2023|Region|Line / 5|EU|Revenue / 7|US|Cost", 1.0),
    "two_labels_swapped": (TWO_LABEL, "Line|Region|2023 / Revenue|EU|5 / Cost|US|7", 0.0),
    "past_width_copied": (PAST, html(PAST), 1.0),
    "past_width_wrong": (PAST, "Name|Qty|Notes / apple|1|other / pear|2|", 1 / 2),
    "past_width_moved": (PAST_EMPTY, "Code|Notes|Name / x|extra|apple / y|more|pear", 1.0),
    "empty_truth_row": ("Name|Qty / apple|1 / |", "Name|Qty / apple|1", 1.0),
    "uncounted_header": (pair(APPLE_ONE, "| / x1|y1", None, 0), APPLE_ONE, 1.0),
    "duplicate_rows_once": (DUPLICATE, PAGE_ONE, 2 / 3),
    "matching_beats_first_fit": ("A|B / x| / |y", "A|B / x|y / x|z", 1.0),
    "matching_moves_a_row": (DIAGONAL, "A|B|C / a|b|c / a|0|0 / 0|b|0", 1.0),
    "matching_chain": (DIAGONAL, "A|B|C / a|b|0 / 0|b|c / a|0|0", 1.0),
    "output_row_used_once": ("A|B / x| / |y / x|y", "A|B / x|y / x|z", 2 / 3),
    "output_order": (LOOSE, markdown_of(pipe(APPLE_ONE), pipe(APPLE_TWO)), 1.0),
    "two_truths_one_output": (groundtruth(FRUIT_PAGES), f"{PAGE_ONE} / plum|3 / fig|4", 1.0),
    "rows_across_tables": (ROW, markdown_of(pipe("Name|Qty / plum|3"), html(PAGE_ONE)), 1.0),
    "wrong_row": (ROW, PEAR_20, 2 / 3),
    "table_twice": (ROW, markdown_of(pipe(PEAR_20), pipe(PEAR_20)), 2 / 3),
    "header_only": (ROW, html("Name|Qty"), 0.0),
    "transposed": (ROW, "Name|apple|pear|plum / Qty|1|2|3", 0.0),
}


@pytest.mark.parametrize(CASE, ROW_MATCH_CASES.values(), ids=ROW_MATCH_CASES)
def test_header_matched_rows(truth: Truth, markdown: str, expected: float) -> None:
    assert scored(ROW_MATCH, truth, as_markdown(markdown)) == pytest.approx(expected)


def changed_cells(columns: int, equal: int) -> tuple[Groundtruth, str]:
    header, rows = grid("w", 2, columns)
    changed = [[*row[:equal], *(f"{cell}!" for cell in row[equal:])] for row in rows]
    return one_table(header, rows), pipe_table(header, changed)


TIE = pair("Item|Amount / a|1 / b|2", "Item|Amount / a|9 / c|3", None, 0)
TIE_BOTH, TIE_X, TIE_Y = (pipe(f"Item|Amount / {row}") for row in ["a|5", "b|5", "a|9"])
H_ROWS = grid("h", 3, 2)[1]
H_TABLE = one_table(None, H_ROWS)
HEADER_ONLY = pair((X_HEADER, None), (Z_HEADER, Z_ROWS), None, 0)
REPEATED = "k|v / r|1 / r|1 / s|2 / t|3"
OWNERS = pair("Section|Owner / Summary|Anna / Revenue|Ben", "Revenue|2023|2024 / Asia|7|9")
HALF_CELLS = [(3, 1, 0.0), (3, 2, 1.0), (4, 2, 1.0)]
PIECE_SHARES = [(S_FIRST, S_SECOND[:1], 1.0), (S_FIRST[:1], S_SECOND, 0.0)]
PIECE_SHARE_CASES = {
    f"piece_share_{i}": (S_SPLIT, pipe_table(S_HEADER, [*first, *second]), expected)
    for i, (first, second, expected) in enumerate(PIECE_SHARES)
}
NORMALIZED = [("Straße", "STRASSE"), ("net income", "*Net  Income*")]
GIANT_TABLE_TRUTHS = [XY_APART, FOUR_TABLES]
Z_PLUS_X = [pipe_table(Z_HEADER, Z_ROWS + X_ROWS[:n]) for n in (2, 1)]
JOINED_FIRST = pipe(f"{SUPERSET} / Rent|10 / Staff|20")
INTEGRITY_CASES = {
    "split_without_layout": (S_SPLIT, markdown_of(*S_PAGES), 0.0),
    "header_row_as_unit_row": (H_TABLE, pipe_table(H_ROWS[0], H_ROWS[1:2]), 1.0),
    "truth_header_not_needed": (Z_TABLE, html_table(None, Z_ROWS), 1.0),
    **PIECE_SHARE_CASES,
    **{t: (f"L / {t}", f"L / {o}", 1.0) for t, o in NORMALIZED},
    **{f"{e}_of_{c}_cells": (*changed_cells(c, e), x) for c, e, x in HALF_CELLS},
    "cells_swapped": ("k|v / p|q / r|s", "k|v / q|p / s|r", 0.0),
    "empty_cells_uncounted": ("k|v|w / p|| / r||", "k|v|w / p|x|y / r|x|y", 1.0),
    "blank_rows_uncounted": ("k|v / | / | / p|q", "k|v / p|q", 1.0),
    "unrelated_rows_uncharged": ("k|v / p|q", "k|v / p|q / u1|w1 / u2|w2 / u3|w3", 1.0),
    "output_row_used_once": ("k|v / p|1 / p|2 / p|3 / q|4", "k|v / p|1", 0.0),
    "best_fit_first": ("k|v / a|1 / b|2 / c|3 / d|4", "k|v / a|2 / a|1", 1.0),
    "repeated_row_per_copy": (REPEATED, "k|v / r|1 / r|1", 1.0),
    "piece_without_rows": (chained(S_HEADER, None, S_ROWS), S_ONE, 1.0),
    **{f"same_labels_{n}": (t, markdown_of(*apart_blocks(t)), 1.0) for n, t in SAME_LABELS.items()},
    **{f"same_labels_{n}_joined": (t, joined(t), 0.0) for n, t in SAME_LABELS.items()},
    **{f"shared_rows_{n}": (t, joined(t), 0.0) for n, t in SHARED_ROWS.items()},
    "tie_in_markdown_order": (TIE, markdown_of(TIE_BOTH, TIE_X), f1(1 / 2, 1 / 2)),
    "tie_reversed": (TIE, markdown_of(TIE_X, TIE_BOTH), 1.0),
    "best_unit_first": (TIE, markdown_of(TIE_Y, pipe("Item|Amount / a|1 / b|2")), 1.0),
    "subset_dropped": (SUBSET_FIRST, SUPERSET, f1(1, 1 / 2)),
    "superset_first": (SUBSET_FIRST, markdown_of(pipe(SUPERSET), pipe(SUBSET)), 1.0),
    "joined_first": (SUBSET_FIRST, markdown_of(JOINED_FIRST, pipe(SUPERSET), pipe(SUBSET)), 0.8),
    "half_of_another": (XZ, markdown_of(Z_PLUS_X[0], X_ONE), f1(1 / 2, 1 / 2)),
    "under_half_of_another": (XZ, markdown_of(Z_PLUS_X[1], X_ONE), 1.0),
    "table_twice": (S_SPLIT, markdown_of(S_ONE, "Again:", S_ONE), f1(1 / 2, 1)),
    "mixed": (FOUR_TABLES, markdown_of(*S_PAGES, XY_JOINED, Z_ONE, EXTRA), f1(1 / 5, 1 / 4)),
    "join_every_break": (REVIEW_SIX, markdown_of(S_ONE, XY_JOINED), f1(1 / 2, 1 / 3)),
    "join_only_continuations": (REVIEW_SIX, markdown_of(S_ONE, X_ONE, Y_ONE), 1.0),
    "split_every_break": (REVIEW_SIX, markdown_of(*REVIEW_BLOCKS), f1(2 / 4, 2 / 3)),
    **{f"giant_table_{i}": (t, joined(t), 0.0) for i, t in enumerate(GIANT_TABLE_TRUTHS)},
    "header_only_unit": (HEADER_ONLY, markdown_of(pipe_table(X_HEADER, []), Z_ONE), f1(1 / 2, 1)),
    "header_only_inside": (HEADER_ONLY, pipe_table(Z_HEADER, [X_HEADER, *Z_ROWS]), 1.0),
    "blank_truth": (BLANK_AND_BODY, Z_ONE, 1.0),
    "blank_unit": (BLANK_AND_BODY, markdown_of(pipe_table(X_HEADER, BLANK_ROWS), Z_ONE), 2 / 3),
    "own_header_like_a_small_table_row": (OWNERS, perfect_pipe_markdown(OWNERS), 1.0),
    "data_row_as_other_header": (pair("k|v / p|q", NORTH), "p|q / north|70 / south|80", 0.0),
    "printed_br": (BR_TABLE, "| |\n|---|\n| `<br>` |", 1.0),
}


@pytest.mark.parametrize(CASE, INTEGRITY_CASES.values(), ids=INTEGRITY_CASES)
def test_table_integrity(truth: Truth, markdown: str, expected: float) -> None:
    assert scored(INTEGRITY, truth, as_markdown(markdown)) == pytest.approx(expected)


def table_items(markdown: str, blocks: Sequence[str]) -> list[LayoutItem]:
    return [layout_over(f"item{i}", ElementClass.TABLE, markdown, b) for i, b in enumerate(blocks)]


S_AROUND_Z = [S_SPLIT.elements[0], table_element("z", 0, Z_HEADER, Z_ROWS), S_SPLIT.elements[1]]
ACROSS_A_TABLE = groundtruth(S_AROUND_Z, S_SPLIT.relations)
S_FIVE_AND_ONE = chained(S_HEADER, S_ROWS[:5], S_ROWS[5:])
FIRST_ROW_AS_HEADER = [pipe_table(S_HEADER, S_ROWS[:5]), pipe_table(S_ROWS[5], [])]
SPLIT_SUPERSET = [pipe(SUBSET), pipe_table(None, cells("Total|30 / Tax|5")), pipe(SUBSET)]
LINKED_CASES = {
    "split_linked": (S_SPLIT, S_PAGES, [(0, 1)], 1.0),
    "chain_of_three": (S_THREE_PAGES, S_THIRDS, [(0, 1), (1, 2)], 1.0),
    "chain_missing_a_link": (S_THREE_PAGES, S_THIRDS, [(0, 1)], 0.0),
    "across_a_table": (ACROSS_A_TABLE, [S_PAGES[0], Z_ONE, S_PAGES[1]], [(0, 2)], 1.0),
    "first_row_as_header": (S_FIVE_AND_ONE, FIRST_ROW_AS_HEADER, [(0, 1)], 1.0),
    "stays_apart_linked": (XY_APART, [X_ONE, Y_ONE], [(0, 1)], 0.0),
    "split_superset_linked": (SUBSET_FIRST_SPLIT, SPLIT_SUPERSET, [(0, 1)], 1.0),
    "honest_links": (REVIEW_SIX, REVIEW_BLOCKS, [(0, 1)], 1.0),
    "link_every_break": (REVIEW_SIX, REVIEW_BLOCKS, [(0, 1), (2, 3)], f1(1 / 2, 1 / 3)),
    "one_chain": (REVIEW_SIX, REVIEW_BLOCKS, [(0, 1), (1, 2), (2, 3)], 0.0),
    "backward_link": (S_SPLIT, S_PAGES, [(1, 0)], 0.0),
    "backward_link_in_chain": (S_THREE_PAGES, S_THIRDS, [(0, 1), (2, 1)], 0.0),
}


LINK_ARGS = ("truth", "blocks", "links", "expected")


@pytest.mark.parametrize(LINK_ARGS, LINKED_CASES.values(), ids=LINKED_CASES)
def test_table_integrity_joins_tables_only_through_forward_continues_in_links(
    truth: Groundtruth, blocks: list[str], links: list[tuple[int, int]], expected: float
) -> None:
    markdown = markdown_of(*blocks)
    relations = [continues_in(f"item{i}", f"item{j}") for i, j in links]
    score = scored(INTEGRITY, truth, markdown, table_items(markdown, blocks), relations)
    assert score == pytest.approx(expected)


UNLINKING_RELATIONS = [stays_apart("item0", "item1"), continues_in("item0", "para")]


@pytest.mark.parametrize("relation", UNLINKING_RELATIONS)
def test_table_integrity_misses_a_split_table_without_a_link_between_its_tables(relation):
    markdown = markdown_of("Intro.", *S_PAGES)
    para = layout_over("para", ElementClass.PARAGRAPH, markdown, "Intro.")
    layout = [*table_items(markdown, S_PAGES), para]
    assert scored(INTEGRITY, S_SPLIT, markdown, layout, [relation]) == 0.0


def text_item(table: str) -> LayoutItem:
    text = " ".join(" ".join(row) for row in cells(table))
    return layout_item("n0", ElementClass.TABLE, text=text)


def span_item(item_id: str, markdown: str, first: str, last: str) -> LayoutItem:
    start, end = markdown.index(first), markdown.index(last) + len(last)
    return layout_item(item_id, ElementClass.TABLE, md_start=start, md_end=end)


def both_pieces(markdown: str) -> list[LayoutItem]:
    return [span_item("both", markdown, *S_PAGES)]


def paragraph_intro(markdown: str) -> list[LayoutItem]:
    return [layout_over("para", ElementClass.PARAGRAPH, markdown, "Intro.")]


def first_anchor_pieces(markdown: str) -> list[LayoutItem]:
    first, last = S_PAGES
    return [span_item("item0", markdown, first, EXTRA), span_item("item1", markdown, last, last)]


BUDGET = "Item|2023|2024 / Rent|10|11 / Staff|20|21 / Total|30|32"
TRAVEL = "Item|2023|2024 / Rent|15|16 / Travel|5|6 / Total|20|22"
S_TEXT = " / ".join("|".join(row) for row in [S_HEADER, *S_ROWS])
AROUND_EXTRA = [S_PAGES[0], EXTRA, S_PAGES[1]]
ITEMS = ("item0", "item1")
FIRST_ANCHORS = ("item0", "item0", "item1")
N0, AB = ("n0", "n0"), ("a", "b")
BUDGET_ITEM = [text_item(BUDGET)]
INTRO_LINKS = [AB, ("para", "a")]
UNCLAIMING_ITEMS = [layout_item(item_id, ElementClass.TABLE) for item_id in AB]
LAYOUT_CASES = {
    "one_item_over_both": (S_SPLIT, S_PAGES, both_pieces, [], ("both", "both"), 0.0),
    "gt_link_to_one_anchor": (S_SPLIT, S_PAGES, [], [AB], ("a", "a"), 0.0),
    "gt_link_between_anchors": (XY_APART, [X_ONE, Y_ONE], [], [AB], AB, 1.0),
    "table_twice": (S_SPLIT, [S_ONE, "Again:", S_ONE], [text_item(S_TEXT)], [], N0, 2 / 3),
    "stays_apart": (pair(BUDGET, TRAVEL), [pipe(BUDGET), pipe(TRAVEL)], BUDGET_ITEM, [], N0, 1.0),
    "first_on_anchor": (S_SPLIT, AROUND_EXTRA, first_anchor_pieces, [ITEMS], FIRST_ANCHORS, 2 / 3),
    "no_table_items": (S_SPLIT, ["Intro.", *S_PAGES], paragraph_intro, INTRO_LINKS, ("a", "a"), 0),
    "unclaiming_items": (XY_APART, [X_ONE, Y_ONE], UNCLAIMING_ITEMS, [AB], AB, 1.0),
}


LAYOUT_ARGS = ("truth", "blocks", "layout", "links", "anchors", "expected")


@pytest.mark.parametrize(LAYOUT_ARGS, LAYOUT_CASES.values(), ids=LAYOUT_CASES)
def test_table_integrity_units_follow_the_layout_anchors(
    truth: Truth, blocks: list[str], layout: Layout, links: list, anchors: tuple, expected: float
) -> None:
    markdown = markdown_of(*blocks)
    items = layout(markdown) if callable(layout) else layout
    relations = [continues_in(source, target) for source, target in links]
    ctx = context(truth_of(truth), markdown, items, relations)
    assert output_table_anchors(ctx) == anchors
    assert INTEGRITY.score(ctx) == pytest.approx(expected)


def test_chart_tables_like_a_truth_chart_are_not_integrity_units() -> None:
    truth = groundtruth([table_element("t", 0, Z_HEADER, Z_ROWS), chart_element("c", CHART_TEXT)])
    markdown = markdown_of(Z_ONE, chart_pipe())
    chart = layout_over("chr", ElementClass.CHART, markdown, chart_pipe())
    layout = [span_item("t", markdown, Z_ONE, Z_ONE), chart]
    assert scored(INTEGRITY, truth, markdown, layout) == 1.0


@pytest.mark.parametrize("extra_class", [ElementClass.TABLE, ElementClass.CHART])
def test_extra_table_costs_the_same_when_the_layout_marks_it_as_a_chart(extra_class) -> None:
    table = pipe_table(HEADER, ROWS)
    markdown = markdown_of(table, table)
    second = markdown.rindex(table)
    layout = [
        layout_item("a", ElementClass.TABLE, md_start=0, md_end=len(table)),
        layout_item("b", extra_class, md_start=second, md_end=second + len(table)),
    ]
    result = TRACK.evaluate(one_table(), context(one_table(), markdown, layout))
    assert result.score == pytest.approx(8 / 9)


def checks(cell: str, under: str | None, left: str | None, below: str | None) -> tuple:
    others = zip((UNDER, LEFT, BELOW), (under, left, below), strict=True)
    return tuple(NeighbourRule(kind, cell, other) for kind, other in others if other is not None)


def not_below(*pairs: tuple[str, str]) -> tuple[NeighbourRule, ...]:
    return tuple(NeighbourRule(NOT_BELOW, cell, other) for cell, other in pairs)


FRUIT_RULES = (
    *checks("plum", "Name", "3", "pear"),
    *checks("3", "Qty", "3.50", "2"),
    *checks("3.50", "Price", None, "2.50"),
    *checks("fig", "Name", "4", "plum"),
    *checks("4", "Qty", "4.50", "3"),
    *checks("4.50", "Price", None, "3.50"),
)
KV_RULES = (
    *checks("b1", "K", "y1", "a2"),
    *checks("y1", "V", None, "x2"),
    *checks("c1", "K", "z1", "b1"),
    *checks("z1", "V", None, "y1"),
    *checks("c2", "K", "z2", "c1"),
    *checks("z2", "V", None, "z1"),
)
EMPTY_CELL_RULES = (
    *checks("plum", "Name", None, "pear"),
    *checks("3.50", "Price", None, None),
    *checks("fig", "Name", "4", "plum"),
    *checks("4", None, "4.50", None),
    *checks("4.50", "Price", None, "3.50"),
)
REPEATED_VALUE_RULES = (
    *checks("plum", "Name", "1", "pear"),
    *checks("3.50", "Price", None, "2.50"),
    *FRUIT_RULES[8:11],
    *checks("4", "Qty", "4.50", "1"),
    *FRUIT_RULES[14:],
)
CASE_RULES = (*checks("3", "Q", None, "1"), *checks("z", "N", "4", "plum"))
CASE_RULES += checks("4", "Q", None, "3")
HEADERLESS_RULES = tuple(rule for rule in FRUIT_RULES if rule.kind is not UNDER)
EMPTY_CELLS = chained(["Name", "", "Price"], "apple|1|1.50 / pear|2|", "plum||3.50 / fig|4|4.50")
REPEATED_VALUE = chained(FRUIT, FRUIT_FIRST, "plum|1|3.50 / fig|4|4.50")
APART_RULES = not_below(("north", "ink"), ("70", "6"))
PEN_AND_TEXT = [table_element("a", 0, *parsed(PEN)), para("p1", "north 70", 1)]
PLUM_TABLE = table_element("t2", 1, ["plum"], [["kiwi"]])
PLUM_HEADER = groundtruth([*FRUIT_SPLIT.elements, PLUM_TABLE], FRUIT_SPLIT.relations)
REPEAT_APART = pair("Item|Cost / pen|5 / |6", "Region|Sales / north|5 / south|80")
NEIGHBOUR_RULE_CASES = {
    "split": (FRUIT_SPLIT, FRUIT_RULES),
    "two_unlinked_tables": (pair(PEN, NORTH, None), ()),
    "short_piece": (chained(FRUIT, FRUIT_FIRST, FRUIT_SECOND[:1]), FRUIT_RULES[:8]),
    "headerless": (chained(None, FRUIT_FIRST, FRUIT_SECOND), HEADERLESS_RULES),
    "three_pages": (KV_PAGES, KV_RULES),
    "empty_cells": (EMPTY_CELLS, EMPTY_CELL_RULES),
    "repeated_value": (REPEATED_VALUE, REPEATED_VALUE_RULES),
    "value_in_another_header": (PLUM_HEADER, tuple(r for r in FRUIT_RULES if r.cell != "plum")),
    "case_only_difference": (chained(["N", "Q"], "x|Plum / y|1", "plum|3 / z|4"), CASE_RULES),
    "stays_apart": (APART, APART_RULES),
    "stays_apart_twice": (groundtruth(APART.elements, [*APART.relations] * 2), APART_RULES),
    "not_below_skips_repeats": (REPEAT_APART, ()),
    "not_below_unique_above": (TOTAL_ABOVE, not_below(("north", "Total"))),
    "stays_apart_paragraph": (groundtruth(PEN_AND_TEXT, [stays_apart("a", "p1")]), ()),
    "table_order": (SPLIT_THEN_APART, FRUIT_RULES + not_below(("north", "fig"), ("70", "4"))),
}


RULE_IDS = NEIGHBOUR_RULE_CASES


@pytest.mark.parametrize(("truth", "expected"), NEIGHBOUR_RULE_CASES.values(), ids=RULE_IDS)
def test_cell_neighbour_rules(truth: Groundtruth, expected: tuple[NeighbourRule, ...]) -> None:
    assert table_neighbour_rules(truth) == expected


def test_rows_after_break_limit_the_checked_rows() -> None:
    five_rows = [[f"n{r}", f"v{r}"] for r in range(5)]
    rules = table_neighbour_rules(chained(["H0", "H1"], "top|0", five_rows))
    assert len(rules) == 3 * (2 + 1 + 2)
    assert {rule.cell for rule in rules} == {f"{c}{r}" for c in "nv" for r in range(3)}


def test_demo_rules(demo_groundtruth: Groundtruth) -> None:
    rules = table_neighbour_rules(demo_groundtruth)
    fill = "The alternating fill remains in sequence."
    last = "The closing rule ends the table on page 2."
    assert len(rules) == 3 * 8
    assert rules[:8] == (
        *checks("38", "ID", "Row 38", "37"),
        *checks("Row 38", "Category", fill, "Row 37"),
        *checks(fill, "Observation", None, "This row continues without a heading above it."),
    )
    assert rules[-1:] == checks(last, None, None, "The last two rows complete the table body.")


NEW_TABLE = markdown_of(pipe_table(FRUIT, FRUIT_FIRST), pipe("plum|3|3.50 / fig|4|4.50"))
REPEATED_HEADER = markdown_of(pipe_table(FRUIT, FRUIT_FIRST), pipe_table(FRUIT, FRUIT_SECOND))
APART_KEPT = markdown_of(pipe(PEN), pipe(NORTH))
PEN_H, *PEN_BODY = cells(PEN)
NORTH_ROWS = cells(NORTH)
APART_JOINS = [
    pipe_table(PEN_H, PEN_BODY + NORTH_ROWS[1:]),
    pipe_table(PEN_H, [*PEN_BODY, ["", ""], *NORTH_ROWS[1:]]),
    pipe_table(NORTH_ROWS[0], NORTH_ROWS[1:] + [PEN_H, *PEN_BODY]),
]
MARKED_FRUIT = "NAME|qty|**Price** / APPLE|1|1.50 / Pear|2|2.50 / *PLUM*|3|3.50 / Fig|4|4.50"
GAP_ROW = pipe_table(FRUIT, [*FRUIT_FIRST, ["", "", ""], *FRUIT_SECOND])
NOTE_ROWS = [[name, "", qty, price] for name, qty, price in FRUIT_ROWS]
NOTE_COLUMN = pipe_table(["Name", "Note", "Qty", "Price"], NOTE_ROWS)
NEIGHBOUR_CASES = {
    "data_row_as_header": (FRUIT_SPLIT, NEW_TABLE, (4 + 3) / 16),
    "header_repeated": (FRUIT_SPLIT, REPEATED_HEADER, (6 + 4 + 3) / 16),
    "first_row_as_header": (FRUIT_SPLIT, html_table(None, [FRUIT, *FRUIT_ROWS]), 1.0),
    "any_header_row": (FRUIT_SPLIT, header_rows_table([["Fruit"] * 3, FRUIT], FRUIT_ROWS), 1.0),
    "wrong_header": (FRUIT_SPLIT, pipe_table(["Qty", "Name", "Price"], FRUIT_ROWS), 12 / 16),
    "left_of_needs_next_column": (FRUIT_SPLIT, NOTE_COLUMN, 14 / 16),
    "below_needs_row_above": (FRUIT_SPLIT, GAP_ROW, 13 / 16),
    "case_and_markup": (FRUIT_SPLIT, MARKED_FRUIT, 1.0),
    "any_output_table": (FRUIT_SPLIT, markdown_of(pipe("Colour / red / blue"), FRUIT_TABLE), 1.0),
    "header_only": (FRUIT_SPLIT, "| Name | Qty | Price |\n| --- | --- | --- |\n", 0.0),
    "every_table_twice": (FRUIT_SPLIT, markdown_of(NEW_TABLE, NEW_TABLE), (4 + 3) / 16),
    "join_everything": (SPLIT_THEN_APART, joined(SPLIT_THEN_APART), 16 / 18),
    "stays_apart_kept": (APART, APART_KEPT, 1.0),
    **{f"stays_apart_joined_{i}": (APART, markdown, 0.0) for i, markdown in enumerate(APART_JOINS)},
    "not_below_cell_missing": (APART, PEN, 0.0),
    "not_below_empty_output": (APART, "", 0.0),
    "any_table_joins": (APART, markdown_of(APART_KEPT, APART_JOINS[0]), 0.0),
}


@pytest.mark.parametrize(CASE, NEIGHBOUR_CASES.values(), ids=NEIGHBOUR_CASES)
def test_cell_neighbour_tests(truth: Truth, markdown: str, expected: float) -> None:
    assert scored(NEIGHBOURS, truth, as_markdown(markdown)) == pytest.approx(expected)


@dataclass(frozen=True)
class _Node:
    kind: str
    text: str = ""
    children: tuple["_Node", ...] = ()


def _table(grid: Grid) -> _Node:
    return _Node(
        "table",
        children=tuple(
            _Node("row", children=tuple(_Node("cell", text) for text in row)) for row in grid
        ),
    )


def _reference_distance(first: Grid, second: Grid) -> float:
    @cache
    def forest(a: tuple[_Node, ...], b: tuple[_Node, ...]) -> float:
        if not a and not b:
            return 0.0
        if not a:
            return forest(a, (*b[:-1], *b[-1].children)) + 1
        if not b:
            return forest((*a[:-1], *a[-1].children), b) + 1
        x, y = a[-1], b[-1]
        rename = (
            1.0
            if x.kind != y.kind
            else Levenshtein.normalized_distance(x.text, y.text)
            if x.kind == "cell"
            else 0.0
        )
        return min(
            forest((*a[:-1], *x.children), b) + 1,
            forest(a, (*b[:-1], *y.children)) + 1,
            forest(a[:-1], b[:-1]) + forest(x.children, y.children) + rename,
        )

    return forest((_table(first),), (_table(second),))


def _assert_matches_reference(first: Grid, second: Grid) -> None:
    a, b = TableTree.of(first), TableTree.of(second)
    expected_distance = _reference_distance(first, second)
    expected_score = max(0.0, 1.0 - expected_distance / max(a.size, b.size))
    assert tree_edit_distance(a, b) == pytest.approx(expected_distance)
    assert tree_edit_distance(b, a) == pytest.approx(expected_distance)
    assert teds(first, second) == pytest.approx(expected_score)
    assert teds(second, first) == pytest.approx(expected_score)


def test_random_table_distances_match_independent_recursion() -> None:
    rng = random.Random(15804)
    labels = ("", "a", "b", "ab", "Revenue", "é", "e\u0301", "東京", "😀")

    def grid() -> Grid:
        return tuple(
            tuple(rng.choice(labels) for _ in range(rng.randint(0, 3)))
            for _ in range(rng.randint(0, 4))
        )

    for _ in range(300):
        _assert_matches_reference(grid(), grid())
