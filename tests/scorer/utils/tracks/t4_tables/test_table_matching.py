import pytest

from tests.builders import (
    HEADER,
    ROWS,
    context,
    groundtruth,
    header_rows_table,
    markdown_of,
    one_table,
    only_table,
    pipe_table,
    table_element,
    two_different_tables,
)
from wholedoc_bench.schema import Groundtruth
from wholedoc_bench.scorer.utils.blocks import MarkdownTable
from wholedoc_bench.scorer.utils.tracks.t4_tables.table_matching import (
    column_labels,
    match_tables,
    output_tables,
)
from wholedoc_bench.scorer.utils.tracks.table_links import linked_tables, parse_truth_tables


def table_of(markdown: str) -> MarkdownTable:
    return only_table(context(one_table(), markdown))


def match_indexes(truth: Groundtruth, markdown: str) -> list:
    ctx = context(truth, markdown)
    matches = match_tables(parse_truth_tables(ctx.truth), output_tables(linked_tables(ctx)))
    return [None if match is None else match.index for match in matches]


METRIC_TABLE = pipe_table(HEADER, ROWS)
TWINS = groundtruth([table_element("t1", 0, HEADER, ROWS), table_element("t2", 0, HEADER, ROWS)])
XY = groundtruth(
    [
        table_element("a", 0, ["x1", "x2"], [["x3", "x4"]]),
        table_element("b", 0, ["y1", "y2"], [["y3", "y4"]]),
    ]
)
XY_OUTPUT = markdown_of(
    pipe_table(["y1", "y2"], [["y3", "y4"], ["x1", "x2"]]),
    pipe_table(["x3", "z1"], [["z2", "z3"]]),
)
CITIES = pipe_table(["City", "Zip"], [["Oslo", "0150"], ["Bern", "3000"]])
MATCH_CASES = {
    "most_similar_first": (XY, XY_OUTPUT, [1, 0]),
    "tie_to_truth_order": (TWINS, METRIC_TABLE, [0, None]),
    "tie_both": (TWINS, markdown_of(METRIC_TABLE, METRIC_TABLE), [0, 1]),
    "no_shared_token": (one_table(), pipe_table(["Staff", "Count"], [["Oslo", "3"]]), [None]),
    "aligned": (two_different_tables(), CITIES, [None, 0]),
}


@pytest.mark.parametrize(("truth", "markdown", "expected"), MATCH_CASES.values(), ids=MATCH_CASES)
def test_tables_match_one_to_one_by_token_similarity(
    truth: Groundtruth, markdown: str, expected: list
):
    assert match_indexes(truth, markdown) == expected


SPAN_HEADER = (
    '<table><tr><th rowspan="2">Item</th><th colspan="2">Revenue</th></tr>'
    "<tr><th>Q1</th><th>Q2</th></tr><tr><td>Rent</td><td>1</td><td>2</td></tr></table>"
)
YEAR_ROWS = [["", "2023", "2023"], ["Item", "Q1", "Q2"]]


@pytest.mark.parametrize(
    ("markdown", "labels"),
    [
        (pipe_table(["Name", "QTY", "Straße"], [["apple", "1", "Main"]]), "name qty strasse"),
        (pipe_table(None, [HEADER, *ROWS]), "metric value"),
        (pipe_table(["", "2023"], [["Rent", "10"]]), " 2023"),
        (header_rows_table(YEAR_ROWS, []), "|item 2023|q1 2023|q2"),
        (SPAN_HEADER, "item revenue|q1 revenue|q2"),
        (header_rows_table([["A"], ["B", "C"]], [["1", "2"]]), "a|b c"),
    ],
)
def test_column_labels_join_the_casefolded_header_cells_of_a_column(markdown: str, labels: str):
    expected = tuple(frozenset(column.split("|")) for column in labels.split(" "))
    assert column_labels(table_of(markdown)) == expected
