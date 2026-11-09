import pytest

from tests.builders import (
    CHART_HEADER,
    CHART_ROWS,
    CHART_TEXT,
    HEADER,
    ROWS,
    as_tuples,
    cells_text,
    chart_element,
    chart_pipe,
    chart_table_anchors,
    chart_tables,
    context,
    continues_in,
    grid,
    groundtruth,
    html_table,
    indexes,
    layout_item,
    layout_over,
    markdown_of,
    markdown_tables,
    one_table,
    output_table_anchors,
    para,
    pipe_table,
    quoted,
    stays_apart,
    table_and_chart,
    table_element,
    two_different_tables,
)
from wholedoc_bench.schema import ElementClass, Groundtruth, LayoutItem
from wholedoc_bench.scorer.utils.tracks.t4_tables.table_matching import output_tables
from wholedoc_bench.scorer.utils.tracks.table_links import (
    linked_tables,
    parse_truth_tables,
    plain_number,
)


def test_demo_truth_table_joins_both_pages(demo_groundtruth: Groundtruth) -> None:
    tables = parse_truth_tables(demo_groundtruth)
    assert len(tables) == 1
    table = tables[0]
    assert tuple(table.element_ids) == ("e4", "e6")
    assert tuple(table.pages) == (0, 1)
    header = table.header
    assert header is not None
    assert header == (("ID", "Category", "Observation"),)
    assert len(table.rows) == 40
    assert tuple(table.break_rows) == (37,)
    assert tuple(table.rows[37]) == ("38", "Row 38", "The alternating fill remains in sequence.")


def test_truth_table_chain_drops_repeated_header_and_records_breaks() -> None:
    truth = groundtruth(
        [
            table_element("t1", 0, ["A", "B"], [["a1", "b1"], ["a2", "b2"]]),
            table_element("t2", 1, ["A", "B"], [["a3", "b3"], ["a4", "b4"]]),
            table_element("t3", 2, None, [["a5", "b5"]]),
        ],
        [continues_in("t1", "t2"), continues_in("t2", "t3")],
    )
    [table] = parse_truth_tables(truth)
    assert tuple(table.element_ids) == ("t1", "t2", "t3")
    assert tuple(table.pages) == (0, 1, 2)
    header = table.header
    assert header is not None
    assert header == (("A", "B"),)
    assert as_tuples(table.rows) == (
        ("a1", "b1"),
        ("a2", "b2"),
        ("a3", "b3"),
        ("a4", "b4"),
        ("a5", "b5"),
    )
    assert tuple(table.break_rows) == (2, 4)
    assert as_tuples(table.grid) == (("A", "B"), *as_tuples(table.rows))


def test_stays_apart_pair_stays_two_truth_tables() -> None:
    tables = parse_truth_tables(two_different_tables(stays_apart))
    assert [tuple(t.element_ids) for t in tables] == [("a",), ("b",)]


def test_continues_in_to_a_paragraph_does_not_join() -> None:
    truth = groundtruth(
        [table_element("t1", 0, HEADER, ROWS), para("p1", "Revenue 10", page=1)],
        [continues_in("t1", "p1")],
    )
    [table] = parse_truth_tables(truth)
    assert tuple(table.element_ids) == ("t1",)


def test_truth_tables_follow_the_position_of_their_first_piece() -> None:
    truth = groundtruth(
        [
            table_element("a2", 1, None, [["a3", "a4"]]),
            table_element("c", 0, ["c1", "c2"], [["c3", "c4"]]),
            table_element("a1", 0, ["a1", "a2"], [["a5", "a6"]]),
        ],
        [continues_in("a1", "a2")],
    )
    tables = parse_truth_tables(truth)
    assert [tuple(t.element_ids) for t in tables] == [("c",), ("a1", "a2")]


def test_empty_truth_table_is_left_out() -> None:
    truth = groundtruth([table_element("e", 0, None, []), table_element("t1", 0, HEADER, ROWS)])
    tables = parse_truth_tables(truth)
    assert [tuple(t.element_ids) for t in tables] == [("t1",)]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_layout_span_marks_a_table_as_chart_with_any_line_ending(newline: str) -> None:
    table_source = pipe_table(HEADER, ROWS).replace("\n", newline)
    intro = newline.join(f"Line {n}." for n in range(len(table_source)))
    markdown = intro + newline * 2 + table_source + newline
    layout = [layout_over("c", ElementClass.CHART, markdown, table_source)]
    [link] = linked_tables(context(one_table(), markdown, layout))
    start = markdown.index(table_source)
    assert (link.kind, link.table.start, link.table.end) == (
        ElementClass.CHART,
        start,
        start + len(table_source),
    )


def test_layout_spans_decide_over_groundtruth_order() -> None:
    chart = chart_pipe()
    table = pipe_table(CHART_HEADER, CHART_ROWS)
    markdown = markdown_of(chart, "Middle.", table)
    second_start = markdown.index(table, markdown.index("Middle."))
    layout = [
        layout_item("c", ElementClass.CHART, md_start=0, md_end=len(chart)),
        layout_item(
            "t", ElementClass.TABLE, md_start=second_start, md_end=second_start + len(table)
        ),
    ]
    ctx = context(table_and_chart(), markdown, layout)
    assert indexes(chart_tables(ctx)) == [0]
    assert indexes(output_tables(linked_tables(ctx))) == [1]


@pytest.mark.parametrize(
    ("chart_share", "expected_kind"),
    [(0.2, "table"), (0.8, "chart")],
)
def test_largest_span_overlap_wins(chart_share: float, expected_kind: str) -> None:
    table_source = pipe_table(HEADER, ROWS)
    markdown = markdown_of(table_source)
    cut = round(len(table_source) * chart_share)
    layout = [
        layout_item("c", ElementClass.CHART, md_start=0, md_end=cut),
        layout_item("t", ElementClass.TABLE, md_start=cut, md_end=len(table_source)),
    ]
    ctx = context(one_table(), markdown, layout)
    assert (len(chart_tables(ctx)) == 1) == (expected_kind == "chart")
    assert (len(output_tables(linked_tables(ctx))) == 1) == (expected_kind == "table")


@pytest.mark.parametrize(
    ("first_cls", "expected_chart_count"),
    [(ElementClass.CHART, 1), (ElementClass.TABLE, 0)],
)
def test_equal_span_overlap_goes_to_the_earliest_anchor(
    first_cls: ElementClass, expected_chart_count: int
) -> None:
    table_source = pipe_table(HEADER, ROWS)
    markdown = markdown_of(table_source)
    half = len(table_source) // 2
    second_cls = ElementClass.TABLE if first_cls is ElementClass.CHART else ElementClass.CHART
    layout = [
        layout_item("first", first_cls, md_start=0, md_end=half),
        layout_item(
            "second", second_cls, md_start=len(table_source) - half, md_end=len(table_source)
        ),
    ]
    ctx = context(one_table(), markdown, layout)
    assert len(chart_tables(ctx)) == expected_chart_count
    assert len(output_tables(linked_tables(ctx))) == 1 - expected_chart_count


def test_layout_anchor_with_a_span_elsewhere_links_by_text() -> None:
    markdown = markdown_of("Intro paragraph.", chart_pipe())
    layout = [
        layout_item("c", ElementClass.CHART, text=CHART_TEXT, md_start=0, md_end=len("Intro")),
    ]
    ctx = context(table_and_chart(), markdown, layout)
    assert indexes(chart_tables(ctx)) == [0]


def test_span_link_and_text_link_combine() -> None:
    table_source = chart_pipe()
    markdown = markdown_of(table_source, "Middle.", table_source)
    second_start = markdown.index(table_source, markdown.index("Middle."))
    layout = [
        layout_item("t", ElementClass.TABLE, text=CHART_TEXT),
        layout_item(
            "c",
            ElementClass.CHART,
            text=CHART_TEXT,
            md_start=second_start,
            md_end=second_start + len(table_source),
        ),
    ]
    ctx = context(table_and_chart(), markdown, layout)
    assert indexes(output_tables(linked_tables(ctx))) == [0]
    assert indexes(chart_tables(ctx)) == [1]
    assert output_table_anchors(ctx) == ("t",)
    assert chart_table_anchors(ctx) == ("c",)


@pytest.mark.parametrize(
    ("layout_order", "expected_tables", "expected_charts"),
    [
        ((ElementClass.TABLE, ElementClass.CHART), [0], [1]),
        ((ElementClass.CHART, ElementClass.TABLE), [1], [0]),
    ],
)
def test_layout_text_and_order_decide_a_chart_next_to_a_table_with_the_same_numbers(
    layout_order: tuple[ElementClass, ElementClass],
    expected_tables: list[int],
    expected_charts: list[int],
) -> None:
    markdown = markdown_of(chart_pipe(), chart_pipe())
    layout = [layout_item(f"i{n}", cls, text=CHART_TEXT) for n, cls in enumerate(layout_order)]
    ctx = context(table_and_chart(), markdown, layout)
    assert indexes(output_tables(linked_tables(ctx))) == expected_tables
    assert indexes(chart_tables(ctx)) == expected_charts


def test_layout_without_tables_or_charts_uses_groundtruth_anchors() -> None:
    truth = groundtruth([chart_element("c1", CHART_TEXT)])
    markdown = markdown_of("Intro.", chart_pipe())
    layout = [layout_item("p", ElementClass.PARAGRAPH, text="Intro.")]
    ctx = context(truth, markdown, layout)
    assert indexes(chart_tables(ctx)) == [0]
    assert chart_table_anchors(ctx) == ("c1",)
    assert output_tables(linked_tables(ctx)) == ()


def test_table_no_layout_item_claims_uses_groundtruth_anchors() -> None:
    header, rows = grid("a", 2, 2)
    truth = groundtruth([table_element("t1", 0, header, rows), chart_element("c1", CHART_TEXT)])
    markdown = markdown_of(pipe_table(header, rows), chart_pipe())
    layout = [layout_item("table-item", ElementClass.TABLE, text=cells_text(header, rows))]
    ctx = context(truth, markdown, layout)
    assert output_table_anchors(ctx) == ("table-item",)
    assert chart_table_anchors(ctx) == ("c1",)


def test_layout_table_class_wins_over_a_groundtruth_chart() -> None:
    truth = groundtruth([chart_element("c1", CHART_TEXT)])
    markdown = markdown_of(chart_pipe())
    layout = [layout_item("t", ElementClass.TABLE, text=CHART_TEXT)]
    ctx = context(truth, markdown, layout)
    assert indexes(output_tables(linked_tables(ctx))) == [0]
    assert chart_tables(ctx) == ()


@pytest.mark.parametrize(
    ("chart_first", "expected_chart_count"),
    [(False, 0), (True, 1)],
)
def test_one_output_table_ties_go_to_the_earliest_anchor(
    chart_first: bool, expected_chart_count: int
) -> None:
    ctx = context(table_and_chart(chart_first), markdown_of(chart_pipe()))
    assert len(chart_tables(ctx)) == expected_chart_count
    assert len(output_tables(linked_tables(ctx))) == 1 - expected_chart_count


def test_repeated_chart_tables_both_stay_charts_on_the_most_similar_anchor() -> None:
    truth = groundtruth(
        [
            table_element("t1", 0, ["Name", "Qty"], [["apple", "1"], ["pear", "2"]]),
            chart_element("c1", CHART_TEXT),
        ]
    )
    table_source = pipe_table(["Name", "Qty"], [["apple", "1"], ["pear", "2"]])
    markdown = markdown_of(table_source, chart_pipe(), chart_pipe())
    ctx = context(truth, markdown)
    assert indexes(output_tables(linked_tables(ctx))) == [0]
    assert indexes(chart_tables(ctx)) == [1, 2]
    assert output_table_anchors(ctx) == ("t1",)
    assert chart_table_anchors(ctx) == ("c1", "c1")


def test_output_table_unlike_every_anchor_is_a_table() -> None:
    truth = groundtruth([chart_element("c1", CHART_TEXT)])
    markdown = markdown_of(pipe_table(["Staff", "Count"], [["Oslo", "3"]]))
    ctx = context(truth, markdown)
    assert indexes(output_tables(linked_tables(ctx))) == [0]
    assert chart_tables(ctx) == ()
    assert output_table_anchors(ctx) == (None,)


@pytest.mark.parametrize("layout_classes", [(), (ElementClass.CHART, ElementClass.TABLE)])
def test_every_output_table_gets_exactly_one_kind_in_markdown_order(
    layout_classes: tuple[ElementClass, ...],
) -> None:
    markdown = markdown_of(
        chart_pipe(),
        pipe_table(HEADER, ROWS),
        html_table(CHART_HEADER, CHART_ROWS),
        pipe_table(["Staff", "Count"], [["Oslo", "3"]]),
    )
    layout = [layout_item(f"i{n}", cls, text=CHART_TEXT) for n, cls in enumerate(layout_classes)]
    ctx = context(table_and_chart(), markdown, layout)
    table_indexes = indexes(output_tables(linked_tables(ctx)))
    chart_indexes = indexes(chart_tables(ctx))
    assert table_indexes == sorted(table_indexes)
    assert chart_indexes == sorted(chart_indexes)
    assert sorted(table_indexes + chart_indexes) == indexes(markdown_tables(ctx))


CONFLICT_TRUTH_TEXT = "Quarter Revenue Q3 18"
FIRST_CONFLICT = pipe_table(["Quarter", "Revenue"], [["Q3", "18"]])
SECOND_CONFLICT = pipe_table(["Quarter", "Revenue"], [["Q3", "25"]])


def conflict_truth() -> Groundtruth:
    return groundtruth([chart_element("c1", CONFLICT_TRUTH_TEXT)])


def test_conflicting_chart_tables_without_layout_are_both_charts() -> None:
    ctx = context(conflict_truth(), markdown_of(FIRST_CONFLICT, SECOND_CONFLICT))
    assert indexes(chart_tables(ctx)) == [0, 1]
    assert output_tables(linked_tables(ctx)) == ()
    assert chart_table_anchors(ctx) == ("c1", "c1")


def test_conflicting_chart_tables_marked_by_layout_spans_are_both_charts() -> None:
    markdown = markdown_of(FIRST_CONFLICT, "Middle.", SECOND_CONFLICT)
    layout = [
        layout_over("chart-a", ElementClass.CHART, markdown, FIRST_CONFLICT),
        layout_over("chart-b", ElementClass.CHART, markdown, SECOND_CONFLICT),
    ]
    ctx = context(conflict_truth(), markdown, layout)
    assert indexes(chart_tables(ctx)) == [0, 1]
    assert output_tables(linked_tables(ctx)) == ()
    assert chart_table_anchors(ctx) == ("chart-a", "chart-b")


def test_conflicting_chart_tables_with_one_layout_chart_are_both_charts() -> None:
    markdown = markdown_of(FIRST_CONFLICT, SECOND_CONFLICT)
    layout = [layout_item("chart-a", ElementClass.CHART, text=CONFLICT_TRUTH_TEXT)]
    ctx = context(conflict_truth(), markdown, layout)
    assert indexes(chart_tables(ctx)) == [0, 1]
    assert output_tables(linked_tables(ctx)) == ()
    assert chart_table_anchors(ctx) == ("chart-a", "chart-a")


YEAR_HEADER = ["Year", "Revenue"]
FIRST_YEAR_ROWS = [["2020", "10"], ["2021", "12"]]
SECOND_YEAR_ROWS = [["2022", "14"], ["2023", "16"]]
YEAR_CHART_TEXT = "Revenue 2020 2021 2022 2023"


def split_table_next_to_chart() -> Groundtruth:
    return groundtruth(
        [
            table_element("t1", 0, YEAR_HEADER, FIRST_YEAR_ROWS),
            table_element("t2", 1, None, SECOND_YEAR_ROWS),
            chart_element("c1", YEAR_CHART_TEXT, page=1),
        ],
        [continues_in("t1", "t2")],
    )


def split_year_markdown() -> str:
    first = pipe_table(YEAR_HEADER, FIRST_YEAR_ROWS)
    second = pipe_table(SECOND_YEAR_ROWS[0], SECOND_YEAR_ROWS[1:])
    return markdown_of(first, second)


def test_split_table_next_to_a_chart_stays_a_table_without_layout() -> None:
    ctx = context(split_table_next_to_chart(), split_year_markdown())
    assert indexes(output_tables(linked_tables(ctx))) == [0, 1]
    assert chart_tables(ctx) == ()
    assert output_table_anchors(ctx) == ("t1", "t1")


def test_split_table_next_to_a_chart_stays_a_table_with_layout_anchors() -> None:
    table_text = cells_text(YEAR_HEADER, FIRST_YEAR_ROWS + SECOND_YEAR_ROWS)
    layout = [
        layout_item("table-item", ElementClass.TABLE, text=table_text),
        layout_item("chart-item", ElementClass.CHART, text=YEAR_CHART_TEXT, page=1),
    ]
    ctx = context(split_table_next_to_chart(), split_year_markdown(), layout)
    assert indexes(output_tables(linked_tables(ctx))) == [0, 1]
    assert chart_tables(ctx) == ()
    assert output_table_anchors(ctx) == ("table-item", "table-item")


@pytest.mark.parametrize(
    ("chart_first", "expected_tables", "expected_charts", "table_anchors"),
    [(False, [0, 1], [2], ("t1", "t2")), (True, [1, 2], [0], ("t2", "t1"))],
)
def test_chart_and_table_with_the_same_numbers_go_by_order_around_another_table(
    chart_first: bool,
    expected_tables: list[int],
    expected_charts: list[int],
    table_anchors: tuple[str, ...],
) -> None:
    same = table_element("t1", 0, CHART_HEADER, CHART_ROWS)
    other = table_element("t2", 0, HEADER, ROWS)
    chart = chart_element("c1", CHART_TEXT)
    elements = [chart, other, same] if chart_first else [same, other, chart]
    markdown = markdown_of(chart_pipe(), pipe_table(HEADER, ROWS), chart_pipe())
    ctx = context(groundtruth(elements), markdown)
    assert indexes(output_tables(linked_tables(ctx))) == expected_tables
    assert indexes(chart_tables(ctx)) == expected_charts
    assert output_table_anchors(ctx) == table_anchors
    assert chart_table_anchors(ctx) == ("c1",)


def test_layout_span_over_a_quoted_pipe_table_marks_it_as_chart() -> None:
    quoted_table = quoted(chart_pipe())
    markdown = markdown_of("Intro text.", quoted_table)
    layout = [layout_over("chart-item", ElementClass.CHART, markdown, quoted_table)]
    ctx = context(table_and_chart(), markdown, layout)
    assert indexes(chart_tables(ctx)) == [0]
    assert output_tables(linked_tables(ctx)) == ()
    assert chart_table_anchors(ctx) == ("chart-item",)


@pytest.mark.parametrize(
    ("layout", "anchor_id"),
    [
        ((), "t1"),
        (
            (layout_item("table-item", ElementClass.TABLE, text=cells_text(HEADER, ROWS)),),
            "table-item",
        ),
    ],
)
def test_table_printed_twice_links_both_copies_to_the_same_table_anchor(
    layout: tuple[LayoutItem, ...], anchor_id: str
) -> None:
    table_source = pipe_table(HEADER, ROWS)
    ctx = context(one_table(), markdown_of(table_source, table_source), layout)
    assert indexes(output_tables(linked_tables(ctx))) == [0, 1]
    assert chart_tables(ctx) == ()
    assert output_table_anchors(ctx) == (anchor_id, anchor_id)


def test_output_tables_without_any_anchor_are_tables_with_no_anchor() -> None:
    truth = groundtruth([para("p1", "Intro.")])
    markdown = markdown_of("Intro.", pipe_table(HEADER, ROWS), chart_pipe())
    ctx = context(truth, markdown)
    assert indexes(output_tables(linked_tables(ctx))) == [0, 1]
    assert output_table_anchors(ctx) == (None, None)
    assert chart_tables(ctx) == ()
    assert chart_table_anchors(ctx) == ()


def test_span_linked_table_is_subtracted_from_its_anchor_residual() -> None:
    table_source = chart_pipe()
    markdown = markdown_of(table_source, table_source)
    layout = [
        layout_item(
            "table-item", ElementClass.TABLE, text=CHART_TEXT, md_start=0, md_end=len(table_source)
        ),
        layout_item("chart-item", ElementClass.CHART, text="Revenue Q1 10 Q2 20"),
    ]
    ctx = context(table_and_chart(), markdown, layout)
    assert indexes(output_tables(linked_tables(ctx))) == [0]
    assert indexes(chart_tables(ctx)) == [1]
    assert output_table_anchors(ctx) == ("table-item",)
    assert chart_table_anchors(ctx) == ("chart-item",)


def test_span_linked_anchor_still_takes_the_next_split_pieces() -> None:
    first = pipe_table(YEAR_HEADER, FIRST_YEAR_ROWS)
    markdown = markdown_of(first, *(pipe_table(row, []) for row in SECOND_YEAR_ROWS))
    table_text = cells_text(YEAR_HEADER, FIRST_YEAR_ROWS + SECOND_YEAR_ROWS)
    layout = [
        layout_item(
            "table-item", ElementClass.TABLE, text=table_text, md_start=0, md_end=len(first)
        ),
        layout_item("chart-item", ElementClass.CHART, text=YEAR_CHART_TEXT, page=1),
    ]
    ctx = context(split_table_next_to_chart(), markdown, layout)
    assert indexes(output_tables(linked_tables(ctx))) == [0, 1, 2]
    assert chart_tables(ctx) == ()
    assert output_table_anchors(ctx) == ("table-item", "table-item", "table-item")


def test_fallback_ties_go_to_the_earliest_anchor() -> None:
    truth = groundtruth([chart_element("c1", CHART_TEXT), chart_element("c2", CHART_TEXT)])
    ctx = context(truth, markdown_of(chart_pipe(), chart_pipe(), chart_pipe()))
    assert indexes(chart_tables(ctx)) == [0, 1, 2]
    assert chart_table_anchors(ctx) == ("c1", "c2", "c1")


@pytest.mark.parametrize(
    ("value", "printed"),
    [(24.0, "24"), (-4.25, "-4.25"), (1e16, "10000000000000000"), (1e-5, "0.00001")],
)
def test_a_chart_value_is_printed_in_its_shortest_form(value: float, printed: str) -> None:
    assert plain_number(value) == printed
