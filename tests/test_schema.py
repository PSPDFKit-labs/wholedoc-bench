from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from tests.builders import (
    CHART_BOX,
    ICAO_TD3,
    LONG_NUMBER_TD1,
    REGIONS,
    REVENUE,
    TITLE,
    TOP_LEFT,
    URL,
    chart_element,
    chart_item,
    code,
    element,
    groundtruth,
    invoice,
    item_series,
    mrz,
    named_series,
    para,
    prediction,
    span,
    truth,
    truth_with_series,
    two_labels,
    two_named,
)
from wholedoc_bench.schema import (
    Box,
    Chart,
    ChartKind,
    ChartPoint,
    ChartSeries,
    Element,
    ElementClass,
    Groundtruth,
    LayoutItem,
    Planted,
    Prediction,
    RelationKind,
    StyleName,
    StyleSpan,
    TableCell,
    ValueAxis,
    is_style_element,
)


def test_relations_use_from_and_to_in_json(demo_groundtruth: Groundtruth) -> None:
    dumped = demo_groundtruth.model_dump(by_alias=True)
    assert dumped["relations"][0] == {
        "kind": "caption_of",
        "from": "e3",
        "to": "e4",
        "hyphenated": False,
    }


def test_repeated_id_in_reading_order_is_rejected(demo_groundtruth: Groundtruth) -> None:
    data = demo_groundtruth.model_dump(by_alias=True)
    data["reading_order"] = ["e2", "e7", "e2"]
    with pytest.raises(ValidationError, match="reading_order: e2 appears more than once"):
        Groundtruth.model_validate(data)


def test_relation_to_unknown_element_is_rejected(demo_groundtruth: Groundtruth) -> None:
    data = demo_groundtruth.model_dump(by_alias=True)
    data["relations"].append({"kind": "caption_of", "from": "e3", "to": "missing"})
    with pytest.raises(ValidationError, match="unknown element id 'missing'"):
        Groundtruth.model_validate(data)


@pytest.mark.parametrize(
    ("box", "message"),
    [
        ([0.9, 0.1, 0.3, 0.2], "ends past the page edge"),
        ([0.1, 0.9, 0.2, 0.3], "ends past the page edge"),
        ([-0.1, 0.1, 0.2, 0.2], "starts before the page edge"),
        ([0.1, 0.1, 0.0, 0.2], "must be positive"),
        ([0.1, 0.1, 0.2, 0.0], "must be positive"),
        ([0.1, 0.1, 1e-200, 1e-200], "area rounds to 0"),
        ([float("nan"), 0.1, 0.2, 0.2], "finite number"),
    ],
)
def test_box_that_does_not_fit_the_page_is_rejected(
    demo_groundtruth: Groundtruth, box: list[float], message: str
) -> None:
    data = demo_groundtruth.model_dump(by_alias=True)
    data["elements"][0]["box"] = box
    with pytest.raises(ValidationError, match=message):
        Groundtruth.model_validate(data)


def test_box_touching_the_page_edge_is_accepted(demo_groundtruth: Groundtruth) -> None:
    data = demo_groundtruth.model_dump(by_alias=True)
    data["elements"][0]["box"] = [0.0, 0.0, 1.0, 1.0]
    Groundtruth.model_validate(data)


def test_layout_item_box_is_checked_too() -> None:
    with pytest.raises(ValidationError, match="layout item a: box ends past"):
        LayoutItem(id="a", cls=ElementClass.PARAGRAPH, page=0, box=(0.9, 0.1, 0.3, 0.2))


def test_unknown_class_is_rejected(demo_groundtruth: Groundtruth) -> None:
    data = demo_groundtruth.model_dump(by_alias=True)
    data["elements"][0]["class"] = "page_number"
    with pytest.raises(ValidationError, match="class"):
        Groundtruth.model_validate(data)


def test_stays_apart_is_accepted_across_one_page_break(demo_groundtruth: Groundtruth) -> None:
    data = demo_groundtruth.model_dump(by_alias=True)
    data["relations"] = [{"kind": "stays_apart", "from": "e2", "to": "e7"}]
    truth = Groundtruth.model_validate(data)
    assert truth.relations[0].kind is RelationKind.STAYS_APART


@pytest.mark.parametrize(
    ("relations", "message"),
    [
        ([{"kind": "stays_apart", "from": "e2", "to": "e9"}], "must cross one page break"),
        (
            [{"kind": "continues_in", "from": "e7", "to": "e2"}],
            "must stay on its page or cross one page break",
        ),
        (
            [
                {"kind": "continues_in", "from": "e4", "to": "e6"},
                {"kind": "stays_apart", "from": "e4", "to": "e6"},
            ],
            "both continues_in and stays_apart",
        ),
    ],
)
def test_bad_page_break_relations_are_rejected(
    demo_groundtruth: Groundtruth, relations: list[dict[str, str]], message: str
) -> None:
    data = demo_groundtruth.model_dump(by_alias=True)
    data["relations"] = relations
    with pytest.raises(ValidationError, match=message):
        Groundtruth.model_validate(data)


def _same_page_truth(
    relations: list[dict[str, str]], reading_order: list[str] | None = None
) -> dict[str, object]:
    classes = {
        "p1": "paragraph",
        "p2": "paragraph",
        "p3": "paragraph",
        "l1": "list_item",
        "l2": "list_item",
        "t1": "table",
        "t2": "table",
        "q1": "paragraph",
    }
    return {
        "schema_version": "0.1",
        "document": {
            "id": "same_page",
            "source": "born-digital",
            "groundtruth_source": "generator",
            "language": ["en"],
            "pages": [{"index": i, "width": 595.0, "height": 842.0} for i in range(2)],
        },
        "elements": [
            {"id": id_, "class": cls, "page": 1 if id_ == "q1" else 0, "text": id_}
            for id_, cls in classes.items()
        ],
        "relations": relations,
        "reading_order": reading_order or [],
    }


@pytest.mark.parametrize(
    "relations",
    [
        [{"kind": "continues_in", "from": "p1", "to": "p2"}],
        [{"kind": "continues_in", "from": "l1", "to": "l2"}],
        [
            {"kind": "continues_in", "from": "p1", "to": "p2"},
            {"kind": "continues_in", "from": "p2", "to": "q1"},
        ],
    ],
)
def test_same_page_continuation_is_accepted(relations: list[dict[str, str]]) -> None:
    Groundtruth.model_validate(_same_page_truth(relations, ["p1", "p2", "p3", "q1"]))


@pytest.mark.parametrize(
    ("relations", "reading_order", "message"),
    [
        (
            [{"kind": "continues_in", "from": "t1", "to": "t2"}],
            None,
            "only a paragraph or a list item can continue",
        ),
        (
            [{"kind": "continues_in", "from": "p1", "to": "l1"}],
            None,
            "only a paragraph or a list item can continue",
        ),
        (
            [{"kind": "continues_in", "from": "p2", "to": "p1"}],
            ["p1", "p2"],
            "must go forward in reading_order",
        ),
        (
            [
                {"kind": "continues_in", "from": "p1", "to": "p2"},
                {"kind": "continues_in", "from": "p2", "to": "p1"},
            ],
            None,
            "is a cycle",
        ),
        (
            [
                {"kind": "continues_in", "from": "p1", "to": "p2"},
                {"kind": "continues_in", "from": "p1", "to": "p3"},
            ],
            None,
            "p1 continues in both p2 and p3",
        ),
        (
            [
                {"kind": "continues_in", "from": "p1", "to": "p3"},
                {"kind": "continues_in", "from": "p2", "to": "p3"},
            ],
            None,
            "p3 continues both p1 and p2",
        ),
        (
            [
                {"kind": "continues_in", "from": "p1", "to": "p2"},
                {"kind": "continues_in", "from": "p2", "to": "q1"},
                {"kind": "stays_apart", "from": "p1", "to": "q1"},
            ],
            None,
            "both are pieces of one continues_in chain",
        ),
    ],
)
def test_bad_same_page_relations_are_rejected(
    relations: list[dict[str, str]], reading_order: list[str] | None, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Groundtruth.model_validate(_same_page_truth(relations, reading_order))


CELL_BOX = [0.1, 0.3, 0.2, 0.1]


def table_element(cls: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": "t", "class": cls, "page": 0, "box": [0.1, 0.3, 0.6, 0.3], "cells": cells}


def table_layout_item(cls: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": "x", "class": cls, "page": 0, "cells": cells}


def test_groundtruth_cell_needs_a_box() -> None:
    Element.model_validate(table_element("table", [{"text": "a", "box": CELL_BOX}]))
    for unboxed in ({"text": "a"}, {"text": "a", "box": None}):
        with pytest.raises(ValidationError, match="every table cell needs a box"):
            Element.model_validate(table_element("table", [unboxed]))


def test_cells_on_an_item_that_is_not_a_table_are_rejected() -> None:
    cells = [{"text": "a", "box": CELL_BOX}]
    Element.model_validate(table_element("table", cells))
    LayoutItem.model_validate(table_layout_item("table", cells))
    with pytest.raises(ValidationError, match="only a table can have cells"):
        Element.model_validate(table_element("paragraph", cells))
    with pytest.raises(ValidationError, match="only a table can have cells"):
        LayoutItem.model_validate(table_layout_item("figure", cells))


def test_predicted_cell_may_have_no_box() -> None:
    item = LayoutItem.model_validate(table_layout_item("table", [{"text": "a"}]))
    assert item.cells is not None
    assert item.cells[0].box is None


def test_a_predicted_cell_has_its_whole_place_in_the_grid_or_none() -> None:
    place = {"row": 2, "column": 0, "rows": 1, "columns": 3}
    item = LayoutItem.model_validate(table_layout_item("table", [{"text": "a", **place}]))
    assert item.cells is not None
    cell = item.cells[0]
    assert (cell.row, cell.column, cell.rows, cell.columns) == (2, 0, 1, 3)
    with pytest.raises(ValidationError, match="go together"):
        LayoutItem.model_validate(
            table_layout_item("table", [{"text": "a", "row": 2, "column": 0}])
        )
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        LayoutItem.model_validate(table_layout_item("table", [{"text": "a", **place, "rows": 0}]))


def test_a_cell_without_a_place_is_written_as_before(demo_groundtruth: Groundtruth) -> None:
    assert TableCell(text="a").model_dump(by_alias=True) == {"text": "a", "box": None}
    placed = TableCell(text="a", row=1, column=2, rows=1, columns=1).model_dump(by_alias=True)
    assert placed == {"text": "a", "box": None, "row": 1, "column": 2, "rows": 1, "columns": 1}
    assert '"row"' not in demo_groundtruth.model_dump_json(by_alias=True)


def test_a_groundtruth_cell_has_no_place_in_the_grid() -> None:
    cell = {"text": "a", "box": CELL_BOX, "row": 0, "column": 0, "rows": 1, "columns": 1}
    with pytest.raises(ValidationError, match="header and rows hold the grid"):
        Element.model_validate(table_element("table", [cell]))


@pytest.mark.parametrize(
    ("box", "message"),
    [
        ([0.9, 0.3, 0.2, 0.1], "ends past the page edge"),
        ([0.1, float("nan"), 0.1, 0.1], "finite number"),
    ],
)
def test_cell_box_off_the_page_is_rejected(box: list[float], message: str) -> None:
    cells = [{"text": "a", "box": box}]
    with pytest.raises(ValidationError, match=message):
        Element.model_validate(table_element("table", cells))
    with pytest.raises(ValidationError, match=message):
        LayoutItem.model_validate(table_layout_item("table", cells))


def test_cells_use_the_cells_key_and_survive_a_json_round_trip(
    demo_groundtruth: Groundtruth,
) -> None:
    dumped = demo_groundtruth.model_dump(by_alias=True)
    table = next(e for e in dumped["elements"] if e["id"] == "e6")
    assert table["cells"][0]["text"] == "38"
    again = Groundtruth.model_validate_json(demo_groundtruth.model_dump_json(by_alias=True))
    assert again == demo_groundtruth


def test_groundtruth_table_with_an_empty_cells_list_is_rejected() -> None:
    Element.model_validate(table_element("table", [{"text": "a", "box": CELL_BOX}]))
    with pytest.raises(ValidationError, match="needs at least one cell"):
        Element.model_validate(table_element("table", []))


def test_predicted_table_may_have_an_empty_cells_list() -> None:
    item = LayoutItem.model_validate(table_layout_item("table", []))
    assert item.cells == []


def test_missing_null_and_empty_styles_have_distinct_meanings() -> None:
    assert element().styles is None
    assert element(styles=None).styles is None
    assert element(styles=[]).styles == []


def test_all_six_style_names_are_supported() -> None:
    names = ["bold", "italic", "underline", "strikethrough", "superscript", "subscript"]
    result = element(styles=[span(name) for name in names])
    expected = [StyleSpan(style=StyleName(name), start=7, end=13) for name in sorted(names)]
    assert result.styles == expected


@pytest.mark.parametrize("style", ["Bold", "strike"])
def test_unknown_style_names_are_rejected(style: str) -> None:
    with pytest.raises(ValidationError):
        element(styles=[span(style)])


@pytest.mark.parametrize(("field", "offset"), [("start", True), ("start", "7"), ("end", 7.0)])
def test_offsets_require_integers_without_coercion(field: str, offset: Any) -> None:
    data = span()
    data[field] = offset
    with pytest.raises(ValidationError):
        element(styles=[data])


@pytest.mark.parametrize("start,end", [(-1, 1), (0, -1), (0, 0), (7, 14)])
def test_offsets_must_fit_nonempty_raw_printed_range(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        element(styles=[span(start=start, end=end)])


def test_offsets_count_raw_unicode_code_points_before_normalization() -> None:
    text = "😀é **ﬁ**¹"
    result = element(
        text=text,
        styles=[span("superscript", 9, 10), span("bold", 6, 7), span("italic", 1, 3)],
    )
    assert result.styles is not None
    assert [result.text[s.start : s.end] for s in result.styles] == ["é", "ﬁ", "¹"]


def test_whitespace_only_spans_are_rejected() -> None:
    with pytest.raises(ValidationError, match="contains only whitespace"):
        element(text=" ", styles=[span(start=0, end=1)])


def test_spaces_with_printed_characters_are_valid_span_content() -> None:
    assert element(text=" a ", styles=[span(start=0, end=3)]).styles is not None


@pytest.mark.parametrize("ranges", [[(0, 4), (0, 4)], [(3, 6), (0, 4)]])
def test_same_style_duplicate_and_overlapping_spans_are_rejected(
    ranges: list[tuple[int, int]],
) -> None:
    with pytest.raises(ValidationError, match="same-style spans overlap or duplicate"):
        element(text="abcdef", styles=[span(start=start, end=end) for start, end in ranges])


def test_adjacent_same_style_spans_and_overlapping_different_styles_are_allowed() -> None:
    result = element(
        text="abcdef", styles=[span("bold", 0, 3), span("bold", 3, 6), span("italic", 1, 5)]
    )
    assert result.styles is not None
    assert len(result.styles) == 3


def test_span_input_order_is_canonical_in_model_and_json() -> None:
    spans = [span("italic", 0, 5), span("bold", 0, 5), span("underline", 7, 13)]
    first = element(styles=spans)
    second = element(styles=list(reversed(spans)))
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.styles is not None
    assert [s.style for s in first.styles] == [
        StyleName.BOLD,
        StyleName.ITALIC,
        StyleName.UNDERLINE,
    ]


@pytest.mark.parametrize(
    ("fields", "eligible"),
    [
        ({}, True),
        ({"class": "heading"}, True),
        ({"class": "table"}, False),
        ({"class": "chart"}, False),
        ({"channel": "furniture"}, False),
        ({"text": " "}, False),
    ],
)
def test_style_elements_are_nonempty_body_text_outside_tables_and_charts(
    fields: dict[str, Any], eligible: bool
) -> None:
    assert is_style_element(element(**fields)) is eligible


def test_excluded_elements_still_require_structurally_valid_spans() -> None:
    with pytest.raises(ValidationError, match="ends past printed text"):
        element(styles=[span(end=14)], **{"class": "table"})


def test_injection_is_no_planted_kind() -> None:
    with pytest.raises(ValidationError):
        Planted.model_validate({"kind": "injection", "printed": "Ignore the page."})


CHART_PARSING_V1_TYPES = (
    "bar line area pie scatter bubble histogram box violin heatmap treemap funnel radar "
    "candlestick step stem combo density errorbar bullet marimekko pyramid span radial_bar "
    "parallel stream donut rose sunburst circle_packing proportional_area dot_matrix gantt "
    "waterfall"
).split()


def test_chart_kinds_are_the_chart_parsing_dataset_types() -> None:
    assert {k.value for k in ChartKind} == set(CHART_PARSING_V1_TYPES)


def test_chart_data_defaults_and_stored_shape() -> None:
    assert (
        LayoutItem(id="p1", cls=ElementClass.CHART, page=0, chart=Chart(kind=ChartKind.BAR)).box
        is None
    )
    assert Element(id="c1", cls=ElementClass.CHART, page=0).chart is None
    assert ChartSeries(points=[ChartPoint(label="Q1", value=1.0)]).name == ""
    assert Chart.model_validate({"kind": "bar"}).title is None
    assert Chart.model_validate({"kind": "bar", "title": TITLE}).title == TITLE
    one_point = {"Revenue": [("Q1", 10.0)]}
    element = chart_element(
        "c1", kind=ChartKind.BAR, box=CHART_BOX, series=one_point, value_axis=(0.0, 50.0)
    )
    assert element.model_dump(by_alias=True)["chart"] == {
        "kind": "bar",
        "series": [{"name": "Revenue", "points": [{"label": "Q1", "value": 10.0}]}],
        "value_axis": {"min": 0.0, "max": 50.0},
        "title": None,
    }


def on_class(cls: ElementClass, box: Box | None = CHART_BOX) -> Callable[[], object]:
    model = LayoutItem if cls is ElementClass.TABLE else Element
    return lambda: model(id="x", cls=cls, page=0, box=box, chart=Chart(kind=ChartKind.BAR))


REPEATS = {two_labels: [("Q1", " Q1 ")], two_named: [("North", " North "), ("", "")]}


INVALID = {
    "chart data on a figure": on_class(ElementClass.FIGURE),
    "chart data on a layout table": on_class(ElementClass.TABLE),
    "groundtruth chart without a box": on_class(ElementClass.CHART, None),
    "unknown kind": lambda: Chart.model_validate({"kind": "multi_panel"}),
    "empty title": lambda: Chart(kind=ChartKind.BAR, title=""),
    "blank title": lambda: Chart(kind=ChartKind.BAR, title="   "),
    "axis max equal to min": lambda: ValueAxis(min=0.0, max=0.0),
    "axis max below min": lambda: ValueAxis(min=10.0, max=5.0),
    "infinite axis": lambda: ValueAxis(min=0.0, max=float("inf")),
    "nan axis": lambda: ValueAxis(min=0.0, max=float("nan")),
    "nan point": lambda: ChartPoint(label="Q1", value=float("nan")),
    "infinite point": lambda: ChartPoint(label="Q1", value=float("inf")),
    "series without points": lambda: truth_with_series(named_series("Costs", [])),
    **{
        f"{build.__name__} {a!r} {b!r}": lambda build=build, a=a, b=b: build(a, b)
        for build, pairs in REPEATS.items()
        for a, b in pairs
    },
}


@pytest.mark.parametrize("build", INVALID.values(), ids=INVALID)
def test_invalid_chart_data_is_rejected(build: Callable[[], object]) -> None:
    with pytest.raises(ValidationError):
        build()


def test_chart_data_survives_a_json_round_trip() -> None:
    first = chart_element(
        "c1", kind=ChartKind.BAR, box=CHART_BOX, series=REGIONS, value_axis=(-5, 50), title=TITLE
    )
    second = chart_element(
        "c2", kind=ChartKind.PIE, page=1, box=TOP_LEFT, series={"": [("A", 0.5)]}
    )
    truth = groundtruth([first, second], page_count=2)
    repeated = item_series(("", [("Jan", 1.0), ("jan", 2.0)]), ("", []), ("", []))
    layout = [
        chart_item("p1", ChartKind.LINE, series=REVENUE, title=TITLE),
        chart_item("p2", None, box=None),
    ]
    pred = prediction(layout=[*layout, *repeated.layout])
    assert Groundtruth.model_validate_json(truth.model_dump_json(by_alias=True)) == truth
    assert Prediction.model_validate_json(pred.model_dump_json(by_alias=True)) == pred


def test_only_a_machine_readable_element_has_a_payload() -> None:
    with pytest.raises(ValidationError, match="only a machine_readable element has a payload"):
        truth({**para("p", "Scan to pay."), "payload": URL})


def test_a_machine_readable_element_has_no_text() -> None:
    with pytest.raises(ValidationError, match="a machine_readable element has no text"):
        invoice(code("q", text=URL))


def test_an_empty_payload_is_rejected() -> None:
    with pytest.raises(ValidationError, match="an empty payload"):
        invoice(code("q", payload="  "))


def test_only_a_machine_readable_element_has_a_format() -> None:
    with pytest.raises(ValidationError, match="only a machine_readable element has a format"):
        truth({**para("p", "Passport"), "format": "MrzTd3"})


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"payload": None}, "needs a box and its lines"),
        ({"box": None}, "needs a box and its lines"),
        ({"payload": ICAO_TD3.split("\n")[0]}, "2 lines of 44"),
        ({"format": "MrzTd9"}, "unknown MRZ format"),
        ({"format": "MrzTd1", "payload": LONG_NUMBER_TD1}, "longer than 9 characters"),
    ],
)
def test_a_zone_needs_a_box_and_lines_of_its_format(changes: dict[str, Any], error: str) -> None:
    zone = {k: v for k, v in {**mrz("m"), **changes}.items() if v is not None}
    with pytest.raises(ValidationError, match=error):
        truth(para("p", "Passport"), zone)


@pytest.mark.parametrize("retired", ["barcode", "machineReadable"])
def test_a_class_name_other_than_machine_readable_is_rejected(retired: str) -> None:
    with pytest.raises(ValidationError, match="machine_readable"):
        truth(para("p", "Passport"), {**mrz("m"), "class": retired})
    with pytest.raises(ValidationError, match="machine_readable"):
        LayoutItem.model_validate({"id": "i", "class": retired, "page": 0, "format": "QrCode"})


def test_an_unknown_class_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="class"):
        LayoutItem.model_validate({"id": "i", "class": "qr_code", "page": 0})


def test_only_a_machine_readable_item_has_a_format_or_a_check() -> None:
    with pytest.raises(
        ValidationError, match="only a machine_readable item has a format or a check"
    ):
        LayoutItem(id="i", cls=ElementClass.PARAGRAPH, page=0, check_digits_valid=True)


def merged_table(**changes: Any) -> dict[str, Any]:
    return {
        "id": "t",
        "class": "table",
        "page": 0,
        "header": ["Revenue", None],
        "rows": [["2025", "2026"], ["100", "120"]],
        "spans": [{"row": 0, "column": 0, "columns": 2}],
        **changes,
    }


def test_a_merged_cell_is_written_once_with_its_span() -> None:
    table = Element.model_validate(merged_table())
    assert table.spans is not None
    assert table.spans[0].positions() == [(0, 0), (0, 1)]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"spans": None}, "row 0, column 1 is null but no span covers it"),
        ({"header": ["Revenue", "Revenue"]}, "row 0, column 1 is covered by the span at row 0"),
        ({"header": [None, "Revenue"]}, "its top-left cell holds the text"),
        ({"spans": [{"row": 0, "column": 1, "columns": 2}]}, "outside the grid"),
        ({"spans": [{"row": 0, "column": 0}]}, "covers one cell"),
        (
            {
                "rows": [["2025", None], ["100", "120"]],
                "spans": [
                    {"row": 0, "column": 0, "columns": 2},
                    {"row": 1, "column": 0, "columns": 2},
                    {"row": 1, "column": 1, "rows": 2},
                ],
            },
            "overlaps another span",
        ),
        (
            {
                "header": ["Region", "Q1"],
                "rows": [[None, "10"], ["North", "30"]],
                "spans": [{"row": 0, "column": 0, "rows": 2}],
            },
            "crosses from the header into the rows",
        ),
        ({"spans": []}, "needs at least one span"),
        ({"class": "paragraph", "text": "x"}, "only a table can have spans"),
    ],
    ids=[
        "null without a span",
        "covered position repeats the text",
        "top-left is null",
        "outside the grid",
        "one cell",
        "overlap",
        "header into rows",
        "empty list",
        "not a table",
    ],
)
def test_a_span_that_does_not_fit_the_grid_is_rejected(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Element.model_validate(merged_table(**changes))


def test_a_header_is_a_list_of_rows_and_a_flat_list_is_one_row() -> None:
    two_rows = merged_table(
        header=[["Region", "Revenue", None], [None, "2025", "2026"]],
        rows=[["North", "100", "120"]],
        spans=[{"row": 0, "column": 0, "rows": 2}, {"row": 0, "column": 1, "columns": 2}],
    )
    assert Element.model_validate(two_rows).header == [
        ["Region", "Revenue", None],
        [None, "2025", "2026"],
    ]
    assert Element.model_validate(merged_table()).header == [["Revenue", None]]
    assert Element.model_validate(merged_table(header=[], spans=None)).header == []
