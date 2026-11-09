import copy

import pytest
from pydantic import ValidationError

from tests.builders import CHART_BOX, groundtruth, pipe_table, truth
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.schema import (
    Channel,
    Chart,
    ChartKind,
    ChartPoint,
    ChartSeries,
    Element,
    ElementClass,
    Font,
    FontClass,
    Groundtruth,
    RelationKind,
)
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t6_charts import TRACK as T6
from wholedoc_bench.scorer.utils.tracks.table_links import linked_tables

CHART_ID = "5dcb8919-7b70-4b17-842d-b5c500cbaeca"
GDP_ROWS = [["Country", "2025", "2024"], ["Germany", "4.5", "4.4"], ["France", "3.1", "3.0"]]


def test_markdown_is_passed_through(demo_nutrient_raw: RawResult) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    assert pred.markdown == demo_nutrient_raw.payload["output"]["markdown"]


def test_caption_and_continuation_links(demo_nutrient_raw: RawResult) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    by_id = {item.id: item for item in pred.layout}
    captions = [r for r in pred.relations if r.kind is RelationKind.CAPTION_OF]
    assert len(captions) == 1
    assert by_id[captions[0].source].text == "Table 1: Sample records"
    assert by_id[captions[0].target].cls == "table"

    continuations = [r for r in pred.relations if r.kind is RelationKind.CONTINUES_IN]
    assert [(by_id[r.source].cls, r.hyphenated) for r in continuations] == [
        ("table", False),
        ("paragraph", True),
    ]


def test_classes_match_the_groundtruth(
    demo_nutrient_raw: RawResult, demo_groundtruth: Groundtruth
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    assert [item.cls for item in pred.layout] == [e.cls for e in demo_groundtruth.elements]


def test_headers_and_footers_are_furniture(demo_nutrient_raw: RawResult) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    furniture = [item.text for item in pred.layout if item.channel is Channel.FURNITURE]
    assert furniture == ["1", "2", "3"]


def test_unknown_role_is_an_error(demo_nutrient_raw: RawResult) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    payload["output"]["elements"][0]["role"] = "Sidebar"
    raw = RawResult(provider="nutrient", document_id="x", payload=payload)
    with pytest.raises(ValueError, match="no class mapping for Nutrient role 'Sidebar'"):
        NutrientProvider().normalize(raw)


def test_chart_element_becomes_a_chart_item_with_its_table_cells(
    chart_nutrient_raw: RawResult,
) -> None:
    pred = NutrientProvider().normalize(chart_nutrient_raw)
    chart = next(item for item in pred.layout if item.id == CHART_ID)
    assert chart.cls is ElementClass.CHART
    assert chart.box is not None
    assert chart.text.startswith("2025 Score in 2024 South Korea 152 146 Canada 134 134")
    assert chart.text.endswith("Mexico 31 31")


def with_gdp_table(raw: RawResult) -> RawResult:
    payload = copy.deepcopy(raw.payload)
    output = payload["output"]
    chart = next(e for e in output["elements"] if e["type"] == "chart")
    cells = [
        {"row": r, "column": c, "text": text, "bounds": chart["bounds"]}
        for r, row in enumerate(GDP_ROWS)
        for c, text in enumerate(row)
    ]
    table = {"id": "gdp", "type": "table", "page": chart["page"], "bounds": chart["bounds"]}
    output["elements"].append({**table, "cells": cells})
    output["markdown"] += "\n" + pipe_table(GDP_ROWS[0], GDP_ROWS[1:]) + "\n"
    return RawResult(provider="nutrient", document_id="x", payload=payload)


def test_chart_item_claims_its_markdown_table(chart_nutrient_raw: RawResult) -> None:
    raw = with_gdp_table(chart_nutrient_raw)
    links = linked_tables(DocContext(truth(), NutrientProvider().normalize(raw)))
    expected = [(ElementClass.CHART, CHART_ID), (ElementClass.TABLE, "gdp")]
    assert [(link.kind, link.anchor_id) for link in links] == expected


def test_table_cells_follow_row_and_column_order_and_the_item_text_is_unchanged(
    demo_nutrient_raw: RawResult, demo_groundtruth: Groundtruth
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    tables = [item for item in pred.layout if item.cls == "table"]
    truth_tables = [e for e in demo_groundtruth.elements if e.cls == "table"]
    for item, element in zip(tables, truth_tables, strict=True):
        assert [c.text for c in item.cells or []] == [c.text for c in element.cells or []]
        assert item.text == " ".join(c.text for c in item.cells or [])
    assert tables[0].text.startswith("ID Category Observation 01 Intake Request entered")


def test_cell_boxes_are_fractions_of_the_page(demo_nutrient_raw: RawResult) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    table = next(item for item in pred.layout if item.cls == "table")
    assert table.cells is not None
    assert table.cells[0].text == "ID"
    assert table.cells[0].box == pytest.approx(
        (193.97609 / 1654, 451.23975 / 2339, 58.686935 / 1654, 40.293427 / 2339)
    )


def test_table_cells_keep_their_place_in_the_grid(demo_nutrient_raw: RawResult) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    table = next(item for item in pred.layout if item.cls == "table")
    places = [(c.row, c.column, c.rows, c.columns) for c in table.cells or []]
    assert places[:4] == [(0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1), (1, 0, 1, 1)]
    assert places == [(r, c, 1, 1) for r in range(38) for c in range(3)]


def test_a_merged_cell_keeps_its_span_and_an_odd_one_has_no_place(
    demo_nutrient_raw: RawResult,
) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    table = next(e for e in payload["output"]["elements"] if e["type"] == "table")
    merged, odd = sorted(table["cells"], key=lambda c: (c["row"], c["column"]))[:2]
    merged["colSpan"] = 2
    odd["rowSpan"] = 0
    raw = RawResult(provider="nutrient", document_id="x", payload=payload)
    item = next(item for item in NutrientProvider().normalize(raw).layout if item.cls == "table")
    assert item.cells is not None
    first, second = item.cells[:2]
    assert (first.text, first.row, first.column, first.rows, first.columns) == ("ID", 0, 0, 1, 2)
    assert (second.text, second.row, second.columns) == ("Category", None, None)


def test_a_table_cell_without_bounds_keeps_its_text_and_has_no_box(
    demo_nutrient_raw: RawResult,
) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    table = next(e for e in payload["output"]["elements"] if e["type"] == "table")
    del table["cells"][0]["bounds"]
    raw = RawResult(provider="nutrient", document_id="x", payload=payload)
    item = next(item for item in NutrientProvider().normalize(raw).layout if item.cls == "table")
    assert item.cells is not None
    assert (item.cells[0].text, item.cells[0].box) == ("ID", None)
    assert all(c.box is not None for c in item.cells[1:])


def test_a_legacy_barcode_element_becomes_a_machine_readable_item_with_its_value_as_text(
    demo_nutrient_raw: RawResult,
) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    element = payload["output"]["elements"][0]
    element.pop("role", None)
    element.update(type="barcode", value="https://example.com/inv/4417", format="QrCode")
    raw = RawResult(provider="nutrient", document_id="x", payload=payload)
    item = NutrientProvider().normalize(raw).layout[0]
    assert (item.cls, item.text) == (ElementClass.MACHINE_READABLE, "https://example.com/inv/4417")
    assert item.box is not None


def test_a_machine_readable_zone_keeps_its_format_and_check_digits_verdict(
    demo_nutrient_raw: RawResult,
) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    element = payload["output"]["elements"][0]
    element.pop("role", None)
    element.update(
        type="machineReadable", value="P<UTO\nL898", format="MrzTd3", checkDigitsValid=False
    )
    raw = RawResult(provider="nutrient", document_id="x", payload=payload)
    item = NutrientProvider().normalize(raw).layout[0]
    assert (item.cls, item.text) == (ElementClass.MACHINE_READABLE, "P<UTO\nL898")
    assert (item.format, item.check_digits_valid) == ("MrzTd3", False)


def _normalize_one(
    raw: RawResult, **changes: object
) -> list[tuple[ElementClass, Channel | None, str]]:
    payload = copy.deepcopy(raw.payload)
    element = next(e for e in payload["output"]["elements"] if e["type"] == "paragraph")
    element.update(changes)
    item = RawResult(provider="nutrient", document_id="x", payload=payload)
    layout = NutrientProvider().normalize(item).layout
    return [(i.cls, i.channel, i.text) for i in layout if i.id == element["id"]]


@pytest.mark.parametrize(
    ("role", "cls"),
    [
        ("Title", ElementClass.HEADING),
        ("SectionHeader", ElementClass.HEADING),
        ("Text", ElementClass.PARAGRAPH),
        (None, ElementClass.PARAGRAPH),
        ("ListItem", ElementClass.LIST_ITEM),
        ("Footnote", ElementClass.FOOTNOTE),
        ("Code", ElementClass.CODE),
        ("Caption", ElementClass.CAPTION),
    ],
)
def test_paragraph_roles_map_to_classes(
    demo_nutrient_raw: RawResult, role: str | None, cls: ElementClass
) -> None:
    [(found, channel, _)] = _normalize_one(demo_nutrient_raw, role=role, text="x")
    assert (found, channel) == (cls, Channel.BODY)


def _font_of(raw: RawResult, font: object) -> Font | None:
    payload = copy.deepcopy(raw.payload)
    element = next(e for e in payload["output"]["elements"] if e["type"] == "paragraph")
    element["font"] = font
    item = RawResult(provider="nutrient", document_id="x", payload=payload)
    layout = NutrientProvider().normalize(item).layout
    return next(i.font for i in layout if i.id == element["id"])


def test_an_element_font_becomes_the_item_font(demo_nutrient_raw: RawResult) -> None:
    font = {"family": "Arimo", "class": "sans", "size": 11.5}
    assert _font_of(demo_nutrient_raw, font) == Font(
        family="Arimo", font_class=FontClass.SANS, size=11.5
    )


@pytest.mark.parametrize(
    ("font", "expected"),
    [
        (None, None),
        ({}, None),
        ({"family": " ", "class": "gothic", "size": 0}, None),
        ({"class": "serif", "size": "12"}, Font(font_class=FontClass.SERIF)),
        ({"family": "Tinos", "size": 9}, Font(family="Tinos", size=9.0)),
        ({"class": ["sans"], "size": 10}, Font(size=10.0)),
        ({"class": " Sans ", "family": "Arial"}, Font(family="Arial", font_class=FontClass.SANS)),
    ],
)
def test_an_element_font_keeps_only_the_fields_it_gives(
    demo_nutrient_raw: RawResult, font: object, expected: Font | None
) -> None:
    assert _font_of(demo_nutrient_raw, font) == expected


def test_a_page_number_is_a_header_at_the_top_and_a_footer_below(
    demo_nutrient_raw: RawResult,
) -> None:
    page = {"pageIndex": 0, "pageNumber": 1, "width": 1000, "height": 2000}
    top = {"x": 450, "y": 40, "width": 100, "height": 30}
    bottom = {"x": 450, "y": 1900, "width": 100, "height": 30}
    [header] = _normalize_one(demo_nutrient_raw, role="PageNumber", text="3", bounds=top, page=page)
    [footer] = _normalize_one(
        demo_nutrient_raw, role="PageNumber", text="3", bounds=bottom, page=page
    )
    assert header[:2] == (ElementClass.HEADER, Channel.FURNITURE)
    assert footer[:2] == (ElementClass.FOOTER, Channel.FURNITURE)


@pytest.mark.parametrize("role", ["CheckboxSelected", "CheckboxUnselected"])
def test_a_checkbox_is_left_out_of_the_layout(demo_nutrient_raw: RawResult, role: str) -> None:
    assert _normalize_one(demo_nutrient_raw, role=role, text="I agree") == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"type": "picture", "altDescription": "A bar chart"},
            (ElementClass.FIGURE, "A bar chart"),
        ),
        ({"type": "formula", "latex": "E = mc^2"}, (ElementClass.FORMULA, "E = mc^2")),
        (
            {"type": "handwriting", "text": "Signed J. Roe"},
            (ElementClass.PARAGRAPH, "Signed J. Roe"),
        ),
    ],
)
def test_pictures_formulas_and_handwriting_map_to_classes(
    demo_nutrient_raw: RawResult, changes: dict[str, str], expected: tuple[ElementClass, str]
) -> None:
    [(cls, _, text)] = _normalize_one(demo_nutrient_raw, role="Text", **changes)
    assert (cls, text) == expected


def test_a_key_value_region_is_left_out_of_the_layout(demo_nutrient_raw: RawResult) -> None:
    assert _normalize_one(demo_nutrient_raw, role="Text", type="keyValueRegion", pairs=[]) == []


def test_an_unknown_type_is_an_error(demo_nutrient_raw: RawResult) -> None:
    with pytest.raises(ValueError, match="no class mapping for Nutrient type 'sticker'"):
        _normalize_one(demo_nutrient_raw, role="Text", type="sticker")


def test_footnote_ids_become_footnote_links(demo_nutrient_raw: RawResult) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    elements = payload["output"]["elements"]
    table = next(e for e in elements if e["type"] == "table")
    note = next(e for e in elements if e["type"] == "paragraph")
    table["footnoteIds"] = [note["id"], "dropped-element"]
    raw = RawResult(provider="nutrient", document_id="x", payload=payload)
    relations = NutrientProvider().normalize(raw).relations
    links = [(r.source, r.target) for r in relations if r.kind is RelationKind.FOOTNOTE_OF]
    assert links == [(note["id"], table["id"])]


CHART_HTML = (
    "<table><caption>Revenue by region (EUR m)</caption>"
    "<tr><th></th><th>2025</th><th>2026</th></tr>"
    "<tr><th>North</th><td>38.5</td><td>42.1</td></tr>"
    "<tr><th>South</th><td>n/a</td><td>31.0</td></tr></table>"
)


def _chart_of(raw: RawResult, **changes: object) -> Chart | None:
    payload = copy.deepcopy(raw.payload)
    element = next(e for e in payload["output"]["elements"] if e["type"] == "chart")
    element.update(changes)
    pred = NutrientProvider().normalize(RawResult("nutrient", "x", payload))
    return next(item.chart for item in pred.layout if item.id == element["id"])


def test_chart_data_comes_from_the_html_table(chart_nutrient_raw: RawResult) -> None:
    chart = _chart_of(chart_nutrient_raw, htmlTable=CHART_HTML, summary=None)
    assert chart is not None
    assert chart.title == "Revenue by region (EUR m)"
    assert [(s.name, [(p.label, p.value) for p in s.points]) for s in chart.series] == [
        ("2025", [("North", 38.5)]),
        ("2026", [("North", 42.1), ("South", 31.0)]),
    ]


@pytest.mark.parametrize(
    ("summary", "kind"),
    [
        ("Grouped bar chart comparing 2025 and 2026 across regions.", ChartKind.BAR),
        ("Line chart showing monthly energy cost per tonne.", ChartKind.LINE),
        ("A donut chart of market share.", ChartKind.DONUT),
        ("A heat map chart of sales by hour.", ChartKind.HEATMAP),
        ("Radial bar chart of scores", ChartKind.RADIAL_BAR),
        ("Error bar chart of means", ChartKind.ERRORBAR),
        ("Step line chart of rates", ChartKind.STEP),
        ("Combo chart with a bar chart and a line chart", ChartKind.COMBO),
        ("Sunburst chart of spend", ChartKind.SUNBURST),
        ("A box and whisker plot of delivery times.", ChartKind.BOX),
        ("Nightingale rose chart of deaths by month.", ChartKind.ROSE),
        ("A histogram of ages.", ChartKind.HISTOGRAM),
        ("A streamgraph of plays by genre.", ChartKind.STREAM),
        ("Revenue grew in every region.", None),
        (None, None),
    ],
)
def test_the_chart_type_comes_from_the_agentic_summary(
    chart_nutrient_raw: RawResult, summary: str | None, kind: ChartKind | None
) -> None:
    chart = _chart_of(chart_nutrient_raw, htmlTable=CHART_HTML, summary=summary)
    assert chart is not None
    assert chart.kind is kind


def test_a_groundtruth_chart_needs_a_type() -> None:
    chart = {"series": [{"name": "Revenue", "points": [{"label": "Q1", "value": 3}]}]}
    with pytest.raises(ValidationError, match="chart data needs a kind"):
        truth({"id": "c", "class": "chart", "page": 0, "box": CHART_BOX, "chart": chart})


def _continuation_links(raw: RawResult) -> list[tuple[str, str]]:
    relations = NutrientProvider().normalize(raw).relations
    return [(r.source, r.target) for r in relations if r.kind is RelationKind.CONTINUES_IN]


def test_a_link_given_by_both_pieces_is_one_link(demo_nutrient_raw: RawResult) -> None:
    links = _continuation_links(demo_nutrient_raw)
    assert len(links) == len(set(links)) == 2


@pytest.mark.parametrize("side", ["continuesTo", "continuedFrom"])
def test_a_link_given_by_one_piece_only_is_kept(demo_nutrient_raw: RawResult, side: str) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    for e in payload["output"]["elements"]:
        e.pop(side, None)
    raw = RawResult(provider="nutrient", document_id="x", payload=payload)
    assert _continuation_links(raw) == _continuation_links(demo_nutrient_raw)


def test_a_hyphen_at_the_end_of_the_first_piece_marks_the_link_hyphenated(
    demo_nutrient_raw: RawResult,
) -> None:
    relations = NutrientProvider().normalize(demo_nutrient_raw).relations
    links = [r for r in relations if r.kind is RelationKind.CONTINUES_IN]
    layout = {i.id: i for i in NutrientProvider().normalize(demo_nutrient_raw).layout}
    assert {(layout[r.source].cls, r.hyphenated) for r in links} == {
        (ElementClass.TABLE, False),
        (ElementClass.PARAGRAPH, True),
    }


SERIES_DOWN = (
    "<table><tr><th>Region</th><th>2025</th><th>2026</th></tr>"
    "<tr><th>North</th><td>10</td><td>20</td></tr>"
    "<tr><th>South</th><td>30</td><td>40</td></tr></table>"
)
SERIES_ACROSS = (
    "<table><tr><th>Year</th><th>North</th><th>South</th></tr>"
    "<tr><th>2025</th><td>10</td><td>30</td></tr>"
    "<tr><th>2026</th><td>20</td><td>40</td></tr></table>"
)


@pytest.mark.parametrize("table", [SERIES_DOWN, SERIES_ACROSS], ids=["down", "across"])
def test_a_chart_scores_the_same_with_its_series_in_rows_or_columns(
    chart_nutrient_raw: RawResult, table: str
) -> None:
    payload = copy.deepcopy(chart_nutrient_raw.payload)
    element = next(e for e in payload["output"]["elements"] if e["type"] == "chart")
    element.update(htmlTable=table, summary="Grouped bar chart of North and South.")
    payload["output"]["markdown"] = table
    pred = NutrientProvider().normalize(RawResult("nutrient", "x", payload))
    item = next(i for i in pred.layout if i.id == element["id"])
    series = [
        ChartSeries(
            name=name, points=[ChartPoint(label="2025", value=a), ChartPoint(label="2026", value=b)]
        )
        for name, a, b in (("North", 10, 20), ("South", 30, 40))
    ]
    chart = Element(
        id="c",
        cls=ElementClass.CHART,
        page=item.page,
        box=item.box,
        chart=Chart(kind=ChartKind.BAR, series=series),
    )
    t = groundtruth([chart])
    result = T6.evaluate(t, DocContext(t, pred))
    assert result.sub_scores["data_points"] == 1.0
    assert result.sub_scores["series"] == 1.0


def test_a_chart_whose_table_is_in_the_markdown_leaves_the_series_to_t6(
    chart_nutrient_raw: RawResult,
) -> None:
    payload = copy.deepcopy(chart_nutrient_raw.payload)
    payload["output"]["markdown"] += "\n\n" + CHART_HTML + "\n"
    chart = _chart_of(RawResult("nutrient", "x", payload), htmlTable=CHART_HTML, summary=None)
    assert chart is not None
    assert chart.series == []
    assert chart.title == "Revenue by region (EUR m)"


def test_a_table_only_in_the_json_reads_its_cells_as_t6_does(chart_nutrient_raw: RawResult) -> None:
    table = (
        "<table><tr><th></th><th>Revenue</th></tr>"
        "<tr><th>Q1</th><td>1,234</td></tr><tr><th>Q2</th><td>45%</td></tr>"
        "<tr><th>Q3</th><td>$12</td></tr><tr><th>Q4</th><td>−3</td></tr></table>"
    )
    chart = _chart_of(chart_nutrient_raw, htmlTable=table, summary=None)
    assert chart is not None
    assert [(p.label, p.value) for p in chart.series[0].points] == [
        ("Q1", 1234.0),
        ("Q2", 45.0),
        ("Q3", 12.0),
        ("Q4", -3.0),
    ]


SECRET = "zebra quartz 9931"


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "CheckboxSelected", "text": f"I agree {SECRET}"},
        {"type": "picture", "altDescription": f"A photo of a card that reads {SECRET}"},
        {
            "type": "keyValueRegion",
            "pairs": [{"key": {"value": "Card"}, "value": {"value": SECRET}}],
        },
    ],
    ids=["checkbox", "picture description", "key-value pair"],
)
def test_text_left_out_of_the_markdown_is_kept_in_the_prediction(
    demo_nutrient_raw: RawResult, changes: dict[str, object]
) -> None:
    payload = copy.deepcopy(demo_nutrient_raw.payload)
    element = next(e for e in payload["output"]["elements"] if e["type"] == "paragraph")
    element.update(changes)
    payload["output"]["markdown"] = "Pay the invoice by Friday.\n"
    pred = NutrientProvider().normalize(RawResult("nutrient", "x", payload))
    texts = [*pred.unscored_text, *(item.text for item in pred.layout)]
    assert any(SECRET in text for text in texts)


def test_usage_is_the_billed_credits(demo_nutrient_raw: RawResult) -> None:
    assert NutrientProvider().usage(demo_nutrient_raw) == {"credits": 27}


def test_a_response_without_credits_gives_its_pages(demo_nutrient_raw: RawResult) -> None:
    payload = {k: v for k, v in demo_nutrient_raw.payload.items() if k != "usage"}
    assert NutrientProvider().usage(RawResult("nutrient", "x", payload)) == {"pages": 3}


def test_a_response_without_credits_or_pages_has_no_usage(demo_nutrient_raw: RawResult) -> None:
    payload = {k: v for k, v in demo_nutrient_raw.payload.items() if k not in ("usage", "metrics")}
    with pytest.raises(ValueError):
        NutrientProvider().usage(RawResult("nutrient", "x", payload))


@pytest.mark.parametrize(
    ("classification", "summary", "kind"),
    [
        ("bar_chart", None, ChartKind.BAR),
        ("line_chart", None, ChartKind.LINE),
        ("scatter_chart", None, ChartKind.SCATTER),
        ("donut_chart", None, ChartKind.DONUT),
        ("radial_bar_chart", None, ChartKind.RADIAL_BAR),
        ("line_chart", "Grouped bar chart comparing 2025 and 2026.", ChartKind.LINE),
        ("other", "Line chart of monthly cost.", ChartKind.LINE),
        ("flow_chart", None, None),
        ("gauge_chart", None, None),
        ("", None, None),
        (None, None, None),
        (42, None, None),
    ],
)
def test_the_chart_type_comes_from_the_classification(
    chart_nutrient_raw: RawResult,
    classification: object,
    summary: str | None,
    kind: ChartKind | None,
) -> None:
    chart = _chart_of(
        chart_nutrient_raw, htmlTable=CHART_HTML, summary=summary, classification=classification
    )
    assert chart is not None
    assert chart.kind is kind
