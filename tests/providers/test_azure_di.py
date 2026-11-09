from dataclasses import replace
from typing import Any

import pytest

from wholedoc_bench.providers import AzureDIProvider, RawResult
from wholedoc_bench.providers.azure_di import FEATURES
from wholedoc_bench.providers.azure_di import _font as azure_font
from wholedoc_bench.schema import Channel, ElementClass, Font, FontClass, Prediction, RelationKind
from wholedoc_bench.scorer.utils.blocks import markdown_text, parse_markdown, parse_markdown_tables

RUNNING_HEADER = "Northwind Logistics - Annual Review 2026"
BOX = (1.0, 1.0, 4.0, 1.5)


class Doc:
    def __init__(self, pages: int = 1) -> None:
        self.content = ""
        self.pages: list[dict[str, Any]] = [
            {"pageNumber": n, "width": 8.5, "height": 11.0, "unit": "inch", "spans": []}
            for n in range(1, pages + 1)
        ]
        self.paragraphs: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self.figures: list[dict[str, Any]] = []

    def spans(self, markdown: str, page: int = 1) -> list[dict[str, int]]:
        if self.content:
            self.content += "\n\n"
        span = {"offset": len(self.content), "length": len(markdown)}
        self.content += markdown
        page_spans = self.pages[page - 1]["spans"]
        if not page_spans:
            page_spans.append(dict(span))
        page_spans[0]["length"] = len(self.content) - page_spans[0]["offset"]
        return [span]

    def paragraph(
        self,
        text: str,
        role: str | None = None,
        *,
        page: int = 1,
        box: tuple[float, ...] | None = BOX,
        markdown: str | None = None,
    ) -> str:
        paragraph: dict[str, Any] = {"content": text, "spans": self.spans(markdown or text, page)}
        if box is not None:
            paragraph["boundingRegions"] = [region(page, box)]
        if role:
            paragraph["role"] = role
        self.paragraphs.append(paragraph)
        return f"/paragraphs/{len(self.paragraphs) - 1}"

    def raw(self, **changes: Any) -> RawResult:
        result = {
            "apiVersion": "2024-11-30",
            "modelId": "prebuilt-layout",
            "stringIndexType": "unicodeCodePoint",
            "contentFormat": "markdown",
            "content": self.content,
            "pages": self.pages,
            "paragraphs": self.paragraphs,
            "tables": self.tables,
            "figures": self.figures,
            **changes,
        }
        return RawResult("azure-di", "d", {"status": "succeeded", "analyzeResult": result})


def region(page: int, box: tuple[float, ...] = BOX) -> dict[str, Any]:
    left, top, right, bottom = box
    return {"pageNumber": page, "polygon": [left, top, right, top, right, bottom, left, bottom]}


def cell(row: int, text: str, paragraph: str, page: int = 1) -> dict[str, Any]:
    return {
        "rowIndex": row,
        "columnIndex": 0,
        "content": text,
        "boundingRegions": [region(page)],
        "elements": [paragraph],
    }


def normalize(raw: RawResult) -> Prediction:
    return checked(AzureDIProvider().normalize(raw))


def checked(pred: Prediction) -> Prediction:
    ids = [i.id for i in pred.layout]
    assert len(ids) == len(set(ids))
    assert {i for r in pred.relations for i in (r.source, r.target)} <= set(ids)
    return pred


def by_text(prediction: Prediction) -> dict[str, Any]:
    return {item.text: item for item in prediction.layout}


def styled(markdown: str) -> dict[str, list[str]]:
    runs: dict[str, list[str]] = {}
    for block in parse_markdown(markdown).blocks:
        chars = list(zip(block.text, block.styles, strict=True))
        for name in sorted({n for _, names in chars for n in names}):
            marked = "".join(c if name in names else "\0" for c, names in chars)
            runs.setdefault(name, []).extend(r.strip() for r in marked.split("\0") if r.strip())
    return runs


def style(kind: str, value: str, *spans: tuple[int, int]) -> dict[str, Any]:
    return {kind: value, "spans": [{"offset": o, "length": n} for o, n in spans]}


def test_the_markdown_is_azures_own(two_column_azure_raw: RawResult) -> None:
    pred = normalize(two_column_azure_raw)
    assert pred.markdown == two_column_azure_raw.payload["analyzeResult"]["content"]
    heading = by_text(pred)["Annual Review 2026"]
    assert pred.markdown[heading.md_start : heading.md_end] == "# Annual Review 2026"


def test_layout_follows_the_markdown_with_azures_classes(two_column_azure_raw: RawResult) -> None:
    pred = normalize(two_column_azure_raw)
    assert [i.cls for i in pred.layout if i.page == 0 and i.channel is Channel.BODY] == [
        ElementClass.HEADING,
        ElementClass.PARAGRAPH,
        ElementClass.HEADING,
        ElementClass.PARAGRAPH,
        ElementClass.PARAGRAPH,
        ElementClass.PARAGRAPH,
        ElementClass.HEADING,
        ElementClass.PARAGRAPH,
        ElementClass.PARAGRAPH,
    ]
    assert all(i.md_start is not None for i in pred.layout)
    footnote = by_text(pred)["* Figures are unaudited and rounded to one decimal."]
    assert pred.markdown[footnote.md_start :].startswith("\\* Figures")


def test_running_headers_and_page_numbers_are_furniture_in_comments(
    two_column_azure_raw: RawResult,
) -> None:
    pred = normalize(two_column_azure_raw)
    furniture = [(i.page, i.cls, i.text) for i in pred.layout if i.channel is Channel.FURNITURE]
    assert [(p, t) for p, cls, t in furniture if cls is ElementClass.HEADER] == [
        (0, RUNNING_HEADER),
        (1, RUNNING_HEADER),
        (2, RUNNING_HEADER),
    ]
    assert [(p, t) for p, cls, t in furniture if cls is ElementClass.FOOTER] == [
        (0, "1"),
        (1, "2"),
        (2, "3"),
    ]
    header = next(i for i in pred.layout if i.text == RUNNING_HEADER)
    assert pred.markdown[header.md_start : header.md_end].startswith("<!-- PageHeader=")
    assert RUNNING_HEADER not in parse_markdown(pred.markdown).document_text


def test_table_captions_and_footnotes_are_linked_to_their_tables(
    two_column_azure_raw: RawResult,
) -> None:
    pred = normalize(two_column_azure_raw)
    items = {i.id: i for i in pred.layout}
    links = [(r.kind, items[r.source].text, items[r.target].page) for r in pred.relations]
    assert links == [
        (RelationKind.CAPTION_OF, "Table 1: Results by region, 2026", 1),
        (RelationKind.FOOTNOTE_OF, "* Figures are unaudited and rounded to one decimal.", 2),
    ]
    assert by_text(pred)["Table 1: Results by region, 2026"].cls is ElementClass.CAPTION
    assert by_text(pred)[links[1][1]].cls is ElementClass.FOOTNOTE


def test_tables_keep_each_cell_once_with_its_box(two_column_azure_raw: RawResult) -> None:
    pred = normalize(two_column_azure_raw)
    tables = parse_markdown_tables(pred.markdown)
    assert [len(t.rows) for t in tables] == [10, 4]
    items = [i for i in pred.layout if i.cls is ElementClass.TABLE]
    assert [len(i.cells or []) for i in items] == [40, 16]
    assert all(cell.box is not None for i in items for cell in i.cells or [])
    assert [(c.row, c.column) for c in items[1].cells or []] == [
        (r, c) for r in range(4) for c in range(4)
    ]
    assert items[0].text.startswith("Region Revenue (EUR m) Change On time North 42.1")
    for item, table in zip(items, tables, strict=True):
        assert item.md_start == table.start
    assert "North" not in by_text(pred)


def test_page_offsets_start_after_each_page_break(two_column_azure_raw: RawResult) -> None:
    pred = normalize(two_column_azure_raw)
    assert pred.page_offsets is not None and pred.page_offsets[0] == 0
    for offset in pred.page_offsets[1:]:
        assert pred.markdown[:offset].rstrip().endswith("<!-- PageBreak -->")
        assert pred.markdown[offset:].lstrip().startswith("<!-- PageHeader=")
    assert pred.unscored_text == []


def test_figures_hold_their_text_and_their_caption(charts_azure_raw: RawResult) -> None:
    pred = normalize(charts_azure_raw)
    figures = [i for i in pred.layout if i.cls is ElementClass.FIGURE]
    assert [(i.page, i.text.split()[:3]) for i in figures] == [
        (0, ["50", "42.1", "40-"]),
        (1, ["Energy", "cost", "per"]),
    ]
    assert all(pred.markdown[i.md_start :].startswith("<figure>") for i in figures)
    assert "North" not in by_text(pred) and "42.1" not in by_text(pred)
    items = {i.id: i for i in pred.layout}
    assert [(items[r.source].text, items[r.target].cls) for r in pred.relations] == [
        ("Figure 1: Revenue by region, 2025 and 2026.", ElementClass.FIGURE),
        ("Table 1: Revenue by region (EUR m)", ElementClass.TABLE),
        ("Figure 2: Monthly energy cost per tonne moved.", ElementClass.FIGURE),
    ]


def test_a_figure_left_out_of_the_markdown_leaves_its_paragraphs_as_items() -> None:
    doc = Doc(pages=2)
    heading = doc.paragraph("Inside the figure", "sectionHeading", markdown="## Inside the figure")
    doc.paragraph("Next page", page=2)
    doc.figures.append({"id": "1.1", "boundingRegions": [region(1)], "elements": [heading]})
    pred = normalize(doc.raw())
    assert [(i.page, i.cls, i.text) for i in pred.layout] == [
        (0, ElementClass.HEADING, "Inside the figure"),
        (0, ElementClass.FIGURE, ""),
        (1, ElementClass.PARAGRAPH, "Next page"),
    ]


def test_a_figure_footnote_is_linked_to_its_figure() -> None:
    doc = Doc()
    label = doc.paragraph("42.1")
    note = doc.paragraph("Source: annual accounts.")
    figure_spans = [{"offset": 0, "length": len(doc.content)}]
    doc.figures.append(
        {
            "id": "1.1",
            "boundingRegions": [region(1)],
            "spans": figure_spans,
            "elements": [label],
            "footnotes": [{"content": "Source: annual accounts.", "elements": [note]}],
        }
    )
    pred = normalize(doc.raw())
    assert [(i.cls, i.text) for i in pred.layout] == [
        (ElementClass.FIGURE, "42.1"),
        (ElementClass.FOOTNOTE, "Source: annual accounts."),
    ]
    [link] = pred.relations
    assert (link.kind, link.source, link.target) == (
        RelationKind.FOOTNOTE_OF,
        "paragraphs/1",
        "figures/0",
    )


@pytest.mark.parametrize(("top", "cls"), [(0.3, ElementClass.HEADER), (10.2, ElementClass.FOOTER)])
def test_a_page_number_is_a_header_or_footer_by_position(top: float, cls: ElementClass) -> None:
    doc = Doc()
    doc.paragraph(
        "7", "pageNumber", box=(4.0, top, 4.2, top + 0.2), markdown='<!-- PageNumber="7" -->'
    )
    item = normalize(doc.raw()).layout[0]
    assert (item.cls, item.channel) == (cls, Channel.FURNITURE)


def test_a_caption_azure_also_calls_a_running_header_stays_a_header() -> None:
    doc = Doc()
    header = doc.paragraph("Annual Review", "pageHeader", markdown='<!-- PageHeader="x" -->')
    north = doc.paragraph("North")
    doc.tables.append(
        {
            "cells": [cell(0, "North", north)],
            "caption": {"content": "Annual Review", "elements": [header]},
            "boundingRegions": [region(1)],
        }
    )
    pred = normalize(doc.raw())
    assert [(i.cls, i.channel) for i in pred.layout if i.text == "Annual Review"] == [
        (ElementClass.HEADER, Channel.FURNITURE)
    ]
    assert pred.relations == []


def test_a_table_cell_on_a_later_page_has_no_box() -> None:
    doc = Doc(pages=2)
    north = doc.paragraph("North", page=1)
    south = doc.paragraph("South", page=2)
    doc.tables.append(
        {
            "cells": [cell(0, "North", north, page=1), cell(1, "South", south, page=2)],
            "boundingRegions": [region(1), region(2)],
            "spans": doc.spans("<table>"),
        }
    )
    [table] = normalize(doc.raw()).layout
    assert (table.page, table.text) == (0, "North South")
    assert [cell.box is not None for cell in table.cells or []] == [True, False]


def test_a_merged_table_cell_keeps_its_span() -> None:
    doc = Doc()
    region_ = doc.paragraph("Region")
    north = doc.paragraph("North")
    doc.tables.append(
        {
            "cells": [{**cell(0, "Region", region_), "columnSpan": 2}, cell(1, "North", north)],
            "boundingRegions": [region(1)],
            "spans": doc.spans("<table>"),
        }
    )
    [table] = normalize(doc.raw()).layout
    places = [(c.text, c.row, c.column, c.rows, c.columns) for c in table.cells or []]
    assert places == [("Region", 0, 0, 1, 2), ("North", 1, 0, 1, 1)]


def test_an_element_without_regions_takes_its_page_from_the_markdown() -> None:
    doc = Doc(pages=2)
    doc.paragraph("First page", box=None, page=1)
    doc.paragraph("Second page", box=None, page=2)
    pred = normalize(doc.raw())
    assert [(i.page, i.box) for i in pred.layout] == [(0, None), (1, None)]


def test_boxes_are_page_fractions_clipped_like_every_provider() -> None:
    doc = Doc()
    doc.paragraphs.append(
        {
            "content": "Tilted",
            "spans": doc.spans("Tilted"),
            "boundingRegions": [
                {"pageNumber": 1, "polygon": [1.0, 1.1, 4.0, 1.0, 4.1, 1.5, 1.1, 1.6]}
            ],
        }
    )
    doc.paragraph("Just past the edge", box=(1.0, 1.0, 8.55, 2.0))
    doc.paragraph("Far past the edge", box=(1.0, 1.0, 9.5, 2.0))
    boxes = [i.box for i in normalize(doc.raw()).layout]
    assert boxes[0] == pytest.approx((1.0 / 8.5, 1.0 / 11, 3.1 / 8.5, 0.6 / 11))
    assert boxes[1] is not None and boxes[1][0] + boxes[1][2] == 1.0
    assert boxes[2] is None


def test_pages_without_markdown_start_where_the_next_page_starts() -> None:
    doc = Doc(pages=3)
    doc.paragraph("First", page=1)
    doc.paragraph("Third", page=3)
    pred = normalize(doc.raw())
    assert pred.page_offsets == [0, pred.markdown.index("Third"), pred.markdown.index("Third")]


def test_an_unknown_role_is_an_error() -> None:
    doc = Doc()
    doc.paragraph("Signature", "signature")
    with pytest.raises(ValueError, match="no class mapping for Azure role 'signature'"):
        AzureDIProvider().normalize(doc.raw())


@pytest.mark.parametrize(
    "changes", [{"contentFormat": "text"}, {"stringIndexType": "textElements"}]
)
def test_text_output_or_other_span_units_are_an_error(changes: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="expected Markdown with code point spans"):
        AzureDIProvider().normalize(Doc().raw(**changes))


def test_barcodes_are_items_and_their_payloads_are_in_the_markdown(
    depot_azure_raw: RawResult,
) -> None:
    pred = normalize(depot_azure_raw)
    barcodes = [i for i in pred.layout if i.cls is ElementClass.MACHINE_READABLE]
    assert [(i.id, i.text) for i in barcodes] == [
        ("barcodes/1.1", "https://example.com/hazard/7731"),
        ("barcodes/1.2", "DLV-58204"),
    ]
    assert all(i.box is not None for i in barcodes)
    for item in barcodes:
        assert pred.markdown[item.md_start : item.md_end].endswith(f'"{item.text}")')
    assert not any(":barcode:" in i.text for i in pred.layout)


def test_a_formula_block_is_a_formula_with_its_latex(depot_azure_raw: RawResult) -> None:
    pred = normalize(depot_azure_raw)
    [formula] = [i for i in pred.layout if i.cls is ElementClass.FORMULA]
    assert formula.text == "d = v ^ { 2 } / \\left( 2 u g \\right)"
    assert pred.markdown[formula.md_start : formula.md_end] == f"$${formula.text}$$"


def test_bold_and_italic_spans_are_written_as_the_scorer_reads_them(
    depot_azure_raw: RawResult,
) -> None:
    pred = normalize(depot_azure_raw)
    assert styled(pred.markdown) == {
        "bold": ["Depot Safety Notice", "high-visibility vest", "red line", "four"],
        "italic": ["side mirrors", "four", "times"],
    }
    assert pred.markdown.startswith("# <b>Depot Safety Notice</b>\n")
    assert "<!-- PageFooter=" in pred.markdown and "<b>Northwind" not in pred.markdown


def test_markdown_spans_and_page_offsets_follow_the_style_tags(
    depot_azure_raw: RawResult,
) -> None:
    pred = normalize(depot_azure_raw)
    for item in pred.layout:
        if item.cls in (ElementClass.HEADING, ElementClass.PARAGRAPH):
            read = markdown_text(pred.markdown[item.md_start : item.md_end])
            assert " ".join(read.split()) == item.text
    assert pred.page_offsets == [0]


def test_placeholders_in_a_paragraph_take_their_values() -> None:
    doc = Doc()
    doc.paragraph(
        "The area is :formula: and the box is :selected:, not :unselected:.",
        markdown="The area is $\\pi r ^ { 2 }$ and the box is \u2612, not \u2610.",
    )
    doc.pages[0]["formulas"] = [
        {"kind": "inline", "value": "\\pi r ^ { 2 }", "span": {"offset": 12, "length": 16}}
    ]
    doc.pages[0]["selectionMarks"] = [
        {"state": "selected", "span": {"offset": 40, "length": 1}},
        {"state": "unselected", "span": {"offset": 47, "length": 1}},
    ]
    [item] = normalize(doc.raw()).layout
    assert item.text == "The area is \\pi r ^ { 2 } and the box is \u2612, not \u2610."


def test_placeholder_words_printed_on_the_page_stay_as_printed() -> None:
    doc = Doc()
    doc.paragraph("status:selected:", markdown="status:selected:")
    doc.paragraph("Type :formula: for :formula:", markdown="Type :formula: for $x ^ { 2 }$")
    start = doc.content.index("$x")
    doc.pages[0]["formulas"] = [
        {"kind": "inline", "value": "x ^ { 2 }", "span": {"offset": start, "length": 11}}
    ]
    assert [i.text for i in normalize(doc.raw()).layout] == [
        "status:selected:",
        "Type :formula: for x ^ { 2 }",
    ]


def test_style_tags_stay_out_of_furniture_tables_and_formulas() -> None:
    doc = Doc()
    doc.paragraph("Running head", "pageHeader", markdown='<!-- PageHeader="Running head" -->')
    north = doc.paragraph("North", markdown="<table><tr><td>North</td></tr></table>")
    doc.tables.append({"cells": [cell(0, "North", north)], "spans": doc.paragraphs[-1]["spans"]})
    doc.paragraph(":formula:", "formulaBlock", markdown="$$x = 1$$")
    doc.paragraph("Plain words")
    everything = [(0, len(doc.content))]
    raw = doc.raw(styles=[style("fontWeight", "bold", *everything)])
    pred = normalize(raw)
    assert pred.markdown == doc.content.replace("Plain words", "<b>Plain words</b>")


def test_a_style_across_a_line_break_is_written_on_each_line() -> None:
    doc = Doc()
    doc.paragraph("bold across lines", markdown="bold across\nlines")
    pred = normalize(doc.raw(styles=[style("fontStyle", "italic", (0, 17))]))
    assert pred.markdown == "<i>bold across</i>\n<i>lines</i>"
    assert styled(pred.markdown) == {"italic": ["bold across", "lines"]}


def test_an_escape_stays_whole_inside_a_style() -> None:
    doc = Doc()
    doc.paragraph("* Note", markdown="\\* Note")
    pred = normalize(doc.raw(styles=[style("fontWeight", "bold", (1, 6))]))
    assert pred.markdown == "<b>\\* Note</b>"
    assert styled(pred.markdown) == {"bold": ["* Note"]}


def test_normal_weights_and_other_font_properties_add_no_tags() -> None:
    doc = Doc()
    doc.paragraph("Plain words")
    raw = doc.raw(
        styles=[
            style("fontWeight", "normal", (0, 11)),
            style("fontStyle", "normal", (0, 11)),
            style("color", "#ff0000", (0, 5)),
            {"isHandwritten": True, "spans": [{"offset": 0, "length": 11}]},
        ]
    )
    assert normalize(raw).markdown == "Plain words"


@pytest.mark.parametrize(
    ("markdown", "bold", "written"),
    [
        ("dir\\\\name end", (5, 4), "dir\\\\<b>name</b> end"),
        ("a \\*", (0, 3), "<b>a</b> \\*"),
        ("x \\*y", (3, 2), "x <b>\\*y</b>"),
    ],
)
def test_a_tag_never_splits_an_escape(markdown: str, bold: tuple[int, int], written: str) -> None:
    doc = Doc()
    doc.paragraph(markdown, markdown=markdown)
    pred = normalize(doc.raw(styles=[style("fontWeight", "bold", bold)]))
    assert pred.markdown == written
    assert parse_markdown(pred.markdown).blocks[0].text == markdown_text(markdown)


def test_style_tags_go_around_an_inline_formula_and_skip_a_barcode() -> None:
    doc = Doc()
    doc.paragraph("Area :formula: here", markdown="Area $\\pi r ^ { 2 }$ here")
    doc.pages[0]["formulas"] = [
        {"kind": "inline", "value": "\\pi r ^ { 2 }", "span": {"offset": 5, "length": 15}}
    ]
    image = '![QRCode](barcodes/1.1 "INV-4417")'
    doc.paragraph(":barcode:", markdown=image)
    doc.pages[0]["barcodes"] = [
        {"kind": "QRCode", "value": "INV-4417", "span": doc.paragraphs[-1]["spans"][0]}
    ]
    pred = normalize(doc.raw(styles=[style("fontWeight", "bold", (0, len(doc.content)))]))
    assert pred.markdown == f"<b>Area</b> $\\pi r ^ {{ 2 }}$ <b>here</b>\n\n{image}"


def test_a_style_over_a_heading_mark_leaves_the_mark_outside() -> None:
    doc = Doc()
    doc.paragraph("Outlook", "sectionHeading", markdown="## Outlook")
    pred = normalize(doc.raw(styles=[style("fontWeight", "bold", (0, 10))]))
    assert pred.markdown == "## <b>Outlook</b>"
    [heading] = parse_markdown(pred.markdown).blocks
    assert (heading.kind.value, heading.level, heading.text) == ("heading", 2, "Outlook")


def test_page_offsets_move_past_the_tags_before_them() -> None:
    doc = Doc(pages=2)
    doc.paragraph("First page", page=1)
    doc.paragraph("Second page", page=2)
    pred = normalize(doc.raw(styles=[style("fontStyle", "italic", (0, 5))]))
    assert pred.markdown.startswith("<i>First</i> page")
    assert pred.page_offsets == [0, pred.markdown.index("Second page")]
    [_, second] = pred.layout
    assert pred.markdown[second.md_start : second.md_end] == "Second page"


def test_a_paragraph_takes_the_similar_font_family_azure_gives_most_of_it(
    depot_azure_raw: RawResult,
) -> None:
    fonts = [i.font for i in normalize(depot_azure_raw).layout if i.font]
    assert fonts
    assert all(f.size is None for f in fonts)
    assert {f.font_class for f in fonts} == {FontClass.SANS}
    assert {f.family for f in fonts} <= {"Arial", "Segoe UI"}


@pytest.mark.parametrize(
    ("families", "expected"),
    [
        ("Arial, Helvetica, sans-serif", Font(family="Arial", font_class=FontClass.SANS)),
        ("'Times New Roman', serif", Font(family="Times New Roman", font_class=FontClass.SERIF)),
        ("Consolas", Font(family="Consolas")),
        ("monospace", Font(font_class=FontClass.MONO)),
        (" , ", None),
    ],
)
def test_a_css_font_list_gives_a_family_and_a_class(families: str, expected: Font | None) -> None:
    assert azure_font(families) == expected


@pytest.mark.parametrize(
    "sent,priced",
    [
        ("barcodes+formulas+ocrHighResolution+styleFont", set(FEATURES)),
        ("formulas", {"formulas"}),
        ("none", set()),
    ],
)
def test_usage_counts_each_page_once_and_again_per_feature_sent(sent: str, priced: set) -> None:
    version = f"azure-di prebuilt-layout; API 2024-11-30; features {sent}"
    raw = replace(Doc(pages=2).raw(), metadata={"version": version})
    usage = AzureDIProvider().usage(raw)
    assert usage == {"pages": 2, **{f"{f}_pages": 2 if f in priced else 0 for f in FEATURES}}


def test_a_response_whose_version_names_no_features_has_no_usage() -> None:
    with pytest.raises(ValueError, match="features sent"):
        AzureDIProvider().usage(Doc().raw())
