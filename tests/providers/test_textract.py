import itertools
from typing import Any

import pytest

from wholedoc_bench.providers import RawResult, TextractProvider
from wholedoc_bench.schema import Channel, ElementClass, Prediction, RelationKind
from wholedoc_bench.scorer.utils.blocks import BlockKind, parse_markdown, parse_markdown_tables

RUNNING_HEADER = "Northwind Logistics - Annual Review 2026"


class Page:
    ids = itertools.count(1)

    def __init__(self) -> None:
        self.blocks: list[dict[str, Any]] = [
            {"BlockType": "PAGE", "Id": "page", "Geometry": _geometry(0, 0, 1, 1)}
        ]

    def _add(self, kind: str, children: list[str] | None = None, **fields: Any) -> str:
        block = {"BlockType": kind, "Id": f"{kind.lower()}-{next(Page.ids)}", **fields}
        block.setdefault("Geometry", _geometry())
        if children:
            block["Relationships"] = [{"Type": "CHILD", "Ids": children}]
        self.blocks.append(block)
        return block["Id"]

    def words(self, text: str) -> list[str]:
        return [self._add("WORD", Text=word) for word in text.split()]

    def line(self, text: str, top: float = 0.1) -> str:
        return self._add("LINE", self.words(text), Text=text, Geometry=_geometry(top=top))

    def layout(self, kind: str, *texts: str, top: float = 0.1) -> str:
        return self._add(
            f"LAYOUT_{kind}", [self.line(t, top) for t in texts], Geometry=_geometry(top=top)
        )

    def layout_of(self, kind: str, children: list[str], top: float = 0.1) -> str:
        return self._add(f"LAYOUT_{kind}", children, Geometry=_geometry(top=top))

    def table(
        self, rows: list[list[str]], kinds: list[str | None] | None = None, top: float = 0.35
    ) -> tuple[str, dict]:
        kinds = kinds or ["COLUMN_HEADER"] + [None] * (len(rows) - 1)
        cells: dict[tuple[int, int], str] = {}
        for r, (row, kind) in enumerate(zip(rows, kinds, strict=True), start=1):
            for c, text in enumerate(row, start=1):
                entity = {"EntityTypes": [kind]} if kind else {}
                cells[r, c] = self._add(
                    "CELL",
                    self.words(text),
                    RowIndex=r,
                    ColumnIndex=c,
                    RowSpan=1,
                    ColumnSpan=1,
                    Geometry=_geometry(0.1 * c, top + 0.05 * (r - 1), 0.1, 0.05),
                    **entity,
                )
        table = self._add("TABLE", list(cells.values()), Geometry=_geometry(0.1, top))
        return table, cells

    def row_lines(self, cells: dict, top: float = 0.35) -> list[str]:
        lines = []
        for row in sorted({r for r, _ in cells}):
            words = [
                w
                for (r, _), cell in sorted(cells.items())
                if r == row
                for w in _children(self, cell)
            ]
            text = " ".join(self.block(w)["Text"] for w in words)
            geometry = _geometry(top=top + 0.05 * (row - 1))
            lines.append(self._add("LINE", words, Text=text, Geometry=geometry))
        return lines

    def block(self, block_id: str) -> dict[str, Any]:
        return next(b for b in self.blocks if b["Id"] == block_id)

    def note(self, table: str, kind: str, words: list[str], top: float) -> str:
        note = self._add(kind, words, Geometry=_geometry(top=top))
        self.relate(table, kind, [note])
        return note

    def relate(self, block_id: str, kind: str, ids: list[str]) -> None:
        block = self.block(block_id)
        block.setdefault("Relationships", []).append({"Type": kind, "Ids": ids})

    def merge(self, table: str, cells: list[str], row: int, column: int, rows: int, columns: int):
        merged = self._add(
            "MERGED_CELL", cells, RowIndex=row, ColumnIndex=column, RowSpan=rows, ColumnSpan=columns
        )
        self.relate(table, "MERGED_CELL", [merged])

    def response(self) -> dict[str, Any]:
        return {
            "AnalyzeDocumentModelVersion": "1.0",
            "DocumentMetadata": {"Pages": 1},
            "Blocks": self.blocks,
        }


def _geometry(left: float = 0.1, top: float = 0.1, width: float = 0.5, height: float = 0.04):
    return {"BoundingBox": {"Left": left, "Top": top, "Width": width, "Height": height}}


def normalize(*pages: Page) -> Prediction:
    payload = {"pages": [page.response() for page in pages]}
    raw = RawResult(provider="textract", document_id="d", payload=payload)
    return checked(TextractProvider().normalize(raw))


def checked(pred: Prediction) -> Prediction:
    ids = [i.id for i in pred.layout]
    assert len(ids) == len(set(ids))
    assert {i for r in pred.relations for i in (r.source, r.target)} <= set(ids)
    return pred


def by_text(prediction: Prediction) -> dict[str, Any]:
    return {item.text: item for item in prediction.layout}


def test_layout_follows_textract_reading_order_and_classes(
    two_column_textract_raw: RawResult,
) -> None:
    pred = TextractProvider().normalize(two_column_textract_raw)
    headings = [b.text for b in parse_markdown(pred.markdown).blocks if b.kind is BlockKind.HEADING]
    assert headings == [
        "Annual Review 2026",
        "Operations",
        "People",
        "Results by region",
        "Outlook",
    ]
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
    left = pred.markdown.index("The warehouse team rebuilt")
    right = pred.markdown.index("person, who checks them by hand")
    assert left < right < pred.markdown.index("## People")


def test_running_headers_and_page_numbers_are_furniture_outside_the_markdown(
    two_column_textract_raw: RawResult,
) -> None:
    pred = TextractProvider().normalize(two_column_textract_raw)
    furniture = [(i.page, i.cls, i.text) for i in pred.layout if i.channel is Channel.FURNITURE]
    assert (0, ElementClass.HEADER, RUNNING_HEADER) in furniture
    assert [(p, t) for p, cls, t in furniture if cls is ElementClass.FOOTER] == [
        (0, "1"),
        (1, "2"),
        (2, "3"),
    ]
    assert RUNNING_HEADER not in pred.markdown
    assert all(i.md_start is None for i in pred.layout if i.channel is Channel.FURNITURE)


def test_table_titles_and_footers_become_captions_and_footnotes(
    two_column_textract_raw: RawResult,
) -> None:
    pred = TextractProvider().normalize(two_column_textract_raw)
    items = {i.id: i for i in pred.layout}
    links = [(r.kind, items[r.source].text, items[r.target].page) for r in pred.relations]
    assert links == [
        (RelationKind.CAPTION_OF, "Table 1: Results by region, 2026", 1),
        (RelationKind.FOOTNOTE_OF, "* Figures are unaudited and rounded to one decimal.", 2),
    ]
    assert by_text(pred)["Table 1: Results by region, 2026"].cls is ElementClass.CAPTION
    assert all(i.cls is ElementClass.HEADER for i in pred.layout if i.text == RUNNING_HEADER)


def test_tables_are_html_with_header_cells_and_cell_boxes(
    two_column_textract_raw: RawResult,
) -> None:
    pred = TextractProvider().normalize(two_column_textract_raw)
    tables = parse_markdown_tables(pred.markdown)
    assert [len(t.rows) for t in tables] == [10, 4]
    assert tables[0].header == (("Region", "Revenue (EUR m)", "Change", "On time"),)
    assert tables[0].body[0] == ("North", "42.1", "+6.2%", "97.9%")
    items = [i for i in pred.layout if i.cls is ElementClass.TABLE]
    assert [len(i.cells or []) for i in items] == [40, 16]
    assert all(cell.box is not None for i in items for cell in i.cells or [])
    for item, table in zip(items, tables, strict=True):
        assert (item.md_start, item.md_end) == (table.start, table.end)


def test_markdown_spans_and_page_offsets(two_column_textract_raw: RawResult) -> None:
    pred = checked(TextractProvider().normalize(two_column_textract_raw))
    heading = by_text(pred)["Annual Review 2026"]
    assert pred.markdown[heading.md_start : heading.md_end] == "# Annual Review 2026"
    assert pred.page_offsets == [
        0,
        pred.markdown.index("# Results by region"),
        pred.markdown.index("<table>", pred.markdown.index("</table>")),
    ]
    assert pred.markdown.endswith("\n") and not pred.markdown.endswith("\n\n")
    assert pred.unscored_text == []


def test_lists_take_their_items_once_with_printed_numbers_kept() -> None:
    page = Page()
    bullets = [page.layout("TEXT", text) for text in ("• First point", "Second", "- Third")]
    page.layout_of("LIST", bullets)
    numbered = [page.layout("TEXT", text) for text in ("1. One", "2) Two")]
    page.layout_of("LIST", numbered)
    pred = normalize(page)
    assert pred.markdown == "- First point\n- Second\n- Third\n\n1. One\n2) Two\n"
    assert [i.cls for i in pred.layout] == [ElementClass.LIST_ITEM] * 5
    assert [i.text for i in pred.layout] == [
        "• First point",
        "Second",
        "- Third",
        "1. One",
        "2) Two",
    ]


def test_figure_text_stays_out_of_the_markdown() -> None:
    page = Page()
    page.layout("TEXT", "Before the chart.")
    page.layout("FIGURE", "50 40 30", "North South")
    pred = normalize(page)
    assert pred.markdown == "Before the chart.\n"
    figure = pred.layout[1]
    assert (figure.cls, figure.text, figure.md_start) == (
        ElementClass.FIGURE,
        "50 40 30 North South",
        None,
    )


def test_key_values_are_paragraphs_with_their_boxes() -> None:
    page = Page()
    page.layout("KEY_VALUE", "Invoice no: 4417")
    pred = normalize(page)
    assert pred.markdown == "Invoice no: 4417\n"
    [item] = pred.layout
    assert (item.cls, item.text, item.box is not None) == (
        ElementClass.PARAGRAPH,
        "Invoice no: 4417",
        True,
    )
    assert pred.markdown[item.md_start : item.md_end] == "Invoice no: 4417"
    assert pred.unscored_text == []


def test_an_empty_key_value_block_gives_no_item() -> None:
    page = Page()
    page.layout("KEY_VALUE")
    page.layout("TEXT", "Kept.", top=0.5)
    pred = normalize(page)
    assert [(i.cls, i.text) for i in pred.layout] == [(ElementClass.PARAGRAPH, "Kept.")]
    assert pred.markdown == "Kept.\n"


def test_lines_outside_every_layout_block_are_unscored_text() -> None:
    page = Page()
    page.layout("TEXT", "Kept.")
    page.line("Stray margin note")
    pred = normalize(page)
    assert pred.markdown == "Kept.\n"
    assert pred.unscored_text == ["Stray margin note"]


@pytest.mark.parametrize(("top", "cls"), [(0.02, ElementClass.HEADER), (0.95, ElementClass.FOOTER)])
def test_a_page_number_is_a_header_or_footer_by_position(top: float, cls: ElementClass) -> None:
    page = Page()
    page.layout("PAGE_NUMBER", "7", top=top)
    pred = normalize(page)
    assert [(i.cls, i.channel) for i in pred.layout] == [(cls, Channel.FURNITURE)]
    assert pred.markdown == ""


def test_lines_join_with_a_space_and_a_line_end_hyphen_joins_directly() -> None:
    page = Page()
    page.layout("TEXT", "A well-", "known fact -", "here")
    assert normalize(page).markdown == "A well-known fact - here\n"


@pytest.mark.parametrize(
    "printed",
    [
        "Type **name** here.",
        "snake_case, `code` and ~~strike~~",
        "a <b>bold</b> tag and <!-- not a comment -->",
        "&copy; &amp; &region=eu",
        "- not a bullet",
        "+ not a bullet",
        "1. not a list",
        "12) not a list",
        "# not a heading",
        "> not a quote",
        "---",
        "===",
        "Issue #",
        "[text](http://example.com) and ![alt](x.png)",
        "back\\slash \\* and a | b",
    ],
)
def test_printed_markdown_syntax_reads_back_as_printed(printed: str) -> None:
    page = Page()
    page.layout("TEXT", printed)
    page.layout("SECTION_HEADER", printed)
    table, _ = page.table([["Head"], [printed]])
    page.layout_of("TABLE", [table], top=0.35)
    pred = normalize(page)
    blocks = parse_markdown(pred.markdown).blocks
    assert [(b.kind, b.text) for b in blocks[:2]] == [
        (BlockKind.TEXT, printed),
        (BlockKind.HEADING, printed),
    ]
    assert parse_markdown_tables(pred.markdown)[0].body == ((printed,),)


def test_merged_cells_are_written_once_with_their_span() -> None:
    page = Page()
    table, cells = page.table(
        [["Revenue", ""], ["2025", "2026"], ["1", "2"]], ["COLUMN_HEADER", "COLUMN_HEADER", None]
    )
    page.merge(table, [cells[1, 1], cells[1, 2]], row=1, column=1, rows=1, columns=2)
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert '<th colspan="2">Revenue</th>' in pred.markdown
    read = parse_markdown_tables(pred.markdown)[0]
    assert read.header == (("Revenue", "Revenue"), ("2025", "2026"))
    assert [(m.row, m.column, m.rows, m.columns) for m in read.merged] == [(0, 0, 1, 2)]
    [table_item] = pred.layout
    places = [(c.text, c.row, c.column, c.rows, c.columns) for c in table_item.cells or []]
    assert places[:3] == [("Revenue", 0, 0, 1, 2), ("2025", 1, 0, 1, 1), ("2026", 1, 1, 1, 1)]


def test_a_header_row_with_an_empty_corner_stays_the_header() -> None:
    page = Page()
    table, cells = page.table([["", "2025", "2026"], ["North", "1", "2"]])
    del page.block(cells[1, 1])["EntityTypes"]
    page.layout_of("TABLE", [table])
    read = parse_markdown_tables(normalize(page).markdown)[0]
    assert read.header == (("", "2025", "2026"),)
    assert read.body == (("North", "1", "2"),)


def test_in_grid_title_and_footer_rows_become_a_caption_and_a_footnote() -> None:
    page = Page()
    table, cells = page.table(
        [["Revenue by region", ""], ["Region", "EUR"], ["North", "42"], ["Source: survey", ""]],
        ["TABLE_TITLE", "COLUMN_HEADER", None, "TABLE_FOOTER"],
    )
    page.merge(table, [cells[1, 1], cells[1, 2]], row=1, column=1, rows=1, columns=2)
    layout_table = page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert pred.markdown.startswith("Revenue by region\n\n<table>")
    assert pred.markdown.endswith("</table>\n\nSource: survey\n")
    assert parse_markdown_tables(pred.markdown)[0].rows == (("Region", "EUR"), ("North", "42"))
    caption, grid, footnote = pred.layout
    assert (caption.cls, caption.text) == (ElementClass.CAPTION, "Revenue by region")
    assert caption.box == pytest.approx((0.1, 0.35, 0.2, 0.05))
    assert (grid.id, grid.text, len(grid.cells or [])) == (layout_table, "Region EUR North 42", 4)
    places = [(c.text, c.row, c.column) for c in grid.cells or []]
    assert places == [("Region", 0, 0), ("EUR", 0, 1), ("North", 1, 0), ("42", 1, 1)]
    assert (footnote.cls, footnote.text) == (ElementClass.FOOTNOTE, "Source: survey")
    assert [(r.kind, r.source, r.target) for r in pred.relations] == [
        (RelationKind.CAPTION_OF, caption.id, layout_table),
        (RelationKind.FOOTNOTE_OF, footnote.id, layout_table),
    ]


def test_a_table_of_title_rows_only_stays_a_table() -> None:
    page = Page()
    table, _ = page.table([["Notes"], ["See annex"]], ["TABLE_TITLE", "TABLE_TITLE"])
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert parse_markdown_tables(pred.markdown)[0].rows == (("Notes",), ("See annex",))
    assert [(i.cls, len(i.cells or [])) for i in pred.layout] == [(ElementClass.TABLE, 2)]
    assert pred.relations == []


def test_text_repeating_an_in_grid_title_is_not_a_second_caption() -> None:
    page = Page()
    table, cells = page.table(
        [["Revenue"], ["Region"], ["North"]], ["TABLE_TITLE", "COLUMN_HEADER", None]
    )
    title_words = _children(page, cells[1, 1])
    page.relate(table, "TABLE_TITLE", [page._add("TABLE_TITLE", title_words)])
    page.layout_of("TEXT", [page._add("LINE", title_words, Text="Revenue")])
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert [i.cls for i in pred.layout] == [ElementClass.CAPTION, ElementClass.TABLE]
    assert [r.kind for r in pred.relations] == [RelationKind.CAPTION_OF]
    assert pred.markdown.count("Revenue") == 1


def test_a_row_span_is_written_once() -> None:
    page = Page()
    table, cells = page.table([["Region", "Year"], ["North", "2025"], ["", "2026"]])
    page.merge(table, [cells[2, 1], cells[3, 1]], row=2, column=1, rows=2, columns=1)
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert '<td rowspan="2">North</td>' in pred.markdown
    read = parse_markdown_tables(pred.markdown)[0]
    assert read.body == (("North", "2025"), ("North", "2026"))
    assert [c.text for c in pred.layout[0].cells or []] == [
        "Region",
        "Year",
        "North",
        "2025",
        "2026",
    ]
    assert [(m.row, m.column, m.rows, m.columns) for m in read.merged] == [(1, 0, 2, 1)]


def test_a_merged_cell_is_one_layout_cell_with_its_full_text_and_box() -> None:
    page = Page()
    table, cells = page.table([["Region", "Sales"], ["North", "America"], ["Europe", "12"]])
    page.merge(table, [cells[2, 1], cells[2, 2]], row=2, column=1, rows=1, columns=2)
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert '<td colspan="2">North America</td>' in pred.markdown
    item = pred.layout[0]
    assert [c.text for c in item.cells or []] == [
        "Region",
        "Sales",
        "North America",
        "Europe",
        "12",
    ]
    assert (item.cells or [])[2].box == pytest.approx((0.1, 0.1, 0.5, 0.04))
    assert item.text == "Region Sales North America Europe 12"


def test_mixed_lines_keep_only_their_words_outside_the_table() -> None:
    page = Page()
    table, cells = page.table([["Item", "Amount"], ["Total", "$500"]])
    header, total = ([w for c in (1, 2) for w in _children(page, cells[r, c])] for r in (1, 2))
    rows = [
        page._add("LINE", header, Text="Item Amount", Geometry=_geometry(top=0.4)),
        page._add(
            "LINE",
            total + page.words("Includes tax"),
            Text="Total $500 Includes tax",
            Geometry=_geometry(top=0.6),
        ),
    ]
    note = page._add(
        "LINE", header[:1] + page.words("prices exclude VAT"), Text="Item prices exclude VAT"
    )
    page.layout_of("TABLE", rows)
    page.layout_of("TEXT", [note], top=0.8)
    pred = normalize(page)
    assert pred.markdown.endswith("</table>\n\nIncludes tax\n\nprices exclude VAT\n")
    assert (
        pred.markdown.count("Total"),
        pred.markdown.count("$500"),
        pred.markdown.count("Item"),
    ) == (1, 1, 1)
    assert pred.layout[-1].text == "prices exclude VAT"


def test_a_selection_element_in_a_cell_adds_no_text() -> None:
    page = Page()
    table, cells = page.table([["Paid"], ["yes"]])
    mark = page._add("SELECTION_ELEMENT", SelectionStatus="SELECTED")
    page.block(cells[2, 1])["Relationships"][0]["Ids"].append(mark)
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert parse_markdown_tables(pred.markdown)[0].body == (("yes",),)


def test_blocks_without_geometry_have_no_box() -> None:
    page = Page()
    number = page.layout("PAGE_NUMBER", "7")
    table, _ = page.table([["A"], ["1"]])
    line = page.line("Units: EUR")
    for block_id in (number, table, line):
        del page.block(block_id)["Geometry"]
    page.layout_of("TABLE", [table, line])
    pred = normalize(page)
    assert [(i.cls, i.box) for i in pred.layout[:1]] == [(ElementClass.FOOTER, None)]
    assert pred.markdown.endswith("</table>\n\nUnits: EUR\n")


def test_the_words_of_a_table_are_written_once() -> None:
    page = Page()
    table, cells = page.table([["Region", "Value"], ["North", "42"]])
    rows = [[w for c in (1, 2) for w in _children(page, cells[r, c])] for r in (1, 2)]
    lines = [
        page._add("LINE", words, Text=text)
        for words, text in zip(rows, ["Region Value", "North 42"], strict=True)
    ]
    page.layout_of("SECTION_HEADER", [page._add("LINE", rows[0], Text="Region Value")])
    page.layout_of("TABLE", [lines[0]])
    page.layout_of("TABLE", [lines[1]])
    pred = normalize(page)
    assert (pred.markdown.count("Region"), pred.markdown.count("North")) == (1, 1)
    assert [(i.cls, i.md_start is None) for i in pred.layout] == [(ElementClass.TABLE, False)]


def test_a_layout_table_matches_the_table_sharing_its_words() -> None:
    page = Page()
    page.table([["A"], ["1"]])
    _, second_cells = page.table([["B"], ["2"]])
    words = [w for c in second_cells.values() for w in _children(page, c)]
    line = page._add("LINE", words, Text="B 2")
    page.layout_of("TABLE", [line])
    pred = normalize(page)
    tables = parse_markdown_tables(pred.markdown)
    assert [t.rows for t in tables] == [(("B",), ("2",)), (("A",), ("1",))]
    assert [c.text for c in pred.layout[0].cells or []] == ["B", "2"]


QTY = [["Qty", "Item", "Price"], ["2", "Chair", "120.00"], ["1", "Table", "340.00"]]


@pytest.mark.parametrize("kind", ["TEXT", "KEY_VALUE"])
def test_a_table_under_another_layout_block_is_still_a_table(kind: str) -> None:
    page = Page()
    page.layout("TEXT", "Before.", top=0.1)
    table, cells = page.table(QTY)
    region = page.layout_of(kind, page.row_lines(cells), top=0.35)
    page.layout("TEXT", "After.", top=0.8)
    pred = normalize(page)
    assert pred.markdown.startswith("Before.\n\n<table>")
    assert pred.markdown.endswith("</table>\n\nAfter.\n")
    assert [t.rows for t in parse_markdown_tables(pred.markdown)] == [tuple(tuple(r) for r in QTY)]
    assert "Qty Item Price" not in pred.markdown
    items = {i.id: i for i in pred.layout}
    assert (items[table].cls, len(items[table].cells or [])) == (ElementClass.TABLE, 9)
    assert region not in items
    assert pred.unscored_text == []


def test_a_block_keeps_its_own_text_around_the_table_it_holds() -> None:
    page = Page()
    table, cells = page.table(QTY, top=0.4)
    intro = page.line("Order summary:", top=0.3)
    note = page.line("Prices include VAT.", top=0.6)
    page.layout_of("TEXT", [intro, *page.row_lines(cells, top=0.4), note], top=0.3)
    pred = normalize(page)
    assert pred.markdown.startswith("Order summary:\n\n<table>")
    assert pred.markdown.endswith("</table>\n\nPrices include VAT.\n")
    paragraph = next(i for i in pred.layout if i.cls is ElementClass.PARAGRAPH)
    assert paragraph.text == "Order summary: Prices include VAT."
    assert paragraph.box == pytest.approx((0.1, 0.3, 0.5, 0.34))
    page = Page()
    table, cells = page.table(QTY, top=0.4)
    note = page.line("Prices include VAT.", top=0.6)
    page.layout_of("TEXT", [*page.row_lines(cells, top=0.4), note], top=0.4)
    assert normalize(page).markdown.endswith("</table>\n\nPrices include VAT.\n")


def test_a_table_region_without_its_own_table_leaves_held_tables_to_their_items() -> None:
    page = Page()
    named, _ = page.table([["Used"], ["1"]], top=0.1)
    page.layout_of("TABLE", [named], top=0.1)
    extra, cells = page.table(QTY, top=0.5)
    page.layout_of("TABLE", [named, *page.row_lines(cells, top=0.5)], top=0.5)
    pred = normalize(page)
    assert [i.cls for i in pred.layout] == [ElementClass.TABLE, ElementClass.TABLE]
    assert extra in {i.id for i in pred.layout}
    assert len(parse_markdown_tables(pred.markdown)) == 2


def test_a_table_region_with_two_tables_writes_both_in_page_order() -> None:
    page = Page()
    first, first_cells = page.table(QTY, top=0.3)
    second, second_cells = page.table([["Total"], ["460.00"]], top=0.6)
    region = page.layout_of(
        "TABLE", [*page.row_lines(second_cells, top=0.6), *page.row_lines(first_cells, top=0.3)]
    )
    pred = normalize(page)
    assert [t.rows[0] for t in parse_markdown_tables(pred.markdown)] == [
        ("Qty", "Item", "Price"),
        ("Total",),
    ]
    assert [(i.id, i.cls) for i in pred.layout] == [
        (region, ElementClass.TABLE),
        (second, ElementClass.TABLE),
    ]
    assert first not in {i.id for i in pred.layout}


@pytest.mark.parametrize("kind", ["TEXT", "TABLE"])
def test_text_between_two_tables_stays_between_them(kind: str) -> None:
    page = Page()
    _, first = page.table(QTY, top=0.3)
    _, second = page.table([["Total"], ["460.00"]], top=0.6)
    between = page.line("Shipping is free.", top=0.5)
    lines = [*page.row_lines(first, top=0.3), between, *page.row_lines(second, top=0.6)]
    page.layout_of(kind, lines, top=0.3)
    pred = normalize(page)
    assert len(parse_markdown_tables(pred.markdown)) == 2
    first_end, second_start = pred.markdown.index("</table>"), pred.markdown.rindex("<table>")
    assert first_end < pred.markdown.index("Shipping is free.") < second_start


def test_a_table_only_a_running_header_holds_stays_out_like_the_header() -> None:
    page = Page()
    table, cells = page.table([["ACME Invoice", "Page 1"]])
    page.layout_of("HEADER", page.row_lines(cells, top=0.02), top=0.02)
    page.layout("TEXT", "Body.", top=0.3)
    pred = normalize(page)
    assert pred.markdown == "Body.\n"
    assert table not in {i.id for i in pred.layout}


def test_a_table_inside_a_figure_is_written_as_a_table() -> None:
    page = Page()
    table, cells = page.table([["Region", "2025"], ["North", "38.5"]])
    figure = page.layout_of("FIGURE", [page.line("Revenue by region"), *page.row_lines(cells)])
    pred = normalize(page)
    rows = [t.rows for t in parse_markdown_tables(pred.markdown)]
    assert rows == [(("Region", "2025"), ("North", "38.5"))]
    assert "Revenue by region" not in pred.markdown
    assert {i.id: i.cls for i in pred.layout} == {
        figure: ElementClass.FIGURE,
        table: ElementClass.TABLE,
    }


def test_a_table_without_words_is_left_out_with_its_title() -> None:
    page = Page()
    title = page.line("Table 3: Empty", top=0.3)
    table, _ = page.table([[""], [""]])
    page.relate(table, "TABLE_TITLE", [page._add("TABLE_TITLE", _children(page, title))])
    page.layout_of("TEXT", [title], top=0.3)
    pred = normalize(page)
    assert (pred.markdown, pred.relations) == ("Table 3: Empty\n", [])
    assert [i.cls for i in pred.layout] == [ElementClass.PARAGRAPH]


def test_a_table_goes_where_most_of_its_words_are() -> None:
    page = Page()
    _, cells = page.table(QTY, top=0.5)
    lines = page.row_lines(cells, top=0.5)
    words = _children(page, lines[0])[:1] + page.words("of the order")
    page.layout_of("TEXT", [page._add("LINE", words, Text="Qty of the order")], top=0.1)
    page.layout("TEXT", "Middle.", top=0.3)
    page.layout_of("TEXT", lines, top=0.5)
    pred = normalize(page)
    assert pred.markdown.startswith("of the order\n\nMiddle.\n\n<table>")


@pytest.mark.parametrize("kind", ["TABLE", "TEXT"])
def test_title_and_footer_lines_around_a_grid_become_its_caption_and_footnote(kind: str) -> None:
    page = Page()
    title = page.line("Table 1: Orders", top=0.3)
    table, cells = page.table(QTY, top=0.4)
    footer = page.line("Source: ERP export", top=0.6)
    title_note = page.note(table, "TABLE_TITLE", _children(page, title), top=0.3)
    footer_note = page.note(table, "TABLE_FOOTER", _children(page, footer), top=0.6)
    region = page.layout_of(kind, [title, *page.row_lines(cells, top=0.4), footer], top=0.3)
    pred = normalize(page)
    assert pred.markdown.startswith("Table 1: Orders\n\n<table>")
    assert pred.markdown.endswith("</table>\n\nSource: ERP export\n")
    owner = region if kind == "TABLE" else table
    assert [(i.id, i.cls) for i in pred.layout] == [
        (title_note, ElementClass.CAPTION),
        (owner, ElementClass.TABLE),
        (footer_note, ElementClass.FOOTNOTE),
    ]
    assert [(r.kind, r.source, r.target) for r in pred.relations] == [
        (RelationKind.CAPTION_OF, title_note, owner),
        (RelationKind.FOOTNOTE_OF, footer_note, owner),
    ]


def test_a_title_in_the_grid_is_not_also_a_note_of_the_region_lines() -> None:
    page = Page()
    table, cells = page.table(
        [["Revenue", ""], ["Region", "EUR"], ["North", "42"]],
        ["TABLE_TITLE", "COLUMN_HEADER", None],
    )
    page.merge(table, [cells[1, 1], cells[1, 2]], row=1, column=1, rows=1, columns=2)
    page.note(table, "TABLE_TITLE", _children(page, cells[1, 1]), top=0.35)
    page.layout_of("TABLE", page.row_lines(cells))
    pred = normalize(page)
    assert pred.markdown.count("Revenue") == 1
    assert [i.cls for i in pred.layout] == [ElementClass.CAPTION, ElementClass.TABLE]


def test_notes_and_region_lines_keep_their_order_on_the_page() -> None:
    page = Page()
    title_words = page.words("Table 1: Orders")
    title = page._add(
        "LINE",
        title_words + page.words("(continued)"),
        Text="Table 1: Orders (continued)",
        Geometry=_geometry(top=0.3),
    )
    table, cells = page.table(QTY, top=0.4)
    between = page.line("Note A", top=0.55)
    footer = page.line("Source: ERP", top=0.6)
    page.note(table, "TABLE_TITLE", title_words, top=0.3)
    page.note(table, "TABLE_FOOTER", _children(page, footer), top=0.6)
    page.layout_of("TABLE", [title, *page.row_lines(cells, top=0.4), between, footer], top=0.3)
    pred = normalize(page)
    order = [
        pred.markdown.index(t)
        for t in ("Table 1: Orders", "(continued)", "<table>", "Note A", "Source: ERP")
    ]
    assert order == sorted(order)
    assert pred.markdown.count("Table 1: Orders") == 1


def test_a_title_printed_below_its_table_follows_it() -> None:
    page = Page()
    table, cells = page.table(QTY, top=0.3)
    title = page.line("Table 4: Orders", top=0.5)
    page.note(table, "TABLE_TITLE", _children(page, title), top=0.5)
    page.layout_of("TABLE", [*page.row_lines(cells, top=0.3), title], top=0.3)
    pred = normalize(page)
    assert pred.markdown.endswith("</table>\n\nTable 4: Orders\n")
    assert [i.cls for i in pred.layout] == [ElementClass.TABLE, ElementClass.CAPTION]


def test_a_line_without_words_keeps_its_item() -> None:
    page = Page()
    page.layout_of("TEXT", [page._add("LINE", Text="Typed note")])
    pred = normalize(page)
    assert pred.markdown == "Typed note\n"
    assert [(i.cls, i.text) for i in pred.layout] == [(ElementClass.PARAGRAPH, "Typed note")]


def test_a_footer_row_a_merged_cell_reaches_stays_in_the_grid() -> None:
    page = Page()
    table, cells = page.table(
        [["Region", "Sales"], ["North", "42"], ["Source: survey", "est."]],
        ["COLUMN_HEADER", None, "TABLE_FOOTER"],
    )
    page.merge(table, [cells[2, 2], cells[3, 2]], row=2, column=2, rows=2, columns=1)
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    read = parse_markdown_tables(pred.markdown)[0]
    assert read.body == (("North", "42 est."), ("Source: survey", "42 est."))
    assert pred.markdown.count("est.") == 1 and pred.markdown.count("Source: survey") == 1
    assert [i.cls for i in pred.layout] == [ElementClass.TABLE]


def test_a_title_row_a_merged_cell_joins_to_the_grid_stays_in_it() -> None:
    page = Page()
    table, cells = page.table(
        [["Orders", "Q1"], ["Region", "Q2"], ["North", "42"]],
        ["TABLE_TITLE", "COLUMN_HEADER", None],
    )
    page.merge(table, [cells[1, 2], cells[2, 2]], row=1, column=2, rows=2, columns=1)
    page.layout_of("TABLE", [table])
    pred = normalize(page)
    assert len(parse_markdown_tables(pred.markdown)[0].rows) == 3
    assert pred.markdown.count("Orders") == 1
    assert [i.cls for i in pred.layout] == [ElementClass.TABLE]


def test_a_table_no_layout_block_holds_ends_the_page() -> None:
    page = Page()
    page.layout("TEXT", "Body.")
    _, cells = page.table(QTY)
    page.row_lines(cells)
    pred = normalize(page)
    assert pred.markdown.startswith("Body.\n\n<table>")
    assert pred.unscored_text == ["Qty Item Price", "2 Chair 120.00", "1 Table 340.00"]


def test_a_table_goes_with_body_text_rather_than_a_running_header() -> None:
    page = Page()
    table, cells = page.table(QTY)
    lines = page.row_lines(cells)
    page.layout_of("HEADER", lines[:1], top=0.02)
    page.layout("TEXT", "Intro.", top=0.2)
    page.layout_of("TEXT", lines[1:], top=0.4)
    pred = normalize(page)
    assert pred.markdown.startswith("Intro.\n\n<table>")


def test_a_title_of_a_table_outside_any_table_region_is_its_caption() -> None:
    page = Page()
    title = page.line("Table 2: Orders", top=0.3)
    table, cells = page.table(QTY)
    page.relate(table, "TABLE_TITLE", [page._add("TABLE_TITLE", _children(page, title))])
    caption = page.layout_of("TEXT", [title], top=0.3)
    page.layout_of("TEXT", page.row_lines(cells))
    pred = normalize(page)
    assert {i.id: i.cls for i in pred.layout}[caption] is ElementClass.CAPTION
    assert [(r.kind, r.source, r.target) for r in pred.relations] == [
        (RelationKind.CAPTION_OF, caption, table)
    ]


def test_a_layout_table_without_a_table_keeps_its_lines_as_text() -> None:
    page = Page()
    page.layout("TABLE", "Region Revenue", "North 42")
    pred = normalize(page)
    assert pred.markdown == "Region Revenue North 42\n"
    assert [(i.cls, i.cells) for i in pred.layout] == [(ElementClass.TABLE, None)]


def test_table_region_lines_outside_the_cells_are_kept_around_the_table() -> None:
    page = Page()
    table, cells = page.table([["A"], ["1"]])
    words = [w for c in cells.values() for w in _children(page, c)]
    table_line = page._add("LINE", words, Text="A 1", Geometry=_geometry(top=0.4))
    above, below = page.line("Units: EUR", top=0.2), page.line("Source: survey", top=0.9)
    page.layout_of("TABLE", [above, table_line, below])
    pred = normalize(page)
    assert pred.markdown.startswith("Units: EUR\n\n<table>")
    assert pred.markdown.endswith("</table>\n\nSource: survey\n")


def test_pages_without_markdown_start_where_the_next_page_starts() -> None:
    first, blank, last, trailing = Page(), Page(), Page(), Page()
    first.layout("TEXT", "One.")
    last.layout("TEXT", "Three.")
    pred = normalize(first, blank, last, trailing)
    assert pred.markdown == "One.\n\nThree.\n"
    assert pred.page_offsets == [0, 6, 6, len(pred.markdown)]


def test_an_unknown_layout_type_is_an_error() -> None:
    page = Page()
    page.layout("SIDEBAR", "Aside")
    with pytest.raises(ValueError, match="no class mapping for Textract 'LAYOUT_SIDEBAR'"):
        normalize(page)


def test_boxes_are_page_fractions_clipped_like_every_provider() -> None:
    page = Page()
    page._add("LAYOUT_TEXT", [page.line("Edge")], Geometry=_geometry(-0.005, 0.2, 0.3, 0.1))
    page._add("LAYOUT_TEXT", [page.line("Off")], Geometry=_geometry(-0.5, 0.2, 0.3, 0.1))
    boxes = [i.box for i in normalize(page).layout]
    assert boxes[0] == pytest.approx((0.0, 0.2, 0.295, 0.1))
    assert boxes[1] is None


def _children(page: Page, block_id: str) -> list[str]:
    block = page.block(block_id)
    return [i for r in block.get("Relationships", []) for i in r["Ids"]]


def test_usage_counts_the_pages_textract_analyzed() -> None:
    payload = {"pages": [Page().response(), Page().response()]}
    assert TextractProvider().usage(RawResult("textract", "d", payload)) == {"pages": 2}
