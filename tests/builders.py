from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from wholedoc_bench.corpus.checks import check_groundtruth
from wholedoc_bench.schema import (
    Box,
    Chart,
    ChartKind,
    ChartPoint,
    ChartSeries,
    DocumentInfo,
    Element,
    ElementClass,
    Groundtruth,
    LayoutItem,
    Page,
    Planted,
    Prediction,
    Relation,
    RelationKind,
)
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.blocks import MarkdownTable
from wholedoc_bench.scorer.utils.groundtruth import next_pieces, piece_chain, table_grid
from wholedoc_bench.scorer.utils.tracks.t4_tables.table_matching import output_links
from wholedoc_bench.scorer.utils.tracks.table_links import linked_tables

Row = Sequence[str]
Rows = Sequence[Row]

SeriesPoints = Sequence[tuple[str, float]]
ChartSeriesData = Mapping[str, SeriesPoints]
AxisRange = tuple[float, float]

PIPE_DELIMITER_CELL = "---"
CHART_BOX: Box = (0.125, 0.125, 0.5, 0.25)
LABEL_CORNER = "Label"


def table_element(
    element_id: str, page: int = 0, header: Row | None = None, rows: Rows | None = None
) -> Element:
    return Element(
        id=element_id,
        cls=ElementClass.TABLE,
        page=page,
        header=None if header is None else [list(header)] if header else [],
        rows=None if rows is None else [list(row) for row in rows],
    )


def _series_fields(series: ChartSeriesData) -> list[dict[str, object]]:
    return [
        {"name": name, "points": [{"label": label, "value": value} for label, value in points]}
        for name, points in series.items()
    ]


def chart_data(
    kind: ChartKind | None,
    series: ChartSeriesData | None = None,
    value_axis: AxisRange | None = None,
    *,
    title: str | None = None,
) -> Chart | None:
    if kind is None:
        return None
    fields: dict[str, object] = {"kind": kind, "title": title}
    if series:
        fields["series"] = _series_fields(series)
    if value_axis is not None:
        fields["value_axis"] = {"min": value_axis[0], "max": value_axis[1]}
    return Chart.model_validate(fields)


def chart_element(
    element_id: str,
    text: str = "",
    page: int = 0,
    *,
    kind: ChartKind | None = None,
    box: Box | None = None,
    series: ChartSeriesData | None = None,
    value_axis: AxisRange | None = None,
    title: str | None = None,
) -> Element:
    return Element(
        id=element_id,
        cls=ElementClass.CHART,
        page=page,
        box=box,
        text=text,
        chart=chart_data(kind, series, value_axis, title=title),
    )


def chart_item(
    item_id: str,
    kind: ChartKind | None,
    *,
    page: int = 0,
    box: Box | None = CHART_BOX,
    text: str = "",
    series: ChartSeriesData | None = None,
    md_start: int | None = None,
    md_end: int | None = None,
    title: str | None = None,
) -> LayoutItem:
    return LayoutItem(
        id=item_id,
        cls=ElementClass.CHART,
        page=page,
        box=box,
        text=text,
        md_start=md_start,
        md_end=md_end,
        chart=chart_data(kind, series, title=title),
    )


def chart_grid(
    series: ChartSeriesData, corner: str = LABEL_CORNER, *, transpose: bool = False
) -> tuple[list[str], list[list[str]]]:
    names = list(series)
    labels = [label for label, _ in next(iter(series.values()))]
    cells = {name: {label: format(value, "g") for label, value in series[name]} for name in names}
    if transpose:
        rows = [[name, *(cells[name][label] for label in labels)] for name in names]
        return [corner, *labels], rows
    rows = [[label, *(cells[name][label] for name in names)] for label in labels]
    return [corner, *names], rows


def continues_in(source: str, target: str) -> Relation:
    return Relation(kind=RelationKind.CONTINUES_IN, source=source, target=target)


def stays_apart(source: str, target: str) -> Relation:
    return Relation(kind=RelationKind.STAYS_APART, source=source, target=target)


def groundtruth(
    elements: Sequence[Element | dict[str, Any]],
    relations: Sequence[Relation] = (),
    planted: Sequence[Planted] = (),
    *,
    page_count: int | None = None,
) -> Groundtruth:
    elements = [e if isinstance(e, Element) else Element.model_validate(e) for e in elements]
    if page_count is None:
        page_count = max((e.page for e in elements), default=0) + 1
    pages = [Page(index=i, width=595.0, height=842.0) for i in range(page_count)]
    result = Groundtruth(
        schema_version="0.1",
        document=DocumentInfo(
            id="t",
            source="born-digital",
            groundtruth_source="human",
            language=["en"],
            pages=pages,
        ),
        elements=list(elements),
        relations=list(relations),
        planted=list(planted),
    )
    check_groundtruth(result)
    return result


def layout_item(
    item_id: str,
    cls: ElementClass,
    text: str = "",
    page: int = 0,
    md_start: int | None = None,
    md_end: int | None = None,
) -> LayoutItem:
    return LayoutItem(id=item_id, cls=cls, page=page, text=text, md_start=md_start, md_end=md_end)


def layout_over(
    item_id: str, cls: ElementClass, markdown: str, block: str, text: str = ""
) -> LayoutItem:
    start = markdown.index(block)
    return layout_item(item_id, cls, text=text, md_start=start, md_end=start + len(block))


def prediction(markdown: str = "", layout: Sequence[LayoutItem] = ()) -> Prediction:
    return Prediction(markdown=markdown, layout=list(layout))


def context(
    truth: Groundtruth,
    markdown: str,
    layout: Sequence[LayoutItem] = (),
    relations: Sequence[Relation] = (),
) -> DocContext:
    return DocContext(
        truth, Prediction(markdown=markdown, layout=list(layout), relations=list(relations))
    )


def grid(prefix: str, body_rows: int, columns: int) -> tuple[list[str], list[list[str]]]:
    header = [f"{prefix}h{c}" for c in range(columns)]
    rows = [[f"{prefix}r{r}c{c}" for c in range(columns)] for r in range(body_rows)]
    return header, rows


def cells_text(header: Row | None, rows: Rows) -> str:
    lines = ([header] if header is not None else []) + list(rows)
    return " ".join(cell for row in lines for cell in row)


def _pipe_line(cells: Row) -> str:
    return "| " + " | ".join(cells) + " |"


def pipe_table(header: Row | None, rows: Rows) -> str:
    columns = len(header) if header is not None else max(len(row) for row in rows)
    header_cells = header if header is not None else [""] * columns
    lines = [_pipe_line(header_cells), _pipe_line([PIPE_DELIMITER_CELL] * columns)]
    lines += [_pipe_line(row) for row in rows]
    return "\n".join(lines)


def html_row(cells: Row, tag: str) -> str:
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def html_table(header: Row | None, rows: Rows, thead: bool = True) -> str:
    body = "".join(html_row(row, "td") for row in rows)
    if header is None:
        return f"<table>{body}</table>"
    head = html_row(header, "th")
    if thead:
        return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"
    return f"<table>{head}{body}</table>"


def markdown_of(*blocks: str) -> str:
    return "\n\n".join(blocks) + "\n"


def logical_tables(truth: Groundtruth) -> list[tuple[list[str] | None, list[list[str]]]]:
    tables_by_id = {e.id: e for e in truth.elements if e.cls is ElementClass.TABLE}
    next_of = next_pieces(truth, tables_by_id)
    continuations = set(next_of.values())
    return [
        (
            _header_row(head),
            [row for i in piece_chain(head.id, next_of) for row in table_grid(tables_by_id[i])[1]],
        )
        for head in tables_by_id.values()
        if head.id not in continuations
    ]


def _header_row(table: Element) -> list[str] | None:
    header = table_grid(table)[0]
    if header is None:
        return None
    [row] = header
    return row


def perfect_pipe_markdown(truth: Groundtruth) -> str:
    return markdown_of(*(pipe_table(header, rows) for header, rows in logical_tables(truth)))


def perfect_html_markdown(truth: Groundtruth) -> str:
    return markdown_of(*(html_table(header, rows) for header, rows in logical_tables(truth)))


HEADER = ["Metric", "Value"]
ROWS = [["Revenue", "10"], ["Cost", "7"]]


def one_table(header: Row | None = HEADER, rows: Rows = ROWS) -> Groundtruth:
    return groundtruth([table_element("t1", 0, header, rows)])


CHART_HEADER = ["Quarter", "Revenue"]
CHART_ROWS = [["Q1", "10"], ["Q2", "20"]]
CHART_TEXT = cells_text(CHART_HEADER, CHART_ROWS)


def table_and_chart(chart_first: bool = False) -> Groundtruth:
    table = table_element("t1", 0, CHART_HEADER, CHART_ROWS)
    chart = chart_element("c1", CHART_TEXT)
    return groundtruth([chart, table] if chart_first else [table, chart])


def chart_pipe() -> str:
    return pipe_table(CHART_HEADER, CHART_ROWS)


def as_tuples(rows) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(row) for row in rows)


def split_truth() -> Groundtruth:
    return groundtruth(
        [
            table_element("a", 0, ["Name", "Qty"], [["apple", "1"], ["pear", "2"]]),
            table_element("b", 1, None, [["plum", "3"], ["fig", "4"]]),
        ],
        [continues_in("a", "b")],
    )


def two_different_tables(relation_kind=stays_apart) -> Groundtruth:
    return groundtruth(
        [
            table_element("a", 0, ["Name", "Qty"], [["apple", "1"], ["pear", "2"]]),
            table_element("b", 1, ["City", "Zip"], [["Oslo", "0150"], ["Bern", "3000"]]),
        ],
        [relation_kind("a", "b")],
    )


def only_table(ctx: DocContext) -> MarkdownTable:
    tables = markdown_tables(ctx)
    assert len(tables) == 1
    return tables[0]


def markdown_tables(ctx: DocContext) -> tuple[MarkdownTable, ...]:
    return tuple(link.table for link in linked_tables(ctx))


def chart_tables(ctx: DocContext) -> tuple[MarkdownTable, ...]:
    return tuple(link.table for link in linked_tables(ctx) if link.kind is ElementClass.CHART)


def chart_table_anchors(ctx: DocContext) -> tuple[str | None, ...]:
    return tuple(link.anchor_id for link in linked_tables(ctx) if link.kind is ElementClass.CHART)


def output_table_anchors(ctx: DocContext) -> tuple[str | None, ...]:
    return tuple(link.anchor_id for link in output_links(linked_tables(ctx)))


def indexes(tables: Sequence[MarkdownTable]) -> list[int]:
    return [t.index for t in tables]


def quoted(block: str, marker: str = "> ") -> str:
    return "\n".join(marker + line for line in block.splitlines())


def as_prediction(truth: Groundtruth) -> Prediction:
    layout = [
        LayoutItem(
            id=e.id,
            cls=e.cls,
            page=e.page,
            box=e.box,
            text=e.text,
            channel=e.channel,
            chart=e.chart,
        )
        for e in truth.elements
    ]
    return prediction("\n\n".join(e.text for e in truth.elements if e.text), layout)


TOP_LEFT: Box = (0.0, 0.0, 0.25, 0.25)
BOTTOM_RIGHT: Box = (0.5, 0.5, 0.25, 0.25)
REVENUE: ChartSeriesData = {"Revenue": [("Q1", 10.0), ("Q2", 20.0), ("Q3", 18.0), ("Q4", 25.0)]}
REGIONS: ChartSeriesData = {
    "North": [("Q1", 10.0), ("Q2", 20.0)],
    "South": [("Q1", 30.0), ("Q2", 40.0)],
}
PROFIT: ChartSeriesData = {"Profit": [("Q1", 3.0), ("Q2", 4.0)]}
TITLE = "Revenue by quarter"


def one_chart(
    series: ChartSeriesData = REVENUE,
    *,
    text: str = "",
    value_axis: AxisRange | None = None,
) -> Groundtruth:
    return groundtruth(
        [
            chart_element(
                "c1",
                text,
                kind=ChartKind.BAR,
                box=CHART_BOX,
                series=series,
                value_axis=value_axis,
            )
        ]
    )


def two_charts(first: ChartSeriesData, second: ChartSeriesData) -> Groundtruth:
    return groundtruth(
        [
            chart_element("c1", kind=ChartKind.BAR, box=TOP_LEFT, series=first),
            chart_element("c2", kind=ChartKind.BAR, box=BOTTOM_RIGHT, series=second),
        ]
    )


def markdown_only(*blocks: str) -> Prediction:
    return prediction(markdown_of(*blocks))


def json_only(series: ChartSeriesData, box: Box = CHART_BOX) -> Prediction:
    return prediction(layout=[chart_item("p1", ChartKind.BAR, box=box, series=series)])


def series_pipe(series: ChartSeriesData = REVENUE, *, transpose: bool = False) -> str:
    return pipe_table(*chart_grid(series, transpose=transpose))


def chart_text(series: ChartSeriesData = REVENUE) -> str:
    return cells_text(*chart_grid(series))


def json_item_with_linked_table(
    json_series: ChartSeriesData, table_series: ChartSeriesData, *, copies: int = 1
) -> Prediction:
    layout = [chart_item("p1", ChartKind.BAR, text=chart_text(table_series), series=json_series)]
    return prediction(markdown_of(*[series_pipe(table_series)] * copies), layout)


def named_series(name: str, points: SeriesPoints) -> ChartSeries:
    return ChartSeries(
        name=name, points=[ChartPoint(label=label, value=value) for label, value in points]
    )


def truth_with_series(*series: ChartSeries) -> Groundtruth:
    chart = Chart(kind=ChartKind.BAR, series=list(series))
    element = Element(id="c1", cls=ElementClass.CHART, page=0, box=CHART_BOX, chart=chart)
    return groundtruth([element])


def item_with_series(*series: ChartSeries) -> LayoutItem:
    chart = Chart(kind=ChartKind.BAR, series=list(series))
    return LayoutItem(id="p1", cls=ElementClass.CHART, page=0, box=CHART_BOX, chart=chart)


def truth(
    *elements: dict[str, Any],
    relations: list[dict[str, Any]] | None = None,
    planted: list[dict[str, Any]] | None = None,
) -> Groundtruth:
    return groundtruth(
        elements,
        [Relation.model_validate(r) for r in relations or []],
        [Planted.model_validate(p) for p in planted or []],
    )


def para(id: str, text: str, page: int = 0, **extra: Any) -> dict[str, Any]:
    return {"id": id, "class": "paragraph", "page": page, "text": text, **extra}


def boxed(
    id_: str, cls: str, box: Box, text: str = "", page: int = 0, **extra: Any
) -> dict[str, Any]:
    return {"id": id_, "class": cls, "page": page, "box": box, "text": text, **extra}


def boxed_item(
    id_: str, cls: str, box: Box | None, text: str = "", page: int = 0, **extra: Any
) -> LayoutItem:
    return LayoutItem.model_validate(
        {"id": id_, "class": cls, "page": page, "box": box, "text": text, **extra}
    )


def element_as_text(element: Element) -> str:
    grid = [*(element.header or []), *(element.rows or [])]
    cells = [cell for row in grid for cell in row]
    return " ".join(part for part in (element.text, *cells) if part)


def element_as_item(element: Element) -> LayoutItem:
    return LayoutItem(
        id=element.id,
        cls=element.cls,
        page=element.page,
        text=element_as_text(element),
        box=element.box,
        channel=element.channel,
        cells=element.cells,
    )


def cell(text: str, box: Box | None) -> dict[str, Any]:
    return {"text": text, "box": box}


GRID_LEFT = 0.1
GRID_TOP = 0.3
CELL_WIDTH = 0.2
CELL_HEIGHT = 0.1


def grid_box(row: int, column: int) -> Box:
    return (GRID_LEFT + CELL_WIDTH * column, GRID_TOP + CELL_HEIGHT * row, CELL_WIDTH, CELL_HEIGHT)


def grid_area(texts: list[list[str]]) -> Box:
    return (GRID_LEFT, GRID_TOP, CELL_WIDTH * len(texts[0]), CELL_HEIGHT * len(texts))


def grid_cells(texts: list[list[str]]) -> list[dict[str, Any]]:
    return [cell(text, grid_box(r, c)) for r, row in enumerate(texts) for c, text in enumerate(row)]


def grid_table(id_: str, texts: list[list[str]], page: int = 0, **extra: Any) -> dict[str, Any]:
    return boxed(
        id_,
        "table",
        grid_area(texts),
        page=page,
        header=texts[0],
        rows=texts[1:],
        cells=grid_cells(texts),
        **extra,
    )


def table_item(
    cells: list[dict[str, Any]], box: Box | None, page: int = 0, id_: str = "x"
) -> LayoutItem:
    text = " ".join(c["text"] for c in cells if c["text"])
    return boxed_item(id_, "table", box, text, page, cells=cells)


def truth_as_items(truth: Groundtruth) -> list[LayoutItem]:
    return [element_as_item(e) for e in truth.elements]


HEADING: Box = (0.1, 0.05, 0.5, 0.05)
PARAGRAPH: Box = (0.1, 0.15, 0.6, 0.1)
TABLE: Box = (0.1, 0.3, 0.8, 0.3)
FIGURE: Box = (0.1, 0.65, 0.8, 0.25)
FOOTER: Box = (0.45, 0.95, 0.1, 0.03)
EMPTY_AREA: Box = (0.7, 0.05, 0.2, 0.05)

HEADING_TEXT = "Quarterly report"
PARAGRAPH_TEXT = "Revenue grew in the third quarter of the year."
OTHER_TEXT = "Costs fell sharply during a cold winter."

FULL_DOCUMENT = truth(
    boxed("h", "heading", HEADING, HEADING_TEXT, level=1),
    boxed("p", "paragraph", PARAGRAPH, PARAGRAPH_TEXT),
    boxed("t", "table", TABLE, header=["Name", "Value"], rows=[["alpha", "1"], ["beta", "2"]]),
    boxed("g", "figure", FIGURE),
    boxed("f", "footer", FOOTER, "1", channel="furniture"),
)

THREE_ELEMENTS = truth(
    boxed("h", "heading", HEADING, HEADING_TEXT, level=1),
    boxed("p", "paragraph", PARAGRAPH, PARAGRAPH_TEXT),
    boxed("f", "footer", FOOTER, "1", channel="furniture"),
)


def item_series(*series: tuple[str, Sequence[tuple[str, float]]]) -> Prediction:
    return prediction(layout=[item_with_series(*(named_series(n, p) for n, p in series))])


def two_named(first: str, second: str) -> Groundtruth:
    return truth_with_series(
        named_series(first, [("Q1", 1.0)]), named_series(second, [("Q1", 1.0)])
    )


def two_labels(first: str, second: str) -> Groundtruth:
    return truth_with_series(named_series("Revenue", [(first, 1.0), (second, 2.0)]))


GRID = [["Name", "Value", "Unit"], ["alpha", "1", "kg"], ["beta", "2", "m"]]


GRID_TABLE = grid_table("t", GRID)


beside_grid = partial(truth, GRID_TABLE)


URL = "https://example.com/inv/4417"


def code(id_: str, payload: str | None = URL, **extra: Any) -> dict[str, Any]:
    element = {
        "id": id_,
        "class": "machine_readable",
        "page": 0,
        "box": (0.1, 0.1, 0.2, 0.2),
        **extra,
    }
    return element if payload is None else {**element, "payload": payload}


ICAO_TD3 = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"
)
LONG_NUMBER_TD1 = "\n".join(
    [
        "I<UTOD23145890<120<<<<<<<<<<<<",
        "7408122F1204159UTO<<<<<<<<<<<2",
        "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
    ]
)
MRZ_BOX: Box = (0.05, 0.8, 0.9, 0.1)


def mrz(id_: str, payload: str = ICAO_TD3, **extra: Any) -> dict[str, Any]:
    return {
        "id": id_,
        "class": "machine_readable",
        "page": 0,
        "box": MRZ_BOX,
        "format": "MrzTd3",
        "payload": payload,
        **extra,
    }


def mrz_item(
    item_id: str = "z",
    text: str = ICAO_TD3,
    *,
    valid: bool | None = True,
    box: Box | None = MRZ_BOX,
    page: int = 0,
    format_: str | None = "MrzTd3",
) -> LayoutItem:
    return LayoutItem(
        id=item_id,
        cls=ElementClass.MACHINE_READABLE,
        page=page,
        box=box,
        text=text,
        format=format_,
        check_digits_valid=valid,
    )


def invoice(*codes: dict[str, Any]) -> Groundtruth:
    return truth(para("p", "Scan to pay."), *(codes or [code("q")]))


def typo(printed: str) -> dict[str, str]:
    return {"kind": "exact", "printed": printed}


def heading(id_: str, text: str, level: int | None, **extra: Any) -> dict[str, Any]:
    return {"id": id_, "class": "heading", "page": 0, "text": text, "level": level, **extra}


FINDINGS = "The committee reviewed the full report and approved the final findings."


MINUTES = "Every member signed the minutes before the meeting closed that evening."


NO_HEADINGS = truth(para("a", FINDINGS), para("b", MINUTES))


A = (
    "The committee met on Monday to review the budget for the coming year and agreed to "
    "delay the final vote until every department had submitted its revised figures."
)


B = (
    "Several members asked for a clearer breakdown of travel costs, which had grown faster "
    "than any other line in the previous two reports."
)


C = (
    "The chair proposed a short working group to prepare that breakdown and to report back "
    "before the end of the month."
)


D = (
    "No objections were raised, and the meeting closed shortly after four o'clock with a "
    "reminder about the date of the next session."
)


E = (
    "Copies of the revised budget will be sent to all members at least one week before "
    "that session begins."
)


def without_marks(text: str) -> str:
    return " ".join(text.replace("*", " ").replace("_", " ").split())


def chain(pieces, edges):
    return truth(
        *[
            para(str(i), text, page=i, styles=[{"style": "bold", "start": 0, "end": len(text)}])
            for i, text in enumerate(pieces)
        ],
        relations=[
            {"kind": "continues_in", "from": str(i), "to": str(i + 1), "hyphenated": edge}
            for i, edge in enumerate(edges)
        ],
    )


def header_rows_table(header_rows: Rows, rows: Rows) -> str:
    head = "".join(html_row(row, "th") for row in header_rows)
    body = "".join(html_row(row, "td") for row in rows)
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


ACCOUNT = {"kind": "hidden", "printed": "acct 4417"}


NAME = {"kind": "redacted", "printed": "Jane Roe"}


VISIBLE = "Pay the invoice by Friday."


def memo(*planted: dict[str, Any]) -> Groundtruth:
    hidden = {**para("h", "acct 4417"), "channel": "hidden"}
    redacted = {**para("r", "Jane Roe"), "channel": "hidden"}
    return truth(para("p", VISIBLE), hidden, redacted, planted=list(planted))


def secret(printed: str, *visible: dict[str, Any]) -> Groundtruth:
    held = {**para("s", printed), "channel": "hidden"}
    trap = {"kind": "redacted", "printed": printed}
    return truth(*(visible or [para("p", VISIBLE)]), held, planted=[trap])


def chart(**fields: Any) -> dict[str, Any]:
    series = [{"name": "Revenue", "points": [{"label": "Q1", "value": 3}]}]
    return {
        "id": "c",
        "class": "chart",
        "page": 0,
        "box": CHART_BOX,
        "chart": {"kind": "bar", "series": series, **fields},
    }


def element(**extra: Any) -> Element:
    return Element.model_validate(
        {"id": "p", "class": "paragraph", "page": 0, "text": "Annual Budget", **extra}
    )


def span(style: str = "bold", start: Any = 7, end: Any = 13) -> dict[str, Any]:
    return {"style": style, "start": start, "end": end}


def incomplete_truth() -> Groundtruth:
    truth = groundtruth([element(styles=[])])
    return truth.model_copy(update={"elements": [element(styles=[]), element(id="unknown")]})
