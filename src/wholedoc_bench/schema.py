from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from wholedoc_bench.scorer.utils.mrz import is_mrz, shape_error

SCHEMA_VERSION = "0.1"

Box = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

_BOX_EPSILON = 1e-6


def _check_box(box: Box | None, where: str) -> None:
    if box is None:
        return
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise ValueError(f"{where}: box width and height must be positive, got {box}")
    if w * h == 0:
        raise ValueError(f"{where}: box area rounds to 0, got {box}")
    if x < -_BOX_EPSILON or y < -_BOX_EPSILON:
        raise ValueError(f"{where}: box starts before the page edge, got {box}")
    if x + w > 1 + _BOX_EPSILON or y + h > 1 + _BOX_EPSILON:
        raise ValueError(f"{where}: box ends past the page edge, got {box}")


class Channel(StrEnum):
    BODY = "body"
    FURNITURE = "furniture"
    ANNOTATION = "annotation"
    METADATA = "metadata"
    HIDDEN = "hidden"


class ElementClass(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    TABLE = "table"
    CHART = "chart"
    FIGURE = "figure"
    FORMULA = "formula"
    CODE = "code"
    HEADER = "header"
    FOOTER = "footer"
    MACHINE_READABLE = "machine_readable"


class RelationKind(StrEnum):
    CAPTION_OF = "caption_of"
    FOOTNOTE_OF = "footnote_of"
    CONTINUES_IN = "continues_in"
    STAYS_APART = "stays_apart"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StyleName(StrEnum):
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKETHROUGH = "strikethrough"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"


class StyleSpan(_Model):
    style: StyleName
    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.start >= self.end:
            raise ValueError("style span needs start < end")
        return self


class FontClass(StrEnum):
    SERIF = "serif"
    SANS = "sans"
    MONO = "mono"
    SCRIPT = "script"
    DECORATIVE = "decorative"


class Font(_Model):
    family: str | None = None
    font_class: FontClass | None = Field(
        default=None,
        validation_alias=AliasChoices("class", "font_class"),
        serialization_alias="class",
    )
    size: FiniteFloat | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_fields(self) -> Self:
        if self.family is not None and not self.family.strip():
            raise ValueError("font family must not be blank")
        if self.family is None and self.font_class is None and self.size is None:
            raise ValueError("font needs a family, a class or a size")
        return self


class ChartKind(StrEnum):
    BAR = "bar"
    LINE = "line"
    AREA = "area"
    PIE = "pie"
    SCATTER = "scatter"
    BUBBLE = "bubble"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    FUNNEL = "funnel"
    RADAR = "radar"
    CANDLESTICK = "candlestick"
    STEP = "step"
    STEM = "stem"
    COMBO = "combo"
    DENSITY = "density"
    ERRORBAR = "errorbar"
    BULLET = "bullet"
    MARIMEKKO = "marimekko"
    PYRAMID = "pyramid"
    SPAN = "span"
    RADIAL_BAR = "radial_bar"
    PARALLEL = "parallel"
    STREAM = "stream"
    DONUT = "donut"
    ROSE = "rose"
    SUNBURST = "sunburst"
    CIRCLE_PACKING = "circle_packing"
    PROPORTIONAL_AREA = "proportional_area"
    DOT_MATRIX = "dot_matrix"
    GANTT = "gantt"
    WATERFALL = "waterfall"


class ChartPoint(_Model):
    label: str
    value: FiniteFloat


class ChartSeries(_Model):
    name: str = ""
    points: list[ChartPoint]


class ValueAxis(_Model):
    min: FiniteFloat
    max: FiniteFloat

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.max <= self.min:
            raise ValueError(f"value axis max {self.max} must be above min {self.min}")
        return self


class Chart(_Model):
    kind: ChartKind | None = None
    series: list[ChartSeries] = []
    value_axis: ValueAxis | None = None
    title: str | None = None

    @model_validator(mode="after")
    def _check_title(self) -> Self:
        if self.title is not None and not self.title.strip():
            raise ValueError("chart title must not be blank; use None for no title")
        return self


def _repeated_texts(texts: Iterable[str]) -> list[str]:
    counts = Counter(text.strip() for text in texts)
    return [text for text, count in counts.items() if count > 1]


def _check_truth_chart_data(chart: Chart, where: str) -> None:
    if chart.kind is None:
        raise ValueError(f"{where}: chart data needs a kind")
    if repeated := _repeated_texts(s.name for s in chart.series):
        raise ValueError(f"{where}: chart series names repeat {repeated}")
    for series in chart.series:
        if not series.points:
            raise ValueError(f"{where}: series {series.name!r} has no points")
        if repeated := _repeated_texts(p.label for p in series.points):
            raise ValueError(f"{where}: series {series.name!r}: labels repeat {repeated}")


def _check_chart_class(cls: ElementClass, chart: Chart | None, where: str) -> None:
    if chart is not None and cls is not ElementClass.CHART:
        raise ValueError(f"{where}: chart data needs class chart, got {cls}")


def _json_name(json_key: str, field_name: str) -> Any:
    return Field(validation_alias=AliasChoices(json_key, field_name), serialization_alias=json_key)


class Relation(_Model):
    kind: RelationKind
    source: str = _json_name("from", "source")
    target: str = _json_name("to", "target")
    hyphenated: bool = False


class Page(_Model):
    index: int = Field(ge=0)
    label: str | None = None
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class TableCell(_Model):
    text: str
    box: Box | None = None
    row: int | None = Field(default=None, ge=0)
    column: int | None = Field(default=None, ge=0)
    rows: int | None = Field(default=None, ge=1)
    columns: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_box(self) -> Self:
        _check_box(self.box, "table cell")
        return self

    @model_validator(mode="after")
    def _check_place(self) -> Self:
        if len({value is None for value in (self.row, self.column, self.rows, self.columns)}) > 1:
            raise ValueError("table cell: row, column, rows and columns go together")
        return self

    @model_serializer(mode="wrap")
    def _omit_no_place(self, handler: SerializerFunctionWrapHandler) -> Any:
        data = handler(self)
        if self.row is None:
            for key in ("row", "column", "rows", "columns"):
                data.pop(key, None)
        return data


class CellSpan(_Model):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    rows: int = Field(default=1, ge=1)
    columns: int = Field(default=1, ge=1)

    def positions(self) -> list[tuple[int, int]]:
        return [
            (self.row + r, self.column + c) for r in range(self.rows) for c in range(self.columns)
        ]


def _check_cells_on_table(cls: ElementClass, cells: list[TableCell] | None, where: str) -> None:
    if cells is not None and cls is not ElementClass.TABLE:
        raise ValueError(f"{where}: only a table can have cells, got class {cls.value}")


class Element(_Model):
    id: str
    cls: ElementClass = _json_name("class", "cls")
    page: int = Field(ge=0)
    box: Box | None = None
    text: str = ""
    channel: Channel = Channel.BODY
    level: int | None = Field(default=None, ge=1)
    header: list[list[str | None]] | None = None
    rows: list[list[str | None]] | None = None
    spans: list[CellSpan] | None = None
    chart: Chart | None = None
    cells: list[TableCell] | None = None
    styles: list[StyleSpan] | None = None
    font: Font | None = None
    payload: str | None = None
    format: str | None = None

    @field_validator("header", mode="before")
    @classmethod
    def _header_rows(cls, value: Any) -> Any:
        if isinstance(value, list) and value and not isinstance(value[0], list):
            return [value]
        return value

    @model_validator(mode="after")
    def _check_styles(self) -> Self:
        if self.styles is None:
            return self
        spans = sorted(self.styles, key=lambda span: (span.start, span.end, span.style))
        previous_end: dict[StyleName, int] = {}
        for span in spans:
            where = f"element {self.id}: {span.style} span [{span.start}, {span.end})"
            if span.end > len(self.text):
                raise ValueError(f"{where}: ends past printed text length {len(self.text)}")
            if not self.text[span.start : span.end].strip():
                raise ValueError(f"{where}: contains only whitespace")
            if span.start < previous_end.get(span.style, 0):
                raise ValueError(f"{where}: same-style spans overlap or duplicate")
            previous_end[span.style] = span.end
        object.__setattr__(self, "styles", spans)
        return self

    @model_validator(mode="after")
    def _check_box(self) -> Self:
        _check_box(self.box, f"element {self.id}")
        return self

    @model_validator(mode="after")
    def _check_chart(self) -> Self:
        where = f"element {self.id}"
        _check_chart_class(self.cls, self.chart, where)
        if self.chart is None:
            return self
        if self.box is None:
            raise ValueError(f"{where}: chart data needs a box")
        _check_truth_chart_data(self.chart, where)
        return self

    @model_validator(mode="after")
    def _check_payload(self) -> Self:
        where = f"element {self.id}"
        if self.cls is ElementClass.MACHINE_READABLE and self.text:
            raise ValueError(
                f"{where}: a machine_readable element has no text; its content goes in payload"
            )
        if self.format is not None and self.cls is not ElementClass.MACHINE_READABLE:
            raise ValueError(f"{where}: only a machine_readable element has a format")
        if is_mrz(self.format) and (self.payload is None or self.box is None):
            raise ValueError(f"{where}: a machine readable zone needs a box and its lines")
        if self.payload is None:
            return self
        if self.cls is not ElementClass.MACHINE_READABLE:
            raise ValueError(f"{where}: only a machine_readable element has a payload")
        if is_mrz(self.format) and (error := shape_error(self.format, self.payload)):
            raise ValueError(f"{where}: {error}")
        if not self.payload.strip():
            raise ValueError(
                f"{where}: an empty payload; leave it out for a code that cannot be decoded"
            )
        return self

    @model_validator(mode="after")
    def _check_cells(self) -> Self:
        _check_cells_on_table(self.cls, self.cells, f"element {self.id}")
        if self.cells == []:
            raise ValueError(f"element {self.id}: a cells list needs at least one cell")
        if any(cell.box is None for cell in self.cells or []):
            raise ValueError(f"element {self.id}: every table cell needs a box")
        if any(cell.row is not None for cell in self.cells or []):
            raise ValueError(f"element {self.id}: header and rows hold the grid, not the cells")
        return self

    @model_validator(mode="after")
    def _check_spans(self) -> Self:
        where = f"element {self.id}"
        header_rows = len(self.header or [])
        grid = (self.header or []) + (self.rows or [])
        if self.spans is not None and self.cls is not ElementClass.TABLE:
            raise ValueError(f"{where}: only a table can have spans, got class {self.cls.value}")
        if self.spans == []:
            raise ValueError(f"{where}: a spans list needs at least one span; leave it out instead")
        owner: dict[tuple[int, int], CellSpan] = {}
        for span in self.spans or []:
            name = f"{where}: span at row {span.row}, column {span.column}"
            if span.rows * span.columns == 1:
                raise ValueError(f"{name} covers one cell; a span merges at least two")
            if span.row < header_rows < span.row + span.rows:
                raise ValueError(f"{name} crosses from the header into the rows")
            for r, c in span.positions():
                if r >= len(grid) or c >= len(grid[r]):
                    raise ValueError(f"{name} reaches row {r}, column {c}, outside the grid")
                if (r, c) in owner:
                    raise ValueError(f"{name} overlaps another span at row {r}, column {c}")
                owner[(r, c)] = span
            if grid[span.row][span.column] is None:
                raise ValueError(f"{name}: its top-left cell holds the text and cannot be null")
        for r, row in enumerate(grid):
            for c, text in enumerate(row):
                span = owner.get((r, c))
                top_left = span is not None and (span.row, span.column) == (r, c)
                if text is None and span is None:
                    raise ValueError(f"{where}: row {r}, column {c} is null but no span covers it")
                if text is not None and span is not None and not top_left:
                    raise ValueError(
                        f"{where}: row {r}, column {c} is covered by the span at row "
                        f"{span.row}, column {span.column}, so it must be null"
                    )
        return self


def is_style_element(element: Element) -> bool:
    return (
        element.channel is Channel.BODY
        and bool(element.text.strip())
        and element.cls not in (ElementClass.TABLE, ElementClass.CHART)
    )


def is_typography_element(element: Element) -> bool:
    return is_style_element(element) and element.cls not in (
        ElementClass.FIGURE,
        ElementClass.FORMULA,
    )


class Planted(_Model):
    kind: Literal["exact", "hidden", "redacted"]
    printed: str


class DocumentInfo(_Model):
    id: str
    sha256: str | None = None
    source: Literal["born-digital", "scan", "photo"]
    groundtruth_source: Literal[
        "generator", "human", "second_system_reviewed", "nutrient_seeded_reviewed"
    ]
    language: list[str]
    pages: list[Page]


class Groundtruth(_Model):
    schema_version: Literal["0.1"]
    document: DocumentInfo
    elements: list[Element]
    relations: list[Relation] = []
    reading_order: list[str] = []
    planted: list[Planted] = []

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        ids = [e.id for e in self.elements]
        known = set(ids)
        if len(known) != len(ids):
            raise ValueError("element ids must be unique")
        page_count = len(self.document.pages)
        for e in self.elements:
            if e.page >= page_count:
                raise ValueError(f"element {e.id}: page {e.page} does not exist")
        for r in self.relations:
            for ref in (r.source, r.target):
                if ref not in known:
                    raise ValueError(f"relation {r.kind}: unknown element id {ref!r}")
        listed: set[str] = set()
        for ref in self.reading_order:
            if ref not in known:
                raise ValueError(f"reading_order: unknown element id {ref!r}")
            if ref in listed:
                raise ValueError(f"reading_order: {ref} appears more than once")
            listed.add(ref)
        by_id = {e.id: e for e in self.elements}
        position = {ref: i for i, ref in enumerate(self.reading_order)}
        pairs: dict[tuple[str, str], RelationKind] = {}
        for r in self.relations:
            if r.kind not in (RelationKind.CONTINUES_IN, RelationKind.STAYS_APART):
                continue
            _check_page_step(r, by_id[r.source], by_id[r.target], position)
            other = pairs.setdefault((r.source, r.target), r.kind)
            if other is not r.kind:
                raise ValueError(f"{r.source} -> {r.target}: both continues_in and stays_apart")
        _check_chains(self.relations)
        return self


_SAME_PAGE_CLASSES = frozenset({ElementClass.PARAGRAPH, ElementClass.LIST_ITEM})


def _check_page_step(
    r: Relation, source: Element, target: Element, position: dict[str, int]
) -> None:
    where = f"relation {r.kind} {r.source} -> {r.target}"
    step = target.page - source.page
    if r.kind is RelationKind.STAYS_APART:
        if step != 1:
            raise ValueError(f"{where}: must cross one page break")
        return
    if step not in (0, 1):
        raise ValueError(f"{where}: must stay on its page or cross one page break")
    if step == 1:
        return
    if source.cls is not target.cls or source.cls not in _SAME_PAGE_CLASSES:
        raise ValueError(f"{where}: on one page, only a paragraph or a list item can continue")
    if r.source in position and r.target in position and position[r.target] < position[r.source]:
        raise ValueError(f"{where}: on one page, it must go forward in reading_order")


def _check_chains(relations: list[Relation]) -> None:
    next_of: dict[str, str] = {}
    previous_of: dict[str, str] = {}
    for r in relations:
        if r.kind is not RelationKind.CONTINUES_IN:
            continue
        if r.source in next_of:
            raise ValueError(
                f"continues_in: {r.source} continues in both {next_of[r.source]} and {r.target}"
            )
        if r.target in previous_of:
            raise ValueError(
                f"continues_in: {r.target} continues both {previous_of[r.target]} and {r.source}"
            )
        next_of[r.source] = r.target
        previous_of[r.target] = r.source
    chain_of: dict[str, str] = {}
    for start in next_of:
        if start in previous_of or start in chain_of:
            continue
        current = start
        chain_of[current] = start
        while current in next_of:
            current = next_of[current]
            chain_of[current] = start
    for piece in next_of:
        if piece not in chain_of:
            raise ValueError(f"continues_in: the chain through {piece} is a cycle")
    for r in relations:
        if r.kind is not RelationKind.STAYS_APART:
            continue
        if chain_of.get(r.source, r.source) == chain_of.get(r.target, r.target):
            raise ValueError(
                f"stays_apart {r.source} -> {r.target}: both are pieces of one continues_in chain"
            )


class LayoutItem(_Model):
    id: str
    cls: ElementClass = _json_name("class", "cls")
    page: int = Field(ge=0)
    text: str = ""
    box: Box | None = None
    channel: Channel | None = None
    md_start: int | None = None
    md_end: int | None = None
    chart: Chart | None = None
    cells: list[TableCell] | None = None
    font: Font | None = None
    format: str | None = None
    check_digits_valid: bool | None = None

    @model_validator(mode="after")
    def _check_box(self) -> Self:
        _check_box(self.box, f"layout item {self.id}")
        return self

    @model_validator(mode="after")
    def _check_machine_readable_fields(self) -> Self:
        if self.cls is not ElementClass.MACHINE_READABLE and (
            self.format is not None or self.check_digits_valid is not None
        ):
            raise ValueError(
                f"layout item {self.id}: only a machine_readable item has a format or a check"
            )
        return self

    @model_validator(mode="after")
    def _check_chart(self) -> Self:
        _check_chart_class(self.cls, self.chart, f"layout item {self.id}")
        return self

    @model_validator(mode="after")
    def _check_cells(self) -> Self:
        _check_cells_on_table(self.cls, self.cells, f"layout item {self.id}")
        return self


class Prediction(_Model):
    markdown: str
    layout: list[LayoutItem] = []
    relations: list[Relation] = []
    page_offsets: list[int] | None = None
    unscored_text: list[str] = []


def prediction_schema() -> dict[str, dict]:
    return {"prediction.schema.json": Prediction.model_json_schema()}
