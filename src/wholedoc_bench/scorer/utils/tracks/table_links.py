from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction

from wholedoc_bench.schema import CellSpan, Element, ElementClass, Groundtruth, LayoutItem
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.blocks import Cells, Grid, MarkdownTable, normalize_text
from wholedoc_bench.scorer.utils.groundtruth import next_pieces, piece_chain, table_grid
from wholedoc_bench.scorer.utils.prose import words


def plain_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(Decimal(repr(value)), "f")


def chart_anchor_text(element: Element | LayoutItem) -> str:
    if element.chart is None:
        return element.text
    words = [element.text]
    for series in element.chart.series:
        words.append(series.name)
        for point in series.points:
            words += [point.label, plain_number(point.value)]
    return " ".join(words)


def cell_tokens(rows: Iterable[Iterable[str]]) -> Counter[str]:
    return Counter(word.casefold() for row in rows for cell in row for word in words(cell))


def text_tokens(text: str) -> Counter[str]:
    return cell_tokens([[normalize_text(text)]])


def token_similarity(a: Counter[str], b: Counter[str]) -> Fraction:
    total = a.total() + b.total()
    if total == 0:
        return Fraction(0)
    return Fraction(2 * (a & b).total(), total)


@dataclass(frozen=True)
class TruthTable:
    element_ids: tuple[str, ...]
    pages: tuple[int, ...]
    header: Grid | None
    rows: Grid
    break_rows: tuple[int, ...]
    merged: tuple[CellSpan, ...] = ()

    @property
    def grid(self) -> Grid:
        return (*(self.header or ()), *self.rows)


class AnchorSource(StrEnum):
    LAYOUT = "layout"
    GROUNDTRUTH = "groundtruth"


AnchorKey = tuple[AnchorSource | None, str | None]


@dataclass(frozen=True, eq=False)
class TableAnchor:
    id: str
    kind: ElementClass
    tokens: Counter[str]
    source: AnchorSource
    md_start: int | None = None
    md_end: int | None = None


@dataclass(frozen=True)
class TableLink:
    table: MarkdownTable
    kind: ElementClass
    anchor_id: str | None
    source: AnchorSource | None = None

    @property
    def anchor_key(self) -> AnchorKey:
        return self.source, self.anchor_id


def _normalized_cells(cells: Iterable[str]) -> Cells:
    return tuple(normalize_text(cell) for cell in cells)


def parse_truth_tables(truth: Groundtruth) -> tuple[TruthTable, ...]:
    tables_by_id = {e.id: e for e in truth.elements if e.cls is ElementClass.TABLE}
    next_piece = next_pieces(truth, tables_by_id)
    continuations = set(next_piece.values())
    heads = [e for e in tables_by_id.values() if e.id not in continuations]
    tables = (_join_chain(head, next_piece, tables_by_id) for head in heads)
    return tuple(t for t in tables if t.header is not None or t.rows)


def _join_chain(
    head: Element, next_piece: dict[str, str], tables_by_id: dict[str, Element]
) -> TruthTable:
    pieces = [tables_by_id[i] for i in piece_chain(head.id, next_piece)]
    header = table_grid(head)[0]
    above = len(header or [])
    rows: list[Cells] = []
    break_rows: list[int] = []
    merged: list[CellSpan] = []
    for piece in pieces:
        if piece is not head:
            break_rows.append(len(rows))
        merged.extend(_joined_spans(piece, piece is head, above + len(rows)))
        rows.extend(_normalized_cells(row) for row in table_grid(piece)[1])
    return TruthTable(
        element_ids=tuple(p.id for p in pieces),
        pages=tuple(p.page for p in pieces),
        header=tuple(map(_normalized_cells, header)) if header else None,
        rows=tuple(rows),
        break_rows=tuple(break_rows),
        merged=tuple(sorted(merged, key=lambda span: (span.row, span.column))),
    )


def _joined_spans(piece: Element, keeps_header: bool, first_row: int) -> Iterator[CellSpan]:
    header_rows = len(piece.header or [])
    for span in piece.spans or []:
        if span.row < header_rows:
            if keeps_header:
                yield span
            continue
        yield span.model_copy(update={"row": first_row + span.row - header_rows})


def layout_anchors(layout: Iterable[LayoutItem]) -> tuple[TableAnchor, ...]:
    return tuple(
        TableAnchor(
            item.id,
            item.cls,
            text_tokens(item.text),
            AnchorSource.LAYOUT,
            item.md_start,
            item.md_end,
        )
        for item in layout
        if item.cls in (ElementClass.TABLE, ElementClass.CHART)
    )


def truth_anchors(
    truth: Groundtruth, truth_tables: Iterable[TruthTable]
) -> tuple[TableAnchor, ...]:
    table_by_first_id = {t.element_ids[0]: t for t in truth_tables}
    anchors: list[TableAnchor] = []
    for element in truth.elements:
        if (table := table_by_first_id.get(element.id)) is not None:
            tokens = cell_tokens(table.grid)
            anchors.append(
                TableAnchor(element.id, ElementClass.TABLE, tokens, AnchorSource.GROUNDTRUTH)
            )
        elif element.cls is ElementClass.CHART:
            tokens = text_tokens(chart_anchor_text(element))
            anchors.append(
                TableAnchor(element.id, ElementClass.CHART, tokens, AnchorSource.GROUNDTRUTH)
            )
    return tuple(anchors)


def linked_tables(ctx: DocContext) -> tuple[TableLink, ...]:
    tables = [block.table for block in ctx.blocks if block.table is not None]
    return link_tables(
        tables,
        layout_anchors(ctx.pred.layout),
        truth_anchors(ctx.truth, parse_truth_tables(ctx.truth)),
    )


def link_tables(
    tables: Sequence[MarkdownTable],
    layout: Sequence[TableAnchor],
    groundtruth: Sequence[TableAnchor],
) -> tuple[TableLink, ...]:
    links = list(_anchor_links(tables, layout))
    unlinked = [t for t, link in enumerate(links) if link.anchor_id is None]
    fallback = _anchor_links([tables[t] for t in unlinked], groundtruth)
    for t, link in zip(unlinked, fallback, strict=True):
        links[t] = link
    return tuple(links)


def _anchor_links(
    tables: Sequence[MarkdownTable], anchors: Sequence[TableAnchor]
) -> tuple[TableLink, ...]:
    table_tokens = [cell_tokens(t.rows) for t in tables]
    linked = _span_links(tables, anchors)
    linked |= _residual_links(table_tokens, anchors, linked)
    for t, tokens in enumerate(table_tokens):
        if t not in linked and (a := _most_similar_anchor(tokens, anchors)) is not None:
            linked[t] = a
    return tuple(_link(table, linked.get(t), anchors) for t, table in enumerate(tables))


def _link(table: MarkdownTable, anchor: int | None, anchors: Sequence[TableAnchor]) -> TableLink:
    if anchor is None:
        return TableLink(table, ElementClass.TABLE, None)
    linked = anchors[anchor]
    return TableLink(table, linked.kind, linked.id, linked.source)


def _span_links(tables: Sequence[MarkdownTable], anchors: Sequence[TableAnchor]) -> dict[int, int]:
    links = ((t, _span_anchor(table, anchors)) for t, table in enumerate(tables))
    return {t: a for t, a in links if a is not None}


def _span_anchor(table: MarkdownTable, anchors: Sequence[TableAnchor]) -> int | None:
    best: int | None = None
    best_overlap = 0
    for a, anchor in enumerate(anchors):
        if anchor.md_start is None or anchor.md_end is None:
            continue
        overlap = min(table.end, anchor.md_end) - max(table.start, anchor.md_start)
        if overlap > best_overlap:
            best, best_overlap = a, overlap
    return best


def _residual_links(
    table_tokens: Sequence[Counter[str]],
    anchors: Sequence[TableAnchor],
    span_links: dict[int, int],
) -> dict[int, int]:
    residuals = [anchor.tokens.copy() for anchor in anchors]
    for t, a in span_links.items():
        residuals[a] -= table_tokens[t]
    similarity = {
        t: [token_similarity(tokens, residual) for residual in residuals]
        for t, tokens in enumerate(table_tokens)
        if t not in span_links
    }
    links: dict[int, int] = {}
    while (pair := _most_similar_pair(similarity)) is not None:
        t, a = pair
        links[t] = a
        del similarity[t]
        residuals[a] -= table_tokens[t]
        for other, row in similarity.items():
            row[a] = token_similarity(table_tokens[other], residuals[a])
    return links


def _most_similar_pair(similarity: dict[int, list[Fraction]]) -> tuple[int, int] | None:
    ranked = ((value, -t, -a) for t, row in similarity.items() for a, value in enumerate(row))
    best = max(ranked, default=None)
    if best is None or best[0] == 0:
        return None
    return -best[1], -best[2]


def _most_similar_anchor(tokens: Counter[str], anchors: Sequence[TableAnchor]) -> int | None:
    similarity = [token_similarity(tokens, anchor.tokens) for anchor in anchors]
    best = max(range(len(similarity)), key=lambda a: (similarity[a], -a), default=None)
    if best is None or similarity[best] == 0:
        return None
    return best
