from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import pairwise
from statistics import fmean
from typing import Self

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from wholedoc_bench.schema import CellSpan, ElementClass, Groundtruth, RelationKind
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.blocks import Cells, Grid, MarkdownTable
from wholedoc_bench.scorer.utils.prose import words
from wholedoc_bench.scorer.utils.tracks.t4_tables.table_matching import (
    casefolded_grid,
    column_labels,
    header_and_body,
    match_tables,
    one_to_one,
    output_links,
    output_tables,
)
from wholedoc_bench.scorer.utils.tracks.table_links import (
    AnchorSource,
    TableLink,
    TruthTable,
    cell_tokens,
    chart_anchor_text,
    linked_tables,
    parse_truth_tables,
    text_tokens,
    token_similarity,
)
from wholedoc_bench.scorer.utils.tree_edit import NODE_EDIT_COST, Costs, Tree
from wholedoc_bench.scorer.utils.tree_edit import tree_edit_distance as ordered_tree_edit_distance


def _group[K, V](pairs: Iterable[tuple[K, V]]) -> dict[K, tuple[V, ...]]:
    groups: defaultdict[K, list[V]] = defaultdict(list)
    for key, value in pairs:
        groups[key].append(value)
    return {key: tuple(values) for key, values in groups.items()}


def _piece_spans(table: TruthTable) -> list[tuple[int, int]]:
    return list(pairwise((0, *table.break_rows, len(table.rows))))


def _cell_at(row: Cells, column: int) -> str:
    return row[column] if 0 <= column < len(row) else ""


class NodeKind(StrEnum):
    TABLE = "table"
    ROW = "row"
    CELL = "cell"


_FREE_RENAME_KINDS = (NodeKind.TABLE, NodeKind.ROW)
_CELL_ROWS_PER_CDIST = 256


class Teds(SubMetric):
    name = "teds"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return bool(parse_truth_tables(truth))

    def score(self, ctx: DocContext) -> float:
        truth_tables = parse_truth_tables(ctx.truth)
        matches = match_tables(truth_tables, output_tables(linked_tables(ctx)))
        pairs = zip(truth_tables, matches, strict=True)
        return fmean(
            [
                0.0 if match is None else teds(t.grid, match.rows, t.merged, match.merged)
                for t, match in pairs
            ]
        )


def teds(
    truth: Grid,
    output: Grid,
    truth_merged: Sequence[CellSpan] = (),
    output_merged: Sequence[CellSpan] = (),
) -> float:
    if truth == output and tuple(truth_merged) == tuple(output_merged):
        return 1.0
    a, b = TableTree.of(truth, truth_merged), TableTree.of(output, output_merged)
    return max(0.0, 1.0 - tree_edit_distance(a, b) / max(a.size, b.size))


@dataclass(frozen=True)
class TableTree:
    kinds: tuple[NodeKind, ...]
    texts: tuple[str, ...]
    leftmost: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]

    @classmethod
    def of(cls, rows: Grid, merged: Sequence[CellSpan] = ()) -> Self:
        span_at = {(span.row, span.column): (span.rows, span.columns) for span in merged}
        covered = {position for span in merged for position in span.positions()[1:]}
        kinds: list[NodeKind] = []
        texts: list[str] = []
        leftmost: list[int] = []
        spans: list[tuple[int, int]] = []
        for r, row in enumerate(rows):
            first = len(kinds)
            for c, cell in enumerate(row):
                if (r, c) in covered:
                    continue
                leftmost.append(len(kinds))
                kinds.append(NodeKind.CELL)
                texts.append(cell)
                spans.append(span_at.get((r, c), (1, 1)))
            leftmost.append(first)
            kinds.append(NodeKind.ROW)
            texts.append("")
            spans.append((0, 0))
        leftmost.append(0)
        kinds.append(NodeKind.TABLE)
        texts.append("")
        spans.append((0, 0))
        return cls(tuple(kinds), tuple(texts), tuple(leftmost), tuple(spans))

    @property
    def size(self) -> int:
        return len(self.kinds)

    def nodes_of(self, kind: NodeKind) -> list[int]:
        return [node for node, node_kind in enumerate(self.kinds) if node_kind is kind]


def tree_edit_distance(a: TableTree, b: TableTree) -> float:
    if a.size > b.size:
        a, b = b, a
    return ordered_tree_edit_distance(
        Tree(a.texts, a.leftmost), Tree(b.texts, b.leftmost), _rename_costs(a, b)
    )


def _rename_costs(a: TableTree, b: TableTree) -> Costs:
    costs = np.full((a.size, b.size), NODE_EDIT_COST)
    for kind in _FREE_RENAME_KINDS:
        costs[np.ix_(a.nodes_of(kind), b.nodes_of(kind))] = 0.0
    a_cells, b_cells = a.nodes_of(NodeKind.CELL), b.nodes_of(NodeKind.CELL)
    b_texts = [b.texts[node] for node in b_cells]
    b_spans = np.array([b.spans[node] for node in b_cells]).reshape(-1, 2)
    a_spans = np.array([a.spans[node] for node in a_cells]).reshape(-1, 2)
    merged = bool((a_spans != 1).any() or (b_spans != 1).any())
    for chunk_start in range(0, len(a_cells), _CELL_ROWS_PER_CDIST):
        chunk = a_cells[chunk_start : chunk_start + _CELL_ROWS_PER_CDIST]
        chunk_costs = cdist(
            [a.texts[node] for node in chunk],
            b_texts,
            scorer=Levenshtein.normalized_distance,
            dtype=np.float64,
        )
        if merged:
            chunk_spans = a_spans[chunk_start : chunk_start + len(chunk)]
            other_span = (chunk_spans[:, None, :] != b_spans[None, :, :]).any(axis=2)
            chunk_costs[other_span] = NODE_EDIT_COST
        costs[np.ix_(chunk, b_cells)] = chunk_costs
    return costs


class HeaderMatchedRows(SubMetric):
    name = "header_matched_rows"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return bool(_expected_tables(parse_truth_tables(truth)))

    def score(self, ctx: DocContext) -> float:
        tables = [_HeadedTable.of(table) for table in output_tables(linked_tables(ctx))]
        candidates = [
            slots
            for expected in _expected_tables(parse_truth_tables(ctx.truth))
            for slots in _candidate_slots(expected, tables)
        ]
        return _maximum_matching(candidates) / len(candidates)


@dataclass(frozen=True)
class _ExpectedCell:
    column: int
    text: str


_ExpectedRow = tuple[_ExpectedCell, ...]
_ColumnMap = tuple[int | None, ...]
_Slot = tuple[int, int]
_Positions = Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class _ExpectedTable:
    labels: tuple[tuple[str, ...], ...]
    rows: tuple[_ExpectedRow, ...]

    @classmethod
    def of(cls, table: TruthTable) -> Self:
        header = casefolded_grid(table.header or ())
        if not any(map(any, header)):
            return cls((), ())
        width = max(map(len, (*header, *table.rows)))
        rows = (
            tuple(_ExpectedCell(column, cell.casefold()) for column, cell in enumerate(row) if cell)
            for row in table.rows
        )
        labels = (header_labels(header, column) for column in range(width))
        return cls(tuple(labels), tuple(row for row in rows if row))


def header_labels(header: Grid | None, column: int) -> tuple[str, ...]:
    return tuple(dict.fromkeys(label for row in header or () if (label := _cell_at(row, column))))


def _joins_labels(text: str, labels: Sequence[str]) -> bool:
    if len(labels) < 2:
        return False
    cell, start = words(text), 0
    for label in labels:
        part = words(label)
        found = next(
            (i for i in range(start, len(cell) - len(part) + 1) if cell[i : i + len(part)] == part),
            None,
        )
        if not part or found is None:
            return False
        start = found + len(part)
    return True


def _shared_labels(labels: Sequence[str], column: Set[str]) -> int:
    if any(_joins_labels(text, labels) for text in column):
        return len(labels)
    return sum(label in column for label in labels)


def _expected_tables(truth_tables: Iterable[TruthTable]) -> list[_ExpectedTable]:
    tables = (_ExpectedTable.of(table) for table in truth_tables)
    return [table for table in tables if table.rows]


@dataclass(frozen=True)
class _HeadedTable:
    labels: tuple[frozenset[str], ...]
    rows_by_text: tuple[_Positions, ...]
    body: Grid

    @classmethod
    def of(cls, table: MarkdownTable) -> Self:
        labels = column_labels(table)
        _, body = header_and_body(table)
        folded = casefolded_grid(body)
        width = max([len(labels), *map(len, folded)])
        return cls(
            labels,
            tuple(
                _group((row[column], r) for r, row in enumerate(folded) if column < len(row))
                for column in range(width)
            ),
            folded,
        )

    def column_map(self, labels: Sequence[Sequence[str]]) -> _ColumnMap:
        shared = {
            (index, column): count
            for index, label_set in enumerate(labels)
            for column, output in enumerate(self.labels)
            if (count := _shared_labels(label_set, output))
        }
        pairs = dict(one_to_one(sorted(shared, key=lambda pair: (-shared[pair], *pair))))
        used = set(pairs.values())
        unnamed = iter([column for column in range(len(self.rows_by_text)) if column not in used])
        return tuple(
            pairs.get(index) if label_set else next(unnamed, None)
            for index, label_set in enumerate(labels)
        )

    def fitting_rows(self, row: _ExpectedRow, columns: _ColumnMap) -> list[int]:
        targets = _targets(row, columns)
        if targets is None:
            return []
        candidates = min(
            (
                self.rows_by_text[target].get(cell.text, ())
                for cell, target in zip(row, targets, strict=True)
            ),
            key=len,
        )
        return [r for r in candidates if self._fits(row, targets, self.body[r])]

    @staticmethod
    def _fits(row: _ExpectedRow, targets: tuple[int, ...], body_row: Cells) -> bool:
        return all(
            target < len(body_row) and body_row[target] == cell.text
            for cell, target in zip(row, targets, strict=True)
        )


def _targets(row: _ExpectedRow, columns: _ColumnMap) -> tuple[int, ...] | None:
    targets = [columns[cell.column] for cell in row]
    known = tuple(target for target in targets if target is not None)
    return known if len(known) == len(targets) else None


def _candidate_slots(expected: _ExpectedTable, tables: Sequence[_HeadedTable]) -> list[list[_Slot]]:
    maps = [table.column_map(expected.labels) for table in tables]
    return [
        [
            (t, r)
            for t, (table, columns) in enumerate(zip(tables, maps, strict=True))
            for r in table.fitting_rows(row, columns)
        ]
        for row in expected.rows
    ]


def _maximum_matching(candidates: Sequence[Sequence[_Slot]]) -> int:
    slot_of: dict[int, _Slot] = {}
    owner: dict[_Slot, int] = {}
    unreachable: set[_Slot] = set()
    matched = 0
    for start in range(len(candidates)):
        reached_from, free = _search(start, candidates, owner, unreachable)
        if free is None:
            unreachable.update(reached_from)
            continue
        _flip_path(free, reached_from, slot_of, owner)
        unreachable.clear()
        matched += 1
    return matched


def _search(
    start: int,
    candidates: Sequence[Sequence[_Slot]],
    owner: Mapping[_Slot, int],
    unreachable: Set[_Slot],
) -> tuple[dict[_Slot, int], _Slot | None]:
    reached_from: dict[_Slot, int] = {}
    queue = deque([start])
    while queue:
        row = queue.popleft()
        for slot in candidates[row]:
            if slot in reached_from or slot in unreachable:
                continue
            reached_from[slot] = row
            if slot not in owner:
                return reached_from, slot
            queue.append(owner[slot])
    return reached_from, None


def _flip_path(
    end: _Slot,
    reached_from: Mapping[_Slot, int],
    slot_of: dict[int, _Slot],
    owner: dict[_Slot, int],
) -> None:
    slot: _Slot | None = end
    while slot is not None:
        row = reached_from[slot]
        previous = slot_of.get(row)
        owner[slot] = row
        slot_of[row] = slot
        slot = previous


MIN_CELL_SHARE = 0.5
MIN_PIECE_SHARE = 0.5


class TableIntegrity(SubMetric):
    name = "table_integrity"
    source = Source.BOTH

    def applies(self, truth: Groundtruth) -> bool:
        return any(_row_pieces(table) for table in parse_truth_tables(truth))

    def score(self, ctx: DocContext) -> float:
        truth_tables = parse_truth_tables(ctx.truth)
        truth = [(table, pieces) for table in truth_tables if (pieces := _row_pieces(table))]
        truth_pieces = [pieces for _, pieces in truth]
        links = linked_tables(ctx)
        units = _output_units(ctx, output_links(links))
        fits = {
            (unit, index): _fit(rows, pieces, frozenset(casefolded_grid(table.header or ())))
            for unit, rows in enumerate(units)
            for index, (table, pieces) in enumerate(truth)
        }
        candidates = {
            (unit, table): fit
            for (unit, table), fit in fits.items()
            if all(fit.covered) and not _joins_another_table(units[unit], fit, table, truth_pieces)
        }
        matches = one_to_one(
            sorted(candidates, key=lambda pair: (-candidates[pair].quality, *pair))
        )
        if not matches:
            return 0.0
        precision = len(matches) / (len(units) + _unmatched_chart_tables(ctx.truth, links))
        recall = len(matches) / len(truth_pieces)
        return 2 * precision * recall / (precision + recall)


def _row_pieces(table: TruthTable) -> list[Grid]:
    slices = (casefolded_grid(table.rows[start:end]) for start, end in _piece_spans(table))
    pieces = (tuple(row for row in rows if any(row)) for rows in slices)
    return [piece for piece in pieces if piece]


def _output_units(ctx: DocContext, links: Sequence[TableLink]) -> list[Grid]:
    first = _first_table_by_anchor(ctx, links)
    labels = list(range(len(links)))
    for relation in ctx.pred.relations:
        if relation.kind is not RelationKind.CONTINUES_IN:
            continue
        earlier, later = first.get(relation.source), first.get(relation.target)
        if earlier is None or later is None or earlier >= later:
            continue
        keep, drop = sorted((labels[earlier], labels[later]))
        labels = [keep if label == drop else label for label in labels]
    units: dict[int, list[Cells]] = {}
    for label, link in zip(labels, links, strict=True):
        units.setdefault(label, []).extend(link.table.rows)
    return [casefolded_grid(tuple(rows)) for rows in units.values()]


def _first_table_by_anchor(ctx: DocContext, links: Sequence[TableLink]) -> dict[str, int]:
    joinable_ids = {item.id for item in ctx.pred.layout if item.cls is ElementClass.TABLE}
    first: dict[str, int] = {}
    for index, link in enumerate(links):
        if link.source is AnchorSource.LAYOUT and link.anchor_id in joinable_ids:
            first.setdefault(link.anchor_id, index)
    return first


def _unmatched_chart_tables(truth: Groundtruth, links: Sequence[TableLink]) -> int:
    truth_charts = [
        text_tokens(chart_anchor_text(element))
        for element in truth.elements
        if element.cls is ElementClass.CHART
    ]
    chart_links = (link for link in links if link.kind is ElementClass.CHART)
    return sum(
        not any(token_similarity(cell_tokens(link.table.rows), chart) for chart in truth_charts)
        for link in chart_links
    )


@dataclass(frozen=True)
class _TruthRow:
    piece: int
    filled: tuple[tuple[int, str], ...]

    @classmethod
    def of(cls, piece: int, cells: Cells) -> Self:
        return cls(piece, tuple((column, cell) for column, cell in enumerate(cells) if cell))


@dataclass(frozen=True)
class _Fit:
    quality: Fraction
    covered: tuple[bool, ...]
    used_rows: frozenset[int]


def _fit(unit: Grid, pieces: list[Grid], header: Set[Cells] = frozenset()) -> _Fit:
    truth_rows = [
        _TruthRow.of(index, cells) for index, piece in enumerate(pieces) for cells in piece
    ]
    shares = _cell_shares(truth_rows, unit)
    pairs = one_to_one(sorted(shares, key=lambda pair: (-shares[pair], *pair)))
    found = Counter(truth_rows[truth_index].piece for truth_index, _ in pairs)
    covered = tuple(
        found[index] >= MIN_PIECE_SHARE * len(piece) for index, piece in enumerate(pieces)
    )
    quality = sum((shares[pair] for pair in pairs), Fraction()) / len(truth_rows)
    header_rows = {unit_index for unit_index, row in enumerate(unit) if row in header}
    return _Fit(quality, covered, frozenset(unit_index for _, unit_index in pairs) | header_rows)


def _cell_shares(truth_rows: Sequence[_TruthRow], unit: Grid) -> dict[tuple[int, int], Fraction]:
    rows_by_cell = _group(
        ((column, cell), r) for r, row in enumerate(unit) for column, cell in enumerate(row) if cell
    )
    shares: dict[tuple[int, int], Fraction] = {}
    for truth_index, truth_row in enumerate(truth_rows):
        filled = truth_row.filled
        equal = Counter(unit_index for key in filled for unit_index in rows_by_cell.get(key, ()))
        for unit_index, count in equal.items():
            if count >= MIN_CELL_SHARE * len(filled):
                shares[truth_index, unit_index] = Fraction(count, len(filled))
    return shares


def _joins_another_table(unit: Grid, fit: _Fit, table: int, truth_pieces: list[list[Grid]]) -> bool:
    remaining = tuple(row for index, row in enumerate(unit) if index not in fit.used_rows)
    return any(
        any(_fit(remaining, pieces).covered)
        for other, pieces in enumerate(truth_pieces)
        if other != table
    )


ROWS_AFTER_BREAK = 3


class NeighbourRuleKind(StrEnum):
    UNDER_HEADER = "under_header"
    LEFT_OF = "left_of"
    BELOW = "below"
    NOT_BELOW = "not_below"


@dataclass(frozen=True)
class NeighbourRule:
    kind: NeighbourRuleKind
    cell: str
    other: str
    more_labels: tuple[str, ...] = ()


_KIND_ORDER = {kind: order for order, kind in enumerate(NeighbourRuleKind)}

_PlacedRule = tuple[int, int, NeighbourRule]
_Position = tuple[int, int]


class CellNeighbourTests(SubMetric):
    name = "cell_neighbour_tests"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return bool(table_neighbour_rules(truth))

    def score(self, ctx: DocContext) -> float:
        views = [_NeighbourView.of(table) for table in output_tables(linked_tables(ctx))]
        rules = table_neighbour_rules(ctx.truth)
        return sum(_passes(rule, views) for rule in rules) / len(rules)


def table_neighbour_rules(truth: Groundtruth) -> tuple[NeighbourRule, ...]:
    tables = parse_truth_tables(truth)
    counts = Counter(cell.casefold() for table in tables for row in table.grid for cell in row)

    def is_unique(text: str) -> bool:
        return bool(text) and counts[text.casefold()] == 1

    rows_kept_apart = _rows_kept_apart_above(truth, tables)
    rules: list[NeighbourRule] = []
    for table in tables:
        above_first_row = rows_kept_apart.get(table.element_ids[0], [])
        placed = [
            *_break_rules(table, is_unique),
            *_not_below_rules(table, above_first_row, is_unique),
        ]
        placed.sort(key=lambda p: (p[0], p[1], _KIND_ORDER[p[2].kind]))
        rules.extend(rule for _, _, rule in placed)
    return tuple(rules)


def _break_rules(table: TruthTable, is_unique: Callable[[str], bool]) -> Iterator[_PlacedRule]:
    for r in _checked_rows(table):
        row = table.rows[r]
        above = table.rows[r - 1] if r > 0 else ()
        for j, cell in enumerate(row):
            if not is_unique(cell):
                continue
            if labels := header_labels(table.header, j):
                top, *lower = labels
                rule = NeighbourRule(NeighbourRuleKind.UNDER_HEADER, cell, top, tuple(lower))
                yield r, j, rule
            if right := _cell_at(row, j + 1):
                yield r, j, NeighbourRule(NeighbourRuleKind.LEFT_OF, cell, right)
            if upper := _cell_at(above, j):
                yield r, j, NeighbourRule(NeighbourRuleKind.BELOW, cell, upper)


def _checked_rows(table: TruthTable) -> Iterator[int]:
    for start, end in _piece_spans(table)[1:]:
        yield from range(start, min(start + ROWS_AFTER_BREAK, end))


def _not_below_rules(
    table: TruthTable, rows_above: Sequence[Cells], is_unique: Callable[[str], bool]
) -> Iterator[_PlacedRule]:
    first_row = table.rows[0] if table.rows else ()
    for j, cell in enumerate(first_row):
        if not is_unique(cell):
            continue
        for row_above in rows_above:
            if is_unique(upper := _cell_at(row_above, j)):
                yield 0, j, NeighbourRule(NeighbourRuleKind.NOT_BELOW, cell, upper)


def _rows_kept_apart_above(
    truth: Groundtruth, tables: Sequence[TruthTable]
) -> dict[str, list[Cells]]:
    last_rows = {piece: row for table in tables for piece, row in _piece_last_rows(table)}
    first_pieces = {table.element_ids[0] for table in tables}
    sources_above: defaultdict[str, dict[str, Cells]] = defaultdict(dict)
    for relation in truth.relations:
        if (
            relation.kind is RelationKind.STAYS_APART
            and relation.source in last_rows
            and relation.target in first_pieces
        ):
            sources_above[relation.target][relation.source] = last_rows[relation.source]
    return {target: list(rows.values()) for target, rows in sources_above.items()}


def _piece_last_rows(table: TruthTable) -> Iterator[tuple[str, Cells]]:
    for piece, (start, end) in zip(table.element_ids, _piece_spans(table), strict=True):
        if end > start:
            yield piece, table.rows[end - 1]


@dataclass(frozen=True)
class _NeighbourView:
    rows: Grid
    body_start: int
    labels: tuple[frozenset[str], ...]
    positions: dict[str, tuple[_Position, ...]]

    @classmethod
    def of(cls, table: MarkdownTable) -> Self:
        header, body = header_and_body(table)
        rows = casefolded_grid(header + body)
        return cls(
            rows=rows,
            body_start=len(header),
            labels=column_labels(table),
            positions=_group(
                (cell, (r, c)) for r, row in enumerate(rows) for c, cell in enumerate(row)
            ),
        )

    def cell_at(self, r: int, c: int) -> str:
        return _cell_at(self.rows[r], c) if 0 <= r < len(self.rows) else ""

    def holds(self, text: str) -> bool:
        return text in self.positions

    def is_under_header(self, position: _Position, labels: Sequence[str]) -> bool:
        r, c = position
        if r < self.body_start or c >= len(self.labels):
            return False
        if any(_joins_labels(text, labels) for text in self.labels[c]):
            return True
        present = [label for label in labels if any(label in column for column in self.labels)]
        return bool(present) and present[-1] in self.labels[c]

    def is_left_of(self, position: _Position, other: str) -> bool:
        r, c = position
        return self.cell_at(r, c + 1) == other

    def is_below(self, position: _Position, other: str) -> bool:
        r, c = position
        return self.cell_at(r - 1, c) == other


def _any_position(
    views: Sequence[_NeighbourView],
    cell: str,
    check: Callable[[_NeighbourView, _Position], bool],
) -> bool:
    return any(check(view, p) for view in views for p in view.positions.get(cell, ()))


def _is_in_output(views: Sequence[_NeighbourView], text: str) -> bool:
    return any(view.holds(text) for view in views)


def _passes(rule: NeighbourRule, views: Sequence[_NeighbourView]) -> bool:
    cell, other = rule.cell.casefold(), rule.other.casefold()
    match rule.kind:
        case NeighbourRuleKind.UNDER_HEADER:
            labels = [other, *(label.casefold() for label in rule.more_labels)]
            return _any_position(views, cell, lambda v, p: v.is_under_header(p, labels))
        case NeighbourRuleKind.LEFT_OF:
            return _any_position(views, cell, lambda v, p: v.is_left_of(p, other))
        case NeighbourRuleKind.BELOW:
            return _any_position(views, cell, lambda v, p: v.is_below(p, other))
        case NeighbourRuleKind.NOT_BELOW:
            both_present = _is_in_output(views, cell) and _is_in_output(views, other)
            return both_present and not any(
                view.holds(cell) and view.holds(other) for view in views
            )


TRACK = Track(
    id="T4",
    name="Tables",
    parts=(
        (Teds(), 1.0),
        (HeaderMatchedRows(), 1.0),
        (TableIntegrity(), 1.0),
        (CellNeighbourTests(), 0.5),
    ),
)
