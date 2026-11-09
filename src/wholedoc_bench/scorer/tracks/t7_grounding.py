from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache, partial
from typing import Protocol

from wholedoc_bench.schema import (
    Box,
    Channel,
    Element,
    ElementClass,
    Groundtruth,
    LayoutItem,
)
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.boxes import (
    MIN_IOU,
    Place,
    box_iou,
    box_place,
    box_share_inside,
    item_place,
)
from wholedoc_bench.scorer.utils.groundtruth import cell_texts
from wholedoc_bench.scorer.utils.pairing import Quality, exceeds, pair_in_order, reaches
from wholedoc_bench.scorer.utils.prose import word_f1
from wholedoc_bench.scorer.utils.tracks.table_links import cell_tokens

MIN_WORD_F1 = 0.8
MIN_CELL_SHARE_INSIDE = 0.5
NEIGHBOUR_COVER_LIMIT = 0.5
MAX_CELL_OVERLAP = 0.01
SCORED_CHANNELS = frozenset({Channel.BODY, Channel.FURNITURE})
BOX_ONLY_CLASSES = frozenset(
    {ElementClass.CHART, ElementClass.FIGURE, ElementClass.MACHINE_READABLE}
)


class _OnPage(Protocol):
    @property
    def page(self) -> int: ...


class _Expected(_OnPage, Protocol):
    @property
    def scored(self) -> bool: ...


@dataclass(frozen=True)
class _Match[Truth, Found]:
    pairs: tuple[tuple[Truth, Found], ...]
    scored: tuple[Truth, ...]
    counted: tuple[Found, ...]

    @property
    def f1(self) -> float:
        return 2 * len(self.pairs) / (len(self.scored) + len(self.counted))


@dataclass(frozen=True)
class _Target:
    page: int
    scored: bool
    element: Element
    words: Counter[str]

    @property
    def words_with_box(self) -> Counter[str]:
        return Counter() if self.element.cls in BOX_ONLY_CLASSES else self.words


@dataclass(frozen=True)
class _Found:
    page: int
    item: LayoutItem
    words: Counter[str]


@dataclass(frozen=True)
class _TruthCell:
    page: int
    box: Box
    words: Counter[str]
    scored: bool


@dataclass(frozen=True)
class _FoundCell:
    page: int
    box: Box | None
    text: str
    words: Counter[str]
    covered: tuple[_TruthCell, ...]


def _is_scored(element: Element) -> bool:
    return element.box is not None and element.channel in SCORED_CHANNELS


def _by_page[T: _OnPage](things: Iterable[T]) -> dict[int, list[T]]:
    pages: dict[int, list[T]] = {}
    for thing in things:
        pages.setdefault(thing.page, []).append(thing)
    return pages


def _match[Truth: _Expected, Found: _OnPage](
    truth: Sequence[Truth],
    found: Sequence[Found],
    quality: Callable[[Truth, Found], Quality],
    place: Callable[[Found], tuple[object, ...]],
) -> _Match[Truth, Found]:
    found_by_page = _by_page(found)
    pairs: list[tuple[Truth, Found]] = []
    for page, on_page in _by_page(truth).items():
        columns = found_by_page.get(page, [])
        pairs.extend(pair_in_order(on_page, columns, quality, place, lambda t: t.scored))
    neutral = {id(f) for t, f in pairs if not t.scored}
    return _Match(
        pairs=tuple((t, f) for t, f in pairs if t.scored),
        scored=tuple(t for t in truth if t.scored),
        counted=tuple(f for f in found if id(f) not in neutral),
    )


def _words(*texts: str) -> Counter[str]:
    return cell_tokens([texts])


def _element_words(element: Element) -> Counter[str]:
    return _words(element.text, *cell_texts(element))


def _word_quality(expected: Counter[str], found: Counter[str]) -> Quality:
    f1 = word_f1(expected, found)
    return min(f1, 1.0) if reaches(f1, MIN_WORD_F1) else None


def _quality(target: _Target, found: _Found) -> Quality:
    box, found_box = target.element.box, found.item.box
    if box is None or found_box is None:
        return None if target.scored else _word_quality(target.words, found.words)
    iou = box_iou(box, found_box)
    if not reaches(iou, MIN_IOU):
        return None
    words = target.words_with_box
    if words and _word_quality(words, found.words) is None:
        return None
    return min(iou, 1.0)


def _class_quality(copies: frozenset[int], target: _Target, found: _Found) -> Quality:
    if target.element.cls != found.item.cls and (target.scored or id(found) in copies):
        return None
    return _quality(target, found)


def _place(found: _Found) -> Place:
    return item_place(found.item, found.item.text)


@dataclass(frozen=True)
class _Elements:
    targets: tuple[_Target, ...]
    found: tuple[_Found, ...]

    @cached_property
    def boxes(self) -> _Match[_Target, _Found]:
        return _match(self.targets, self.found, _quality, _place)

    @cached_property
    def classes(self) -> _Match[_Target, _Found]:
        copies = _wrong_class_copies(self.targets, self.found)
        return _match(self.targets, self.found, partial(_class_quality, copies), _place)


def _wrong_class_copies(targets: Sequence[_Target], found: Sequence[_Found]) -> frozenset[int]:
    with_words_by_page = _by_page(t for t in targets if t.scored and t.words_with_box)
    return frozenset(
        id(f)
        for f in found
        if any(
            t.element.cls != f.item.cls and _word_quality(t.words_with_box, f.words) is not None
            for t in with_words_by_page.get(f.page, [])
        )
    )


@lru_cache(maxsize=1)
def _elements(ctx: DocContext) -> _Elements:
    targets = tuple(
        _Target(e.page, _is_scored(e), e, _element_words(e)) for e in ctx.truth.elements
    )
    found = tuple(_Found(item.page, item, _words(item.text)) for item in ctx.pred.layout)
    return _Elements(targets, found)


class ElementBoxes(SubMetric):
    name = "element_boxes"
    source = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        return any(_is_scored(e) for e in truth.elements)

    def score(self, ctx: DocContext) -> float:
        return _elements(ctx).boxes.f1


class ElementClasses(SubMetric):
    name = "element_classes"
    source = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        return any(_is_scored(e) for e in truth.elements)

    def score(self, ctx: DocContext) -> float:
        return _elements(ctx).classes.f1


def _truth_tables(truth: Groundtruth) -> list[Element]:
    return [e for e in truth.elements if e.cls is ElementClass.TABLE]


def _truth_cells(truth: Groundtruth) -> list[_TruthCell]:
    return [
        _TruthCell(e.page, cell.box, _words(cell.text), _is_scored(e))
        for e in _truth_tables(truth)
        for cell in e.cells or []
        if cell.box is not None
    ]


def _covered_cells(box: Box | None, truth_cells: Sequence[_TruthCell]) -> tuple[_TruthCell, ...]:
    if box is None:
        return ()
    return tuple(
        t for t in truth_cells if reaches(box_share_inside(t.box, box), NEIGHBOUR_COVER_LIMIT)
    )


def _found_cells(
    layout: Sequence[LayoutItem], truth_by_page: dict[int, list[_TruthCell]]
) -> list[_FoundCell]:
    found: list[_FoundCell] = []
    for item in layout:
        for cell in item.cells or []:
            if words := _words(cell.text):
                covered = _covered_cells(cell.box, truth_by_page.get(item.page, []))
                found.append(_FoundCell(item.page, cell.box, cell.text, words, covered))
    return found


def cells_overlap(a: Box, b: Box) -> bool:
    return exceeds(max(box_share_inside(a, b), box_share_inside(b, a)), MAX_CELL_OVERLAP)


def _covers_a_neighbour(found: _FoundCell, truth_cell: _TruthCell) -> bool:
    return any(not cells_overlap(c.box, truth_cell.box) for c in found.covered)


def _cell_quality(truth_cell: _TruthCell, found: _FoundCell) -> Quality:
    if found.box is None:
        return None if truth_cell.scored else _word_quality(truth_cell.words, found.words)
    if _covers_a_neighbour(found, truth_cell):
        return None
    share = box_share_inside(found.box, truth_cell.box)
    if not reaches(share, MIN_CELL_SHARE_INSIDE):
        return None
    if _word_quality(truth_cell.words, found.words) is None:
        return None
    return min(share, 1.0)


def _cell_place(cell: _FoundCell) -> tuple[bool, tuple[float, ...], str]:
    return (*box_place(cell.box), cell.text)


@dataclass(frozen=True)
class _TableWithoutCells:
    page: int
    box: Box | None
    words_left: Counter[str]

    def takes(self, cell: _FoundCell) -> bool:
        if self.page != cell.page or not cell.words <= self.words_left:
            return False
        if self.box is None or cell.box is None:
            return True
        return reaches(box_share_inside(cell.box, self.box), MIN_CELL_SHARE_INSIDE)


def _clear_of_scored_cells_or_a_copy(cell: _FoundCell, scored: Sequence[_TruthCell]) -> bool:
    box = cell.box
    return (
        box is None
        or not any(cells_overlap(box, t.box) for t in scored)
        or any(_cell_quality(t, cell) is not None for t in scored)
    )


def _taken_by_tables_without_cells(
    truth: Groundtruth, scored_by_page: dict[int, list[_TruthCell]], left: Iterable[_FoundCell]
) -> set[int]:
    tables = [
        _TableWithoutCells(e.page, e.box, _element_words(e))
        for e in _truth_tables(truth)
        if e.cells is None
    ]
    taken: set[int] = set()
    for cell in sorted(left, key=_cell_place):
        if not _clear_of_scored_cells_or_a_copy(cell, scored_by_page.get(cell.page, [])):
            continue
        table = next((t for t in tables if t.takes(cell)), None)
        if table is not None:
            table.words_left.subtract(cell.words)
            taken.add(id(cell))
    return taken


def _match_cells(
    truth: Groundtruth, layout: Sequence[LayoutItem]
) -> _Match[_TruthCell, _FoundCell]:
    truth_cells = _truth_cells(truth)
    found = _found_cells(layout, _by_page(truth_cells))
    match = _match([c for c in truth_cells if c.words], found, _cell_quality, _cell_place)
    paired = {id(cell) for _, cell in match.pairs}
    left = (cell for cell in match.counted if id(cell) not in paired)
    scored_by_page = _by_page(c for c in truth_cells if c.scored)
    taken = _taken_by_tables_without_cells(truth, scored_by_page, left)
    counted = tuple(cell for cell in match.counted if id(cell) not in taken)
    return _Match(pairs=match.pairs, scored=match.scored, counted=counted)


class TableCellBoxes(SubMetric):
    name = "table_cell_boxes"
    source = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        scored_tables = [e for e in _truth_tables(truth) if _is_scored(e)]
        every_table_has_cells = all(e.cells is not None for e in scored_tables)
        return every_table_has_cells and any(c.scored and c.words for c in _truth_cells(truth))

    def score(self, ctx: DocContext) -> float:
        return _match_cells(ctx.truth, ctx.pred.layout).f1


TRACK = Track(
    id="T7",
    name="Visual grounding",
    parts=(
        (ElementBoxes(), 1.0),
        (ElementClasses(), 0.5),
        (TableCellBoxes(), 0.5),
    ),
)
