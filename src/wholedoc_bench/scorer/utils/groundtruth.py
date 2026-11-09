from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass

from wholedoc_bench.schema import Channel, Element, ElementClass, Groundtruth, RelationKind
from wholedoc_bench.scorer.utils.reading_order import in_reading_order

MAX_VARIANT_EDGES = 3


@dataclass(frozen=True)
class JoinBoundary:
    first_end: int
    second_start: int
    separator: str


def join_boundary(first: str, second: str, *, hyphenated: bool) -> JoinBoundary:
    first_end = len(first.rstrip())
    second_start = len(second) - len(second.lstrip())
    has_hyphen = first[:first_end].endswith("-")
    if hyphenated and has_hyphen:
        first_end -= 1
    return JoinBoundary(first_end, second_start, "" if hyphenated or has_hyphen else " ")


def join_pieces(first: str, second: str, *, hyphenated: bool) -> str:
    boundary = join_boundary(first, second, hyphenated=hyphenated)
    return first[: boundary.first_end] + boundary.separator + second[boundary.second_start :]


def body_texts(truth: Groundtruth, skip: frozenset[ElementClass] = frozenset()) -> list[str]:
    return [variants[0] for variants in body_text_variants(truth, skip)]


def body_text_variants(
    truth: Groundtruth, skip: frozenset[ElementClass] = frozenset()
) -> list[tuple[str, ...]]:
    body = [e for e in truth.elements if e.channel is Channel.BODY and e.cls not in skip]
    by_id = {e.id: e for e in body}
    next_of = next_pieces(truth, by_id)
    hyphenated = {
        r.source
        for r in truth.relations
        if r.kind is RelationKind.CONTINUES_IN
        and r.hyphenated
        and next_of.get(r.source) == r.target
    }
    continued = set(next_of.values())

    blocks: list[tuple[str, ...]] = []
    for e in body:
        blocks.extend((cell,) for cell in cell_texts(e))
        if not e.text or e.id in continued:
            continue
        chain = piece_chain(e.id, next_of)
        pieces = [by_id[i].text for i in chain]
        edges = [previous in hyphenated for previous in chain[:-1]]
        blocks.append(join_variants(pieces, edges))
    return blocks


def next_pieces(truth: Groundtruth, ids: Container[str]) -> dict[str, str]:
    return {
        r.source: r.target
        for r in truth.relations
        if r.kind is RelationKind.CONTINUES_IN and r.source in ids and r.target in ids
    }


def piece_chain(head: str, next_of: Mapping[str, str]) -> list[str]:
    chain = [head]
    while chain[-1] in next_of:
        chain.append(next_of[chain[-1]])
    return chain


def join_choices(pieces: Sequence[str], hyphenated: Sequence[bool]) -> list[frozenset[int]]:
    compound = [i for i, h in enumerate(hyphenated) if not h and pieces[i].rstrip().endswith("-")]
    if len(compound) > MAX_VARIANT_EDGES:
        return [frozenset(), frozenset(compound)]
    return [
        frozenset(i for bit, i in enumerate(compound) if mask >> bit & 1)
        for mask in range(2 ** len(compound))
    ]


def join_variants(pieces: list[str], hyphenated: list[bool]) -> tuple[str, ...]:
    choices = join_choices(pieces, hyphenated)
    result: list[str] = []
    for joined in choices:
        text = pieces[0]
        for i, piece in enumerate(pieces[1:]):
            text = join_pieces(text, piece, hyphenated=hyphenated[i] or i in joined)
        if text not in result:
            result.append(text)
    return tuple(result)


def chains(truth: Groundtruth, keep: Callable[[Element], bool]) -> list[list[Element]]:
    by_id = {e.id: e for e in truth.elements if keep(e)}
    next_of = next_pieces(truth, by_id)
    continued = set(next_of.values())
    heads = [i for i in in_reading_order(truth, by_id) if i not in continued]
    return [[by_id[i] for i in piece_chain(head, next_of)] for head in heads]


def hyphenated_breaks(truth: Groundtruth) -> set[str]:
    return {
        r.source for r in truth.relations if r.kind is RelationKind.CONTINUES_IN and r.hyphenated
    }


def all_texts(truth: Groundtruth, keep: Callable[[Element], bool] = lambda e: True) -> list[str]:
    hyphenated = hyphenated_breaks(truth)
    return [text for chain in chains(truth, keep) for text in chain_texts(chain, hyphenated)]


def chain_texts(chain: list[Element], hyphenated: Container[str]) -> list[str]:
    texts: list[str] = []
    pieces = [e for e in chain if e.text]
    if pieces:
        text = pieces[0].text
        for previous, piece in zip(pieces, pieces[1:], strict=False):
            text = join_pieces(text, piece.text, hyphenated=previous.id in hyphenated)
        texts.append(text)
    for e in chain:
        texts.extend(cell_texts(e))
    return texts


def cell_texts(e: Element) -> list[str]:
    grid = (e.header or []) + (e.rows or [])
    return [c for row in grid for c in row if c]


def table_grid(e: Element) -> tuple[list[list[str]] | None, list[list[str]]]:
    grid = (e.header or []) + (e.rows or [])
    filled = [list(row) for row in grid]
    for span in e.spans or []:
        text = grid[span.row][span.column]
        for r, c in span.positions():
            filled[r][c] = text
    rows = [[c or "" for c in row] for row in filled]
    if e.header is None:
        return None, rows
    return rows[: len(e.header)], rows[len(e.header) :]
