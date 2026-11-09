from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb

from wholedoc_bench.schema import Channel, Element, ElementClass, Groundtruth
from wholedoc_bench.scorer.utils.pairing import best_pairing, reaches
from wholedoc_bench.scorer.utils.prose import edit_similarity
from wholedoc_bench.scorer.utils.reading_order import in_reading_order

Normalize = Callable[[str], str]

MATCH_CUTOFF = 0.5


def _unchanged(text: str) -> str:
    return text


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


def is_heading_element(e: Element) -> bool:
    return e.cls is ElementClass.HEADING and e.channel is Channel.BODY and bool(e.text.strip())


def has_headings(truth: Groundtruth) -> bool:
    return any(is_heading_element(e) for e in truth.elements)


def truth_headings(truth: Groundtruth) -> list[Heading]:
    elements = {e.id: e for e in truth.elements if is_heading_element(e)}
    headings: list[Heading] = []
    for element_id in in_reading_order(truth, elements):
        e = elements[element_id]
        if e.level is None:
            raise ValueError(f"{truth.document.id}: heading {e.id} has no level")
        headings.append(Heading(e.level, e.text))
    return headings


def score_heading_tree(
    truth: Groundtruth, output: Sequence[Heading], *, normalize: Normalize = _unchanged
) -> float:
    expected = truth_headings(truth)
    written = [h for h in output if normalize(h.text).strip()]
    if not expected and not written:
        return 1.0
    pairs = _pairs(expected, written, normalize)
    found = 2 * sum(pairs.values()) / (len(expected) + len(written))
    if len(expected) < 2:
        return found
    return (found + _levels(expected, written, pairs)) / 2


def _pairs(
    expected: Sequence[Heading], written: Sequence[Heading], normalize: Normalize
) -> dict[tuple[int, int], float]:
    a = [normalize(h.text).casefold() for h in expected]
    b = [normalize(h.text).casefold() for h in written]
    similar = [[edit_similarity(x, y) for y in b] for x in a]
    quality = [[s if reaches(s, MATCH_CUTOFF) else None for s in row] for row in similar]
    return {(r, c): similar[r][c] for r, c in best_pairing(quality)}


def _levels(
    expected: Sequence[Heading], written: Sequence[Heading], pairs: dict[tuple[int, int], float]
) -> float:
    output_of = {r: c for r, c in pairs}
    right = sum(
        i in output_of
        and j in output_of
        and _higher(expected[i], expected[j])
        == _higher(written[output_of[i]], written[output_of[j]])
        for i, j in combinations(range(len(expected)), 2)
    )
    return right / comb(len(expected), 2)


def _higher(a: Heading, b: Heading) -> int:
    return (a.level < b.level) - (a.level > b.level)
