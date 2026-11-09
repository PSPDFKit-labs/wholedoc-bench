from dataclasses import dataclass, replace
from itertools import accumulate

from fuzzysearch import Match, find_near_matches
from rapidfuzz.distance import Levenshtein

from wholedoc_bench.schema import Channel, Element, ElementClass, Groundtruth
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.groundtruth import (
    all_texts,
    chains,
    hyphenated_breaks,
    join_pieces,
)
from wholedoc_bench.scorer.utils.prose import sentence_language, sentences, words
from wholedoc_bench.scorer.utils.reading_order import in_reading_order

UNIT_CLASSES = frozenset({ElementClass.HEADING, ElementClass.PARAGRAPH, ElementClass.LIST_ITEM})
CHARACTERS_PER_EDIT = 20
MAX_EDITS = 15


@dataclass(frozen=True)
class OrderTest:
    before: str
    after: str


def order_tests(truth: Groundtruth) -> list[OrderTest]:
    return _analysis(truth)[0]


def locating_sentences(truth: Groundtruth) -> list[str]:
    return _analysis(truth)[1]


_ANALYSED: dict[int, tuple[Groundtruth, tuple[list[OrderTest], list[str]]]] = {}
_ANALYSED_MAX = 8


def _analysis(truth: Groundtruth) -> tuple[list[OrderTest], list[str]]:
    hit = _ANALYSED.get(id(truth))
    if hit is not None and hit[0] is truth:
        return hit[1]
    doc = _document(truth)
    result = (_order_tests(doc), _locating_sentences(doc))
    if len(_ANALYSED) >= _ANALYSED_MAX:
        _ANALYSED.clear()
    _ANALYSED[id(truth)] = (truth, result)
    return result


def _order_tests(doc: "_Document") -> list[OrderTest]:
    pairs: list[tuple[int, int]] = []
    for unit, following in zip(doc.units, doc.units[1:], strict=False):
        pairs.append((unit.last, following.first))
    for unit in doc.units:
        pairs.extend(unit.breaks)
    tests: list[OrderTest] = []
    for before, after in sorted(set(pairs)):
        a, b = _glued(doc, before, -1), _glued(doc, after, 1)
        if a is not None and b is not None:
            tests.append(OrderTest(a, b))
    return list(dict.fromkeys(tests))


def _locating_sentences(doc: "_Document") -> list[str]:
    result: list[str] = []
    for unit in doc.units:
        starts = (_trim(doc.sentences[i], keep_end=False) for i in range(unit.first, unit.last + 1))
        found = next((start for start in starts if _unique(start, doc)), None)
        found = found or _glued(doc, unit.first, 1)
        if found is not None:
            result.append(found)
    return result


def output_order(locating: list[str], output: str) -> list[int]:
    placed = [
        (copies[0], i)
        for i, sentence in enumerate(locating)
        if len(copies := _copies(sentence, output)) == 1
    ]
    return [i for _, i in sorted(placed)]


def passes(test: OrderTest, output: str) -> bool:
    before, after = _copies(test.before, output), _copies(test.after, output)
    return bool(before) and bool(after) and max(before) < min(after)


class OrderTests(SubMetric):
    name = "order_tests"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return bool(order_tests(truth))

    def score(self, ctx: DocContext) -> float:
        tests = order_tests(ctx.truth)
        output = _searchable(ctx.document_text)
        return sum(passes(test, output) for test in tests) / len(tests)


MAX_GLUED = 2
MAX_SIDE_WORDS = 12
MAX_GAP_WORDS = 12
GAP = "\N{HORIZONTAL ELLIPSIS}"
JOIN = "\N{PILCROW SIGN}"


@dataclass(frozen=True)
class _Unit:
    first: int
    last: int
    breaks: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Document:
    sentences: tuple[str, ...]
    run_of: tuple[int, ...]
    units: tuple[_Unit, ...]
    groundtruth: str


def _document(truth: Groundtruth) -> _Document:
    language = sentence_language(truth.document.language)
    hyphenated = hyphenated_breaks(truth)
    body = in_reading_order(truth, [e.id for e in truth.elements if e.channel is Channel.BODY])
    position = {element_id: k for k, element_id in enumerate(body)}
    found_all: list[str] = []
    printed_all: list[str] = []
    run_of: list[int] = []
    units: list[_Unit] = []
    previous: Element | None = None
    run = -1
    for chain in chains(truth, _is_unit):
        found, breaks = _sentences(chain, hyphenated, language)
        if not found:
            continue
        head = chain[0]
        touches = (
            previous is not None
            and previous.page == head.page
            and position[head.id] == position[previous.id] + 1
        )
        run = run if touches else run + 1
        previous = chain[-1]
        base = len(found_all)
        found_all.extend(s.text for s in found)
        printed_all.extend(s.printed for s in found)
        local = [0] * len(found)
        unit_breaks: list[tuple[int, int]] = []
        for word, split_word, new_page in breaks:
            limit = word - 1 if split_word else word
            before = [k for k, s in enumerate(found) if s.end <= limit]
            after = [k for k, s in enumerate(found) if s.start >= word]
            if before and after:
                unit_breaks.append((base + before[-1], base + after[0]))
            if new_page:
                for k, s in enumerate(found):
                    local[k] += 2 if s.start >= word else 1 if s.end > limit else 0
        run_of.extend(run + extra for extra in local)
        run += max(local)
        units.append(_Unit(base, base + len(found) - 1, tuple(unit_breaks)))
    runs = [
        _concat([t for t, r in zip(printed_all, run_of, strict=True) if r == k])
        for k in range(run + 1)
    ]
    others = [_searchable(t) for t in all_texts(truth, keep=lambda e: not _is_unit(e))]
    groundtruth = " | ".join([*runs, *others])
    return _Document(tuple(found_all), tuple(run_of), tuple(units), groundtruth)


def _glued(doc: _Document, i: int, step: int) -> str | None:
    glued = [doc.sentences[i]]
    for k in range(1, MAX_GLUED + 2):
        text = _trim(_concat(glued, JOIN), keep_end=step < 0)
        if _unique(text, doc):
            return text
        j = i + step * k
        if k > MAX_GLUED or not 0 <= j < len(doc.sentences) or doc.run_of[j] != doc.run_of[i]:
            return None
        glued = [doc.sentences[j], *glued] if step < 0 else [*glued, doc.sentences[j]]
    return None


def _trim(text: str, *, keep_end: bool) -> str:
    found = text.split()[::-1] if keep_end else text.split()
    kept: list[str] = []
    for token in found:
        if sum(t != JOIN for t in kept) == MAX_SIDE_WORDS:
            break
        kept.append(token)
    kept = kept[::-1] if keep_end else kept
    while kept and kept[0] in (GAP, JOIN):
        kept.pop(0)
    while kept and kept[-1] in (GAP, JOIN):
        kept.pop()
    return f" {' '.join(kept)} " if kept else ""


def _concat(parts: list[str], mark: str = "") -> str:
    separator = f" {mark} " if mark else " "
    return " " + separator.join(p.strip() for p in parts if p.strip()) + " " if parts else ""


@dataclass(frozen=True)
class _Sentence:
    text: str
    start: int
    end: int
    printed: str


def _is_unit(e: Element) -> bool:
    return e.cls in UNIT_CLASSES and e.channel is Channel.BODY


def _sentences(
    chain: list[Element], hyphenated: set[str], language: str
) -> tuple[list[_Sentence], list[tuple[int, bool, bool]]]:
    pieces = [e for e in chain if e.text.strip()]
    if not pieces:
        return [], []
    text = pieces[0].text
    breaks: list[tuple[int, bool, bool]] = []
    for previous, piece in zip(pieces, pieces[1:], strict=False):
        joined = join_pieces(text, piece.text, hyphenated=previous.id in hyphenated)
        split_word = len(words(joined)) < len(words(text)) + len(words(piece.text))
        breaks.append((len(words(text)), split_word, piece.page != previous.page))
        text = joined
    found = [(s, len(words(s))) for s in sentences(text, language)]
    found = [(s, n) for s, n in found if n]
    ends = list(accumulate(n for _, n in found))
    printed = [_searchable(s) for s, _ in found]
    result = [
        _Sentence(p, end - n, end, p) for p, (_, n), end in zip(printed, found, ends, strict=True)
    ]
    page_breaks = [(word, split_word) for word, split_word, new_page in breaks if new_page]
    return [_with_gaps(s, page_breaks) for s in result], breaks


def _with_gaps(sentence: _Sentence, page_breaks: list[tuple[int, bool]]) -> _Sentence:
    tokens = sentence.text.split()
    for word, split_word in reversed(page_breaks):
        if not sentence.start < word < sentence.end:
            continue
        cut = word - sentence.start
        tokens[cut - 1 if split_word else cut : cut] = [GAP]
    return replace(sentence, text=f" {' '.join(tokens)} ")


def _searchable(text: str) -> str:
    found = words(text)
    return f" {' '.join(found).casefold()} " if found else ""


def _unique(sentence: str, doc: _Document) -> bool:
    return len(_copies(sentence, doc.groundtruth)) == 1


def _copies(sentence: str, text: str) -> list[int]:
    pieces = _pieces(sentence)
    if not pieces:
        return []
    (first, _), *rest = pieces
    starts: list[int] = []
    for match in _matches(first, text):
        found: tuple[int, int] | None = (match.end, _edits(first) - match.dist)
        for piece, mark in rest:
            found = _next_piece(piece, text, *found, mark) if found is not None else None
        if found is not None:
            starts.append(match.start)
    return starts


def _pieces(sentence: str) -> list[tuple[str, str | None]]:
    pieces: list[tuple[str, str | None]] = []
    found: list[str] = []
    mark: str | None = None
    for token in sentence.split():
        if token not in (GAP, JOIN):
            found.append(token)
        elif found:
            pieces.append((" ".join(found), mark))
            found, mark = [], token
        else:
            mark = GAP if GAP in (mark, token) else token
    if found:
        pieces.append((" ".join(found), mark))
    return pieces


def _matches(piece: str, text: str) -> list[Match]:
    return find_near_matches(piece, text, max_l_dist=_edits(piece))


def _edits(piece: str) -> int:
    return min(len(piece) // CHARACTERS_PER_EDIT, MAX_EDITS)


def _next_piece(
    piece: str, text: str, start: int, spare: int, mark: str | None
) -> tuple[int, int] | None:
    edits = _edits(piece)
    if mark == JOIN:
        window = text[start : start + spare + len(piece) + 2 * edits + 2]
        found = [m for m in _matches(piece, window) if m.start <= 1 + spare + edits]
    else:
        latest = start
        for _ in range(MAX_GAP_WORDS + 1):
            space = text.find(" ", latest + 1)
            if space == -1:
                latest = len(text)
                break
            latest = space
        window = text[start : latest + len(piece) + MAX_EDITS + 2]
        found = [m for m in _matches(piece, window) if start + m.start <= latest + 1]
    return (start + found[0].end, edits - found[0].dist) if found else None


class BlockOrderSimilarity(SubMetric):
    name = "block_order_similarity"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return len(locating_sentences(truth)) >= 2

    def score(self, ctx: DocContext) -> float:
        locating = locating_sentences(ctx.truth)
        found = output_order(locating, _searchable(ctx.document_text))
        expected = list(range(len(locating)))
        return 1.0 - Levenshtein.distance(expected, found) / len(expected)


TRACK = Track(
    id="T2",
    name="Reading order",
    parts=(
        (OrderTests(), 1.0),
        (BlockOrderSimilarity(), 1.0),
    ),
)
