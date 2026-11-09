from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import Match, SequenceMatcher
from enum import StrEnum

from rapidfuzz import fuzz

from wholedoc_bench.schema import Channel, Element, ElementClass, Groundtruth, RelationKind
from wholedoc_bench.scorer.utils.groundtruth import join_variants
from wholedoc_bench.scorer.utils.pairing import best_pairing
from wholedoc_bench.scorer.utils.prose import edit_similarity
from wholedoc_bench.scorer.utils.reading_order import in_reading_order

THRESHOLD = 0.9
BLOCK_CLASSES = frozenset({ElementClass.PARAGRAPH, ElementClass.LIST_ITEM})
MIN_EDGE_CHARACTERS = 3

_CLAIMED = "\x00"

Normalize = Callable[[str], str]
Similarity = Callable[[str, str], float]


def _unchanged(text: str) -> str:
    return text


@dataclass(frozen=True)
class TruthBlock:
    element_ids: tuple[str, ...]
    pages: tuple[int, ...]
    text: str
    variants: tuple[str, ...]


@dataclass(frozen=True)
class OutputBlock:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Junction:
    kind: RelationKind
    before: str
    after: str
    same_page: bool


class JunctionOutcome(StrEnum):
    JOINED = "joined"
    MERGED = "merged"
    SPLIT = "split"
    KEPT_APART = "kept_apart"
    MISSING = "missing"


@dataclass(frozen=True)
class JunctionResult:
    junction: Junction
    outcome: JunctionOutcome


@dataclass(frozen=True)
class BlockIntegrityResult:
    f1: float
    precision: float
    recall: float
    truth_blocks: tuple[TruthBlock, ...]
    matches: tuple[tuple[int, int], ...]
    merged_blocks: tuple[int, ...]
    counted_blocks: tuple[int, ...]
    junctions: tuple[JunctionResult, ...]


def is_block_element(e: Element) -> bool:
    return e.cls in BLOCK_CLASSES and e.channel is Channel.BODY and bool(e.text.strip())


def has_blocks(truth: Groundtruth, normalize: Normalize = _unchanged) -> bool:
    return bool(truth_blocks(truth, normalize))


def truth_blocks(truth: Groundtruth, normalize: Normalize = _unchanged) -> list[TruthBlock]:
    pieces = {e.id: e for e in truth.elements if is_block_element(e)}
    next_of: dict[str, str] = {}
    hyphen_after: set[str] = set()
    for r in truth.relations:
        if r.kind is RelationKind.CONTINUES_IN and r.source in pieces and r.target in pieces:
            next_of[r.source] = r.target
            if r.hyphenated:
                hyphen_after.add(r.source)
    continued = set(next_of.values())
    blocks: list[TruthBlock] = []
    for start in in_reading_order(truth, pieces):
        if start in continued:
            continue
        ids = [start]
        while ids[-1] in next_of:
            ids.append(next_of[ids[-1]])
        joined = join_variants([pieces[i].text for i in ids], [i in hyphen_after for i in ids[:-1]])
        variants = tuple(dict.fromkeys(normalize(text) for text in joined))
        if not variants[0].strip():
            continue
        pages = tuple(pieces[i].page for i in ids)
        blocks.append(TruthBlock(tuple(ids), pages, variants[0], variants))
    return blocks


def junctions(truth: Groundtruth) -> list[Junction]:
    pieces = {e.id: e for e in truth.elements if is_block_element(e)}
    found: list[Junction] = []
    for r in truth.relations:
        if r.kind not in (RelationKind.CONTINUES_IN, RelationKind.STAYS_APART):
            continue
        if r.source in pieces and r.target in pieces:
            same_page = pieces[r.source].page == pieces[r.target].page
            found.append(Junction(r.kind, r.source, r.target, same_page))
    return found


def score_block_integrity(
    truth: Groundtruth,
    blocks: Sequence[OutputBlock],
    *,
    normalize: Normalize = _unchanged,
    similarity: Similarity = edit_similarity,
    threshold: float = THRESHOLD,
) -> BlockIntegrityResult:
    targets = truth_blocks(truth, normalize)
    if not targets:
        raise ValueError(f"{truth.document.id}: no body paragraph or list item to score")
    texts = [normalize(b.text) for b in blocks]
    candidates = [j for j, text in enumerate(texts) if text.strip()]

    owners = [i for i, t in enumerate(targets) for _ in t.variants]
    spellings = [text for t in targets for text in t.variants]
    occurrences = {
        j: {
            owners[k]
            for k in _occurrences(
                spellings,
                texts[j],
                threshold,
            )
        }
        for j in candidates
    }
    merged = [j for j in candidates if len(occurrences[j]) >= 2]
    merged_set = set(merged)

    free = [j for j in candidates if j not in merged_set]
    table = [
        [max(similarity(text, texts[j]) for text in target.variants) for j in free]
        for target in targets
    ]
    rejected: dict[int, set[frozenset[int]]] = {}
    while True:
        pairs = _best_pairing(table, threshold)
        dropped = False
        for i, c in pairs:
            if glue := _glued(targets, i, texts[free[c]], similarity, threshold):
                rejected.setdefault(c, set()).update(glue)
                table[i][c] = 0.0
                dropped = True
        if not dropped:
            break
    paired = {c for _, c in pairs}
    edges = {c: glue for c, glue in rejected.items() if c not in paired}
    merged = sorted(merged + [free[c] for c in edges])
    edge_blocks = {free[c] for c in edges}
    glued = {pair for glue in edges.values() for pair in glue}
    matches = sorted((i, free[c]) for i, c in pairs)

    matched_blocks = {j for _, j in matches}
    unmatched = [j for j in candidates if j not in matched_blocks]
    unscored = _unscored_texts(truth, normalize)
    copies = _best_pairing(
        [[similarity(u, texts[j]) for j in unmatched] for u in unscored], threshold
    )
    excused = {unmatched[c] for _, c in copies}
    counted = [j for j in candidates if j not in excused]

    recall = len(matches) / len(targets)
    precision = len(matches) / len(counted) if counted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    outcomes = _junction_outcomes(
        truth, targets, blocks, texts, candidates, matches, edge_blocks, glued, normalize, threshold
    )
    return BlockIntegrityResult(
        f1=f1,
        precision=precision,
        recall=recall,
        truth_blocks=tuple(targets),
        matches=tuple(matches),
        merged_blocks=tuple(merged),
        counted_blocks=tuple(counted),
        junctions=tuple(outcomes),
    )


def _junction_outcomes(
    truth: Groundtruth,
    targets: Sequence[TruthBlock],
    blocks: Sequence[OutputBlock],
    texts: Sequence[str],
    candidates: Sequence[int],
    matches: Sequence[tuple[int, int]],
    edge_blocks: set[int],
    glued: set[frozenset[int]],
    normalize: Normalize,
    threshold: float,
) -> list[JunctionResult]:
    ids = [piece for target in targets for piece in target.element_ids]
    continues = {piece for target in targets for piece in target.element_ids[:-1]}
    pieces = {e.id: e for e in truth.elements if e.id in set(ids)}
    piece_texts = [normalize(_searched_text(pieces[i].text, i in continues)) for i in ids]
    judged = {i for i, text in zip(ids, piece_texts, strict=True) if text.strip()}
    links = [link for link in junctions(truth) if link.before in judged and link.after in judged]
    if not links:
        return []
    holders: dict[str, set[int]] = {i: set() for i in ids}
    held: dict[int, set[str]] = {}
    for j in candidates:
        held[j] = {ids[k] for k in _occurrences(piece_texts, texts[j], threshold)}
        for piece in held[j]:
            holders[piece].add(j)
    block_of = {piece: i for i, t in enumerate(targets) for piece in t.element_ids}
    matched_truth = {i for i, _ in matches}
    results: list[JunctionResult] = []
    for link in links:
        before, after = holders[link.before], holders[link.after]
        shared = before & after
        if link.kind is RelationKind.CONTINUES_IN:
            own = block_of[link.before]
            if any(
                j not in edge_blocks and all(block_of[p] == own for p in held[j]) for j in shared
            ):
                outcome = JunctionOutcome.JOINED
            elif shared:
                outcome = JunctionOutcome.MERGED
            elif before and after:
                outcome = JunctionOutcome.SPLIT
            else:
                outcome = JunctionOutcome.MISSING
        elif shared or frozenset((block_of[link.before], block_of[link.after])) in glued:
            outcome = JunctionOutcome.JOINED
        elif block_of[link.before] in matched_truth and block_of[link.after] in matched_truth:
            outcome = JunctionOutcome.KEPT_APART
        else:
            outcome = JunctionOutcome.MISSING
        results.append(JunctionResult(link, outcome))
    return results


def _searched_text(text: str, continues: bool) -> str:
    end = text.rstrip()
    return end.removesuffix("-") if continues and end.endswith("-") else text


def _best_pairing(similarities: list[list[float]], threshold: float) -> list[tuple[int, int]]:
    return best_pairing([[s if s >= threshold else None for s in row] for row in similarities])


def _occurrences(needles: Sequence[str], haystack: str, threshold: float) -> list[int]:
    order = sorted(
        (k for k, needle in enumerate(needles) if needle),
        key=lambda k: (-len(needles[k]), k),
    )
    remaining = haystack
    found: list[int] = []
    for k in order:
        stretch = _aligned_stretch(needles[k], remaining, threshold)
        if stretch is None:
            continue
        start, end = stretch
        remaining = remaining[:start] + _CLAIMED * (end - start) + remaining[end:]
        found.append(k)
    return found


def _aligned_stretch(needle: str, haystack: str, threshold: float) -> tuple[int, int] | None:
    if len(needle) > len(haystack):
        reachable = len(haystack) >= threshold * len(needle)
        if reachable and edit_similarity(needle, haystack) >= threshold:
            return 0, len(haystack)
        return None
    alignment = fuzz.partial_ratio_alignment(needle, haystack, score_cutoff=threshold * 100)
    return None if alignment is None else (alignment.dest_start, alignment.dest_end)


def _glued(
    targets: Sequence[TruthBlock], i: int, text: str, similarity: Similarity, threshold: float
) -> set[frozenset[int]]:
    spelling = max(targets[i].variants, key=lambda variant: similarity(variant, text))
    head, tail = _leftovers(spelling, text)
    glue: set[frozenset[int]] = set()
    if i + 1 < len(targets) and _is_edge(tail, targets[i + 1].variants, similarity, threshold):
        glue.add(frozenset((i, i + 1)))
    if i > 0 and _is_edge(head, targets[i - 1].variants, similarity, threshold, end=True):
        glue.add(frozenset((i - 1, i)))
    return glue


def _leftovers(paragraph: str, block: str) -> tuple[str, str]:
    runs = _without_stray_ends(
        SequenceMatcher(None, paragraph, block, autojunk=False).get_matching_blocks()[:-1]
    )
    if not runs:
        return "", ""
    first, last = runs[0], runs[-1]
    return block[: first.b].strip(), block[last.b + last.size :].strip()


def _without_stray_ends(runs: list[Match]) -> list[Match]:
    while len(runs) > 1 and _is_stray(runs[-1], runs[-1].b - (runs[-2].b + runs[-2].size)):
        runs = runs[:-1]
    while len(runs) > 1 and _is_stray(runs[0], runs[1].b - (runs[0].b + runs[0].size)):
        runs = runs[1:]
    return runs


def _is_stray(run: Match, gap: int) -> bool:
    return run.size < MIN_EDGE_CHARACTERS and run.size <= gap


def _is_edge(
    leftover: str,
    neighbour: Sequence[str],
    similarity: Similarity,
    threshold: float,
    *,
    end: bool = False,
) -> bool:
    if sum(ch.isalnum() for ch in leftover) < MIN_EDGE_CHARACTERS:
        return False
    size = len(leftover)
    edges = (spelling[-size:] if end else spelling[:size] for spelling in neighbour)
    return any(similarity(leftover, edge) >= threshold for edge in edges)


def _unscored_texts(truth: Groundtruth, normalize: Normalize) -> list[str]:
    visible = (Channel.BODY, Channel.FURNITURE)
    return [
        normalize(e.text)
        for e in truth.elements
        if e.text.strip() and e.channel in visible and not is_block_element(e)
    ]
