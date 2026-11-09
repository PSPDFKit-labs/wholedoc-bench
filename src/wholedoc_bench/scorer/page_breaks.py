import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from wholedoc_bench.schema import Channel, Groundtruth, Relation, RelationKind
from wholedoc_bench.scorer.block_integrity import (
    THRESHOLD,
    BlockIntegrityResult,
    OutputBlock,
    TruthBlock,
    _best_pairing,
    _glued,
    _searched_text,
    _unscored_texts,
    is_block_element,
)
from wholedoc_bench.scorer.utils.groundtruth import join_variants
from wholedoc_bench.scorer.utils.prose import edit_similarity, normalize
from wholedoc_bench.scorer.utils.text_matching import Match, align, exact_candidate, find_candidate


@dataclass(frozen=True)
class PageBreakResult:
    score: float | None
    details: dict[str, Any]


@dataclass(frozen=True)
class _Unit:
    owner: int
    ids: tuple[str, ...]
    variants: tuple[str, ...]


@dataclass(frozen=True)
class _Occurrence:
    block: int
    start: int
    end: int
    ids: tuple[str, ...]
    similarity: float
    role: str = "piece"

    def evidence(self, role: str | None = None) -> dict[str, Any]:
        return {
            "block": self.block,
            "start": self.start,
            "end": self.end,
            "source_ids": list(self.ids),
            "similarity": self.similarity,
            "role": role or self.role,
        }


def _eligible_links(truth: Groundtruth) -> list[Relation]:
    pieces = {e.id: e for e in truth.elements if is_block_element(e)}
    continues = {r.source for r in truth.relations if r.kind is RelationKind.CONTINUES_IN}
    usable = {
        id_ for id_, e in pieces.items() if normalize(_searched_text(e.text, id_ in continues))
    }
    seen: set[tuple[RelationKind, str, str]] = set()
    links: list[Relation] = []
    for link in truth.relations:
        key = (link.kind, link.source, link.target)
        if (
            link.kind in (RelationKind.CONTINUES_IN, RelationKind.STAYS_APART)
            and link.source in usable
            and link.target in usable
            and pieces[link.source].page != pieces[link.target].page
            and key not in seen
        ):
            links.append(link)
            seen.add(key)
    return links


def has_page_breaks(truth: Groundtruth) -> bool:
    return bool(_eligible_links(truth))


def _result(links: list[dict[str, Any]], status: str) -> PageBreakResult:
    details: dict[str, Any] = {"status": status, "links": links}
    rates: list[float] = []
    for kind in (RelationKind.CONTINUES_IN, RelationKind.STAYS_APART):
        selected = [link for link in links if link["kind"] == kind]
        passed = sum(link["passed"] for link in selected)
        total = len(selected)
        rate = passed / total if total else None
        details[kind.value] = {"passed": passed, "total": total, "rate": rate}
        if rate is not None:
            rates.append(rate)
    if not rates:
        details["reason"] = "no eligible cross-page body links"
    return PageBreakResult(sum(rates) / len(rates) if rates else None, details)


def failed_page_breaks(truth: Groundtruth) -> PageBreakResult:
    links = [
        {
            "kind": link.kind.value,
            "from": link.source,
            "to": link.target,
            "passed": False,
            "reasons": ["parse_failed"],
            "evidence": [],
        }
        for link in _eligible_links(truth)
    ]
    return _result(links, "parse_failed" if links else "not_applicable")


def _best(
    variants: Sequence[str],
    texts: Sequence[str],
    claimed: Sequence[list[tuple[int, int]]],
    *,
    exact: bool = False,
) -> tuple[int, Match] | None:
    starts: list[int] = []
    block_ids: list[int] = []
    chunks: list[str] = []
    occupied: list[tuple[int, int]] = []
    offset = 0
    for block, value in enumerate(texts):
        if not value or any(a <= 0 and b >= len(value) for a, b in claimed[block]):
            continue
        starts.append(offset)
        block_ids.append(block)
        chunks.append(value + "\x00")
        occupied.extend((offset + a, offset + b) for a, b in claimed[block])
        offset += len(value)
        occupied.append((offset, offset + 1))
        offset += 1
    if not chunks:
        return None
    search = exact_candidate if exact else find_candidate
    found = search(variants, "".join(chunks), occupied)
    if found is None:
        return None
    index = bisect_right(starts, found.start) - 1
    return block_ids[index], Match(
        found.start - starts[index],
        found.end - starts[index],
        found.variant,
        found.distance,
        found.length,
    )


def _recover(
    unit: _Unit,
    block: int,
    match: Match,
    text: str,
    piece_texts: dict[str, str],
) -> dict[str, _Occurrence]:
    variant = unit.variants[match.variant]
    output = text[match.start : match.end]
    pairs = dict(align(variant, output, match.distance))
    recovered: dict[str, _Occurrence] = {}
    cursor = 0
    for id_ in unit.ids:
        needle = piece_texts[id_]
        start = variant.find(needle, cursor)
        if start < 0:
            continue
        end = start + len(needle)
        cursor = end
        positions = [pairs[i] for i in range(start, end) if i in pairs]
        if not positions:
            continue
        out_start, out_end = min(positions), max(positions) + 1
        quality = edit_similarity(needle, output[out_start:out_end])
        if quality >= THRESHOLD:
            recovered[id_] = _Occurrence(
                block, match.start + out_start, match.start + out_end, (id_,), quality
            )
    return recovered


def _units(
    truth: Groundtruth, targets: Sequence[TruthBlock]
) -> tuple[list[_Unit], list[_Unit], list[_Unit], dict[str, str]]:
    by_id = {e.id: e for e in truth.elements}
    continuing = {piece for target in targets for piece in target.element_ids[:-1]}
    piece_texts = {
        id_: normalize(_searched_text(by_id[id_].text, id_ in continuing))
        for target in targets
        for id_ in target.element_ids
    }
    hyphenated = {
        r.source for r in truth.relations if r.kind is RelationKind.CONTINUES_IN and r.hyphenated
    }
    full = [_Unit(i, target.element_ids, target.variants) for i, target in enumerate(targets)]
    segments: list[_Unit] = []
    singles: list[_Unit] = []
    for owner, target in enumerate(targets):
        ids = target.element_ids
        for start in range(len(ids)):
            for end in range(start + 2, len(ids) + 1):
                selected = ids[start:end]
                if selected == ids:
                    continue
                variants = join_variants(
                    [by_id[id_].text for id_ in selected],
                    [id_ in hyphenated for id_ in selected[:-1]],
                )
                segments.append(
                    _Unit(
                        owner,
                        selected,
                        tuple(
                            normalize(_searched_text(v, selected[-1] in continuing))
                            for v in variants
                        ),
                    )
                )
        singles.extend(_Unit(owner, (id_,), (piece_texts[id_],)) for id_ in ids if piece_texts[id_])
    return full, segments, singles, piece_texts


def _allocate(
    units: Sequence[_Unit],
    texts: Sequence[str],
    claimed: list[list[tuple[int, int]]],
    recovered: dict[str, _Occurrence],
    piece_texts: dict[str, str],
) -> None:
    order = sorted(enumerate(units), key=lambda item: (-max(map(len, item[1].variants)), item[0]))
    assigned: set[int] = set()
    for exact in (True, False):
        for index, unit in order:
            if index in assigned or any(id_ in recovered for id_ in unit.ids):
                continue
            candidate = (
                next(
                    (
                        (block, Match(0, len(text), variant, 0, len(text)))
                        for block, text in enumerate(texts)
                        if text and not claimed[block]
                        for variant, value in enumerate(unit.variants)
                        if text == value
                    ),
                    None,
                )
                if exact
                else None
            )
            if candidate is None:
                candidate = _best(unit.variants, texts, claimed, exact=exact)
            if candidate is None:
                continue
            block, match = candidate
            found = _recover(unit, block, match, texts[block], piece_texts)
            claimed[block].append((match.start, match.end))
            recovered.update(found)
            assigned.add(index)


def _reserve_unscored(
    truth: Groundtruth, texts: Sequence[str], claimed: list[list[tuple[int, int]]]
) -> None:
    candidates = [block for block, text in enumerate(texts) if text and not claimed[block]]
    sources = _unscored_texts(truth, normalize)
    pairs = _best_pairing(
        [[edit_similarity(source, texts[block]) for block in candidates] for source in sources],
        THRESHOLD,
    )
    for _, candidate in pairs:
        block = candidates[candidate]
        claimed[block].append((0, len(texts[block])))


def _fragment_ranges(
    source: str,
    text: str,
    source_tokens: Sequence[re.Match[str]],
    output_tokens: Sequence[re.Match[str]],
    occurrence: _Occurrence,
) -> list[tuple[int, int, int, int]]:
    first = next(i for i, token in enumerate(output_tokens) if token.end() > occurrence.start)
    last = first
    while last + 1 < len(output_tokens) and output_tokens[last + 1].start() < occurrence.end:
        last += 1
    needle = text[output_tokens[first].start() : output_tokens[last].end()]
    source_starts = {token.start(): i for i, token in enumerate(source_tokens)}
    result = []
    found = source.find(needle)
    while found >= 0:
        left = source_starts.get(found)
        if left is not None:
            right = left + last - first
            if right < len(source_tokens) and source_tokens[right].end() == found + len(needle):
                out_left, out_right = first, last
                while (
                    left
                    and out_left
                    and source_tokens[left - 1].group() == output_tokens[out_left - 1].group()
                ):
                    left -= 1
                    out_left -= 1
                while (
                    right + 1 < len(source_tokens)
                    and out_right + 1 < len(output_tokens)
                    and source_tokens[right + 1].group() == output_tokens[out_right + 1].group()
                ):
                    right += 1
                    out_right += 1
                a, b = source_tokens[left].start(), source_tokens[right].end()
                start, end = output_tokens[out_left].start(), output_tokens[out_right].end()
                if (
                    source[a:b] == text[start:end]
                    and start <= occurrence.start
                    and end >= occurrence.end
                ):
                    result.append((a, b, start, end))
        found = source.find(needle, found + 1)
    return result


def _reserve_fragments(
    singles: Sequence[_Unit],
    full: Sequence[_Unit],
    texts: Sequence[str],
    claimed: list[list[tuple[int, int]]],
    recovered: dict[str, _Occurrence],
    extras: Sequence[_Occurrence],
) -> list[_Occurrence]:
    by_block: dict[int, list[int]] = {}
    for index, occurrence in enumerate(extras):
        if not claimed[occurrence.block]:
            by_block.setdefault(occurrence.block, []).append(index)
    output_tokens = {block: list(re.finditer(r"\S+", texts[block])) for block in by_block}
    output_used = {block: bytearray(len(texts[block])) for block in by_block}
    excused: set[int] = set()
    source_used: dict[int, bytearray] = {}
    candidates: set[tuple[int, int, int, int, int, int]] = set()
    for unit_index, unit in enumerate(singles):
        if any(id_ in recovered for id_ in unit.ids):
            continue
        source = unit.variants[0]
        source_tokens = list(re.finditer(r"\S+", source))
        source_used[unit_index] = bytearray(len(source))
        own_variants = set(full[unit.owner].variants)
        others = [
            variant
            for other in full
            if not own_variants.intersection(other.variants)
            for variant in other.variants
        ]
        candidates.update(
            {
                (unit_index, block, a, b, start, end)
                for block, indices in by_block.items()
                for index in indices
                for a, b, start, end in _fragment_ranges(
                    source, texts[block], source_tokens, output_tokens[block], extras[index]
                )
                if not any(_phrase_ranges(source[a:b], other) for other in others)
            }
        )
    for unit_index, block, a, b, start, end in sorted(
        candidates,
        key=lambda item: (
            -(item[3] - item[2]),
            item[0],
            texts[item[1]],
            item[4],
            item[5],
            item[2],
            item[1],
        ),
    ):
        covered = {
            index
            for index in by_block[block]
            if index not in excused and start <= extras[index].start and extras[index].end <= end
        }
        if not covered or any(source_used[unit_index][a:b]) or any(output_used[block][start:end]):
            continue
        source_used[unit_index][a:b] = b"\x01" * (b - a)
        output_used[block][start:end] = b"\x01" * (end - start)
        excused.update(covered)
    return [occurrence for index, occurrence in enumerate(extras) if index not in excused]


def _extras(
    units: Sequence[_Unit],
    texts: Sequence[str],
    claimed: list[list[tuple[int, int]]],
    full_ids: dict[int, tuple[str, ...]],
) -> list[_Occurrence]:
    order = sorted(enumerate(units), key=lambda item: (-max(map(len, item[1].variants)), item[0]))
    owners: dict[str, set[str]] = {}
    for unit in units:
        for variant in unit.variants:
            owners.setdefault(variant, set()).update(unit.ids)
    identical_ids = [
        tuple(sorted(set().union(*(owners[v] for v in unit.variants)))) for unit in units
    ]
    result: list[_Occurrence] = []
    for exact in (True, False):
        for index, unit in order:
            while (candidate := _best(unit.variants, texts, claimed, exact=exact)) is not None:
                block, match = candidate
                claimed[block].append((match.start, match.end))
                result.append(
                    _Occurrence(
                        block,
                        match.start,
                        match.end,
                        identical_ids[index],
                        match.similarity,
                        "whole"
                        if unit.ids == full_ids[unit.owner]
                        else "segment"
                        if len(unit.ids) > 1
                        else "piece",
                    )
                )
    return result


def _phrase_ranges(phrase: str, text: str) -> list[tuple[int, int]]:
    return [
        (m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
    ]


def _contamination(
    truth: Groundtruth, link: Relation, target: TruthBlock, text: str, block: int
) -> list[dict[str, Any]]:
    by_id = {e.id: e for e in truth.elements}
    pages = {by_id[link.source].page, by_id[link.target].page}
    spelling = max(target.variants, key=lambda variant: edit_similarity(variant, text))
    known: dict[str, list[str]] = {}
    for element in truth.elements:
        if element.page not in pages:
            continue
        if element.channel is Channel.FURNITURE or (
            element.channel is Channel.BODY
            and element.cls.value in ("caption", "footnote", "heading")
        ):
            phrase = normalize(element.text)
            if phrase:
                known.setdefault(phrase, []).append(element.id)
    evidence: list[dict[str, Any]] = []
    for phrase, ids in known.items():
        ranges = _phrase_ranges(phrase, text)
        if len(ranges) > len(_phrase_ranges(phrase, spelling)):
            evidence.extend(
                {
                    "block": block,
                    "start": start,
                    "end": end,
                    "source_ids": ids,
                    "role": "extra_source_text",
                }
                for start, end in ranges
            )
    return evidence


def score_page_breaks(
    truth: Groundtruth, blocks: Sequence[OutputBlock], block_result: BlockIntegrityResult
) -> PageBreakResult:
    links = _eligible_links(truth)
    if not links:
        return _result([], "not_applicable")
    texts = [normalize(block.text) for block in blocks]
    targets = block_result.truth_blocks
    full, segments, singles, piece_texts = _units(truth, targets)
    claimed: list[list[tuple[int, int]]] = [[] for _ in texts]
    recovered: dict[str, _Occurrence] = {}
    _allocate(full, texts, claimed, recovered, piece_texts)
    _allocate(segments, texts, claimed, recovered, piece_texts)
    _allocate(singles, texts, claimed, recovered, piece_texts)
    _reserve_unscored(truth, texts, claimed)
    all_units = full + segments + singles
    full_ids = {unit.owner: unit.ids for unit in full}
    extras = _extras(all_units, texts, [list(ranges) for ranges in claimed], full_ids)
    if extras:
        extras = _reserve_fragments(singles, full, texts, claimed, recovered, extras)
    owner = {id_: i for i, target in enumerate(targets) for id_ in target.element_ids}
    held: dict[int, set[int]] = {}
    for id_, occurrence in recovered.items():
        held.setdefault(occurrence.block, set()).add(owner[id_])
    matched = dict(block_result.matches)
    outcomes: list[dict[str, Any]] = []
    for link in links:
        before, after = recovered.get(link.source), recovered.get(link.target)
        reasons: list[str] = []
        evidence = [
            occurrence.evidence(role)
            for occurrence, role in ((before, "before"), (after, "after"))
            if occurrence is not None
        ]
        if before is None or after is None:
            reasons.append("missing")
        else:
            own = owner[link.source]
            same = before.block == after.block
            ordered = before.end <= after.start
            glue = _glued(targets, own, texts[before.block], edit_similarity, THRESHOLD)
            if link.kind is RelationKind.CONTINUES_IN:
                if not same:
                    reasons.append("split")
                elif not ordered:
                    reasons.append("misplaced_break")
                extra_others = [
                    occurrence
                    for occurrence in extras
                    if occurrence.block == before.block
                    and any(owner[id_] != own for id_ in occurrence.ids)
                ]
                if same and (held.get(before.block, set()) - {own} or extra_others):
                    reasons.append("merged")
                    evidence.extend(occurrence.evidence("extra") for occurrence in extra_others)
                if same and glue:
                    reasons.append("misplaced_break")
                if same:
                    contamination = _contamination(
                        truth, link, targets[own], texts[before.block], before.block
                    )
                    if contamination:
                        reasons.append("extra_source_text")
                        evidence.extend(contamination)
            else:
                other = owner[link.target]
                if same:
                    reasons.append("merged")
                if frozenset((own, other)) in glue:
                    reasons.append("misplaced_break")
                if (
                    own not in matched
                    or other not in matched
                    or matched.get(own) == matched.get(other)
                ):
                    if not reasons:
                        reasons.append("unmatched")
        affected = [
            occurrence
            for occurrence in extras
            if link.source in occurrence.ids or link.target in occurrence.ids
        ]
        if affected:
            reasons.append("duplicate")
            if any(occurrence.role != "whole" for occurrence in affected):
                reasons.append("conflicting_copy")
            evidence.extend(occurrence.evidence("extra") for occurrence in affected)
        outcomes.append(
            {
                "kind": link.kind.value,
                "from": link.source,
                "to": link.target,
                "passed": not reasons,
                "reasons": list(dict.fromkeys(reasons)),
                "evidence": evidence,
            }
        )
    return _result(outcomes, "scored")
