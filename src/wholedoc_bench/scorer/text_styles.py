from dataclasses import dataclass
from typing import Any

from wholedoc_bench.schema import Element, ElementClass, Groundtruth, StyleName, is_style_element
from wholedoc_bench.scorer.utils.blocks import Block, BlockKind
from wholedoc_bench.scorer.utils.groundtruth import (
    chains,
    hyphenated_breaks,
    join_boundary,
    join_choices,
)
from wholedoc_bench.scorer.utils.text_mapping import (
    MappedText,
    concatenate,
    mapped_text,
    normalize_mapped,
)
from wholedoc_bench.scorer.utils.text_matching import align, allocate


@dataclass(frozen=True)
class StyleResult:
    score: float | None
    details: dict[str, Any]


def _element_text(element: Element) -> MappedText:
    styles = [set() for _ in element.text]
    for span in element.styles or []:
        for index in range(span.start, span.end):
            styles[index].add(span.style)
    return mapped_text(element.text, tuple(frozenset(s) for s in styles), element.id)


def _whole_bold(text: str, styles: tuple[frozenset[StyleName], ...]) -> bool:
    marks = [s for char, s in zip(text, styles, strict=True) if char.isalnum()]
    return bool(marks) and all(StyleName.BOLD in s for s in marks)


def _without_heading_bold(value: MappedText) -> MappedText:
    if not _whole_bold(value.text, value.styles):
        return value
    return MappedText(value.text, tuple(s - {StyleName.BOLD} for s in value.styles), value.origins)


def _slice(value: MappedText, start: int, end: int) -> MappedText:
    return MappedText(value.text[start:end], value.styles[start:end], value.origins[start:end])


def style_variants(truth: Groundtruth) -> list[tuple[tuple[str, ...], tuple[MappedText, ...]]]:
    hyphenated = hyphenated_breaks(truth)
    result = []
    for chain in chains(truth, is_style_element):
        pieces = [_element_text(element) for element in chain]
        edges = [e.id in hyphenated for e in chain[:-1]]
        choices = join_choices([piece.text for piece in pieces], edges)
        variants: dict[str, MappedText] = {}
        owners: dict[str, tuple[bool, ...]] = {}
        for joined in choices:
            value = pieces[0]
            for i, second in enumerate(pieces[1:]):
                boundary = join_boundary(
                    value.text, second.text, hyphenated=edges[i] or i in joined
                )
                value = concatenate(
                    [
                        _slice(value, 0, boundary.first_end),
                        _slice(second, boundary.second_start, len(second.text)),
                    ],
                    boundary.separator,
                )
            if all(e.cls is ElementClass.HEADING for e in chain):
                value = _without_heading_bold(value)
            owner = tuple(i in joined for i in range(len(edges)))
            if value.text not in owners or owner > owners[value.text]:
                variants[value.text] = normalize_mapped(value)
                owners[value.text] = owner
        result.append((tuple(e.id for e in chain), tuple(variants.values())))
    return result


def style_output(blocks: list[Block]) -> MappedText:
    return normalize_mapped(
        concatenate(
            [
                mapped_text(block.text, block.styles, f"block:{i}")
                for i, block in enumerate(blocks)
                if block.kind is not BlockKind.TABLE
            ]
        )
    )


def _heading_bold(blocks: list[Block], output: MappedText) -> dict[int, str]:
    owners = {
        f"block:{i}"
        for i, block in enumerate(blocks)
        if block.kind is BlockKind.HEADING
        and block.styles
        and _whole_bold(block.text, block.styles)
    }
    return {
        index: origins[0].owner
        for index, (char, styles, origins) in enumerate(
            zip(output.text, output.styles, output.origins, strict=True)
        )
        if not char.isspace()
        and StyleName.BOLD in styles
        and origins
        and origins[0].owner in owners
    }


def _counts(value: MappedText, unknown: frozenset[str] = frozenset()) -> dict[StyleName, int]:
    counts = dict.fromkeys(StyleName, 0)
    for index, (char, styles) in enumerate(zip(value.text, value.styles, strict=True)):
        if not char.isspace() and not _hidden(value, index, unknown):
            for style in styles:
                counts[style] += 1
    return counts


def unknown_styles(truth: Groundtruth) -> frozenset[str]:
    return frozenset(e.id for e in truth.elements if is_style_element(e) and e.styles is None)


def _hidden(value: MappedText, index: int, unknown: frozenset[str]) -> bool:
    return bool(unknown) and any(origin.owner in unknown for origin in value.origins[index])


def _origin_ranges(value: MappedText, indices: list[int]) -> list[dict[str, Any]]:
    positions: dict[str, set[int]] = {}
    for index in indices:
        for origin in value.origins[index]:
            positions.setdefault(origin.owner, set()).add(origin.offset)
    result: list[dict[str, Any]] = []
    for owner, offsets in positions.items():
        for offset in sorted(offsets):
            if result and result[-1]["owner"] == owner and result[-1]["end"] == offset:
                result[-1]["end"] = offset + 1
            else:
                result.append({"owner": owner, "start": offset, "end": offset + 1})
    return result


def style_status(truth: Groundtruth, status: str) -> StyleResult:
    from wholedoc_bench.corpus.checks import check_style_annotations

    if not check_style_annotations(truth):
        return StyleResult(None, {"status": "not_applicable", "reason": "no complete body styles"})
    expected = dict.fromkeys(StyleName, 0)
    unknown = unknown_styles(truth)
    for _, variants in style_variants(truth):
        for style, count in _counts(variants[0], unknown).items():
            expected[style] += count
    return StyleResult(
        0.0,
        {
            "status": status,
            "styles": {
                style.value: {"expected": count, "found": 0, "matched": 0, "f1": 0.0}
                for style, count in expected.items()
                if count
            },
            "unmatched_source_ids": [
                e.id for e in truth.elements if is_style_element(e) and e.styles is not None
            ],
            "unmatched_output_ranges": [],
        },
    )


def score_text_styles(
    truth: Groundtruth, blocks: list[Block], *, reference: bool = False
) -> StyleResult:
    from wholedoc_bench.corpus.checks import check_style_annotations

    if not check_style_annotations(truth):
        return style_status(truth, "not_applicable")
    targets = style_variants(truth)
    unknown = unknown_styles(truth)
    output = style_output(blocks)
    heading_bold = _heading_bold(blocks, output)
    assigned = allocate(
        [[v.text for v in variants] for _, variants in targets], output.text, reference=reference
    )
    aligned: dict[int, tuple[MappedText, list[tuple[int, int]]]] = {}
    for index, (_, variants) in enumerate(targets):
        if (candidate := assigned.get(index)) is not None:
            value = variants[candidate.variant]
            pairs = align(
                value.text, output.text[candidate.start : candidate.end], candidate.distance
            )
            aligned[index] = (value, [(s, o + candidate.start) for s, o in pairs])
    unearned = {
        heading_bold[o]
        for value, pairs in aligned.values()
        for s, o in pairs
        if o in heading_bold
        and value.text[s].isalnum()
        and StyleName.BOLD not in value.styles[s]
        and not _hidden(value, s, unknown)
    }
    ungraded = {
        o
        for value, pairs in aligned.values()
        for s, o in pairs
        if _hidden(value, s, unknown) and not output.text[o].isspace()
    }
    expected = dict.fromkeys(StyleName, 0)
    matched = dict.fromkeys(StyleName, 0)
    credited: set[tuple[int, StyleName]] = set()
    unmatched_source: set[str] = set()
    assignments = []
    for index, (ids, variants) in enumerate(targets):
        candidate = assigned.get(index)
        value = variants[candidate.variant if candidate is not None else 0]
        for style, count in _counts(value, unknown).items():
            expected[style] += count
        if candidate is None:
            unmatched_source.update(i for i in ids if i not in unknown)
            continue
        value, pairs = aligned[index]
        source_credit: set[tuple[int, StyleName]] = set()
        for source_index, output_index in pairs:
            if value.text[source_index].isspace() or output.text[output_index].isspace():
                continue
            if _hidden(value, source_index, unknown):
                continue
            common = value.styles[source_index] & output.styles[output_index]
            if heading_bold.get(output_index) in unearned:
                common -= {StyleName.BOLD}
            for style in common:
                matched[style] += 1
                credited.add((output_index, style))
                source_credit.add((source_index, style))
        for i, memberships in enumerate(value.styles):
            if not value.text[i].isspace() and any(
                (i, s) not in source_credit for s in memberships
            ):
                unmatched_source.update(
                    origin.owner for origin in value.origins[i] if origin.owner not in unknown
                )
        assignments.append(
            {
                "source_ids": list(ids),
                "comparison_range": [candidate.start, candidate.end],
                "output_origins": _origin_ranges(
                    output, list(range(candidate.start, candidate.end))
                ),
                "variant": candidate.variant,
                "similarity": candidate.similarity,
            }
        )
    free = {i for i in heading_bold if (i, StyleName.BOLD) not in credited and i not in ungraded}
    found = _counts(output)
    found[StyleName.BOLD] -= len(free)
    for index in ungraded:
        for style in output.styles[index]:
            found[style] -= 1
    styles = {
        style.value: {
            "expected": expected[style],
            "found": found[style],
            "matched": matched[style],
            "f1": 2 * matched[style] / (expected[style] + found[style]),
        }
        for style in StyleName
        if expected[style] + found[style]
    }
    ranges = []
    for style in StyleName:
        indices = [
            i
            for i, memberships in enumerate(output.styles)
            if style in memberships
            and not output.text[i].isspace()
            and (i, style) not in credited
            and not (style is StyleName.BOLD and i in free)
            and i not in ungraded
        ]
        for index in indices:
            if ranges and ranges[-1]["style"] == style.value and ranges[-1]["end"] == index:
                ranges[-1]["end"] = index + 1
            else:
                ranges.append({"style": style.value, "start": index, "end": index + 1})
    for row in ranges:
        row["output_origins"] = _origin_ranges(output, list(range(row["start"], row["end"])))
    score = sum(value["f1"] for value in styles.values()) / len(styles) if styles else 1.0
    return StyleResult(
        score,
        {
            "status": "scored",
            "styles": styles,
            "unmatched_source_ids": sorted(unmatched_source),
            "unmatched_output_ranges": ranges,
            "assignments": assignments,
        },
    )
