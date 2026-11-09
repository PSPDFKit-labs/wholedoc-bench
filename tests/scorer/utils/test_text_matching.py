import random
from fractions import Fraction

import pytest
from rapidfuzz.distance import Levenshtein

from wholedoc_bench.scorer.utils.text_matching import Match, align, exact_candidate, find_candidate


@pytest.mark.parametrize("reference", [False, True])
def test_later_length_can_reach_word_boundary(reference):
    actual = find_candidate(["abcdefghij"], "abcdefghiX", reference=reference)
    assert actual == Match(0, 10, 0, 1, 10)


@pytest.mark.parametrize("finder", [exact_candidate, find_candidate])
def test_overlapping_claims_end_before_the_free_occurrence(finder):
    text = "alpha beta alpha beta alpha beta"
    actual = finder(["alpha beta"], text, [(0, 8), (3, 11), (11, 21), (0, 8)])
    assert actual == Match(22, 32, 0, 0, 10)
    assert finder(["alpha beta"], text, [(0, len(text))]) is None


def test_claimed_character_blocks_every_longer_window_at_that_start():
    text = "abcdefghiX abcdefghiY"
    assert find_candidate(["abcdefghij"], text, [(9, 10)]) == Match(11, 21, 0, 1, 10)


def _complete_oracle(variants, text, claimed, boundary_start, boundary_end):
    matches = []
    for variant, needle in enumerate(variants):
        if not needle:
            continue
        for start in range(len(text)):
            if boundary_start and needle[0].isalnum() and start and text[start - 1].isalnum():
                continue
            for end in range(start + 1, len(text) + 1):
                if any(start < b and a < end for a, b in claimed):
                    continue
                if (
                    boundary_end
                    and needle[-1].isalnum()
                    and end < len(text)
                    and text[end].isalnum()
                ):
                    continue
                distance = Levenshtein.distance(needle, text[start:end])
                denominator = max(len(needle), end - start)
                if 10 * distance <= denominator:
                    matches.append((Fraction(distance, denominator), distance, start, end, variant))
    if not matches:
        return None
    ratio, distance, start, end, variant = min(matches)
    return Match(start, end, variant, distance, max(len(variants[variant]), end - start))


def test_claimed_search_matches_independent_complete_windows():
    rng = random.Random(2609)
    for _ in range(120):
        source = " ".join(rng.choices(["one", "two", "ten", "cat"], k=4))
        changed = source[:5] + "x" + source[6:]
        text = f"{changed} {source} {changed}"
        claims = []
        for _ in range(rng.randrange(6)):
            start = rng.randrange(len(text))
            claims.append((start, min(len(text), start + rng.randrange(1, 15))))
        variants = [source, source.replace("o", "u", 1)]
        boundary_start, boundary_end = rng.choice([True, False]), rng.choice([True, False])
        expected = _complete_oracle(variants, text, claims, boundary_start, boundary_end)
        for reference in (False, True):
            actual = find_candidate(
                variants,
                text,
                claims,
                reference=reference,
                boundary_start=boundary_start,
                boundary_end=boundary_end,
            )
            assert actual == expected


def test_one_edit_patterns_match_the_complete_oracle():
    source = "abcdefghi jklmn"
    outputs = {source}
    for i in range(len(source)):
        outputs.add(source[:i] + source[i + 1 :])
        outputs.add(source[:i] + "x" + source[i + 1 :])
    for i in range(len(source) + 1):
        outputs.add(source[:i] + "x" + source[i:])
    for output in outputs:
        text = "junk " + output + " tail " + source
        for claimed in [(), ((len(text) - len(source), len(text)),)]:
            expected = _complete_oracle([source], text, list(claimed), True, True)
            for reference in (False, True):
                assert find_candidate([source], text, claimed, reference=reference) == expected


def test_alignment_ties_trace_back_from_the_ends():
    assert align("aa", "a", 1) == [(1, 0)]
    assert align("ab", "ba", 2) == [(0, 0), (1, 1)]
