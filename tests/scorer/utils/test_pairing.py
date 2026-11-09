import random
from itertools import permutations, product

import pytest

from wholedoc_bench.scorer.utils.pairing import Quality, best_pairing, exceeds, reaches

BEST_PAIRING_CASES = {
    "more_pairs_beat_one_strong_pair": ([[0.95, 0.9], [0.9, None]], None, [(0, 1), (1, 0)]),
    "best_total_among_equal_counts": ([[0.55, 0.95], [0.95, 0.55]], None, [(0, 1), (1, 0)]),
    "not_allowed_skipped": ([[None, None], [None, 0.2]], None, [(1, 1)]),
    "only_not_allowed": ([[None]], None, []),
    "no_rows": ([], None, []),
    "no_columns": ([[]], None, []),
    "better_pair_without_priority": ([[0.1], [1.0]], None, [(1, 0)]),
    "priority_row_first": ([[0.1], [1.0]], [True, False], [(0, 0)]),
}


@pytest.mark.parametrize(
    ("quality", "priority", "expected"), BEST_PAIRING_CASES.values(), ids=BEST_PAIRING_CASES
)
def test_best_pairing(
    quality: list[list[Quality]], priority: list[bool] | None, expected: list[tuple[int, int]]
) -> None:
    assert best_pairing(quality, priority_rows=priority) == expected


def test_column_order_makes_ties_independent_of_the_input_order() -> None:
    def pick(names: list[str]) -> str:
        order = sorted(range(len(names)), key=names.__getitem__)
        [(_, column)] = best_pairing([[0.5] * len(names)], column_order=order)
        return names[column]

    assert pick(["left", "right"]) == pick(["right", "left"])
    assert pick(["b", "a", "c"]) == pick(["c", "b", "a"]) == pick(["a", "c", "b"])


@pytest.mark.parametrize(
    ("quality", "order", "message"),
    [
        ([[1.5]], None, "from 0 to 1"),
        ([[-0.1]], None, "from 0 to 1"),
        ([[0.5, 0.5]], [0, 0], "each of the 2 columns"),
        ([[0.5], [0.5, 1.0]], None, "same number of columns"),
        ([[0.5, 0.5], [0.5]], None, "same number of columns"),
        ([[], [0.5]], None, "same number of columns"),
    ],
)
def test_bad_input_is_rejected(
    quality: list[list[Quality]], order: list[int] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        best_pairing(quality, column_order=order)


def _brute_force(quality: list[list[Quality]]) -> tuple[int, float]:
    rows, columns = len(quality), len(quality[0])
    best = (0, 0.0)
    for chosen in permutations(range(columns), min(rows, columns)):
        values = [q for r, c in enumerate(chosen) if (q := quality[r][c]) is not None]
        best = max(best, (len(values), round(sum(values), 9)))
    if rows > columns:
        for chosen in permutations(range(rows), columns):
            values = [q for c, r in enumerate(chosen) if (q := quality[r][c]) is not None]
            best = max(best, (len(values), round(sum(values), 9)))
    return best


def test_matches_brute_force_on_random_matrices() -> None:
    rng = random.Random(158)
    for _ in range(300):
        rows, columns = rng.randint(1, 4), rng.randint(1, 4)
        quality: list[list[Quality]] = [
            [rng.choice([None, 0.0, 0.5, 0.55, 0.9, 1.0]) for _ in range(columns)]
            for _ in range(rows)
        ]
        pairs = best_pairing(quality)
        assert len({r for r, _ in pairs}) == len({c for _, c in pairs}) == len(pairs)
        got = (len(pairs), round(sum(quality[r][c] or 0.0 for r, c in pairs), 9))
        assert got == _brute_force(quality)


def test_reaches_allows_for_float_rounding_only() -> None:
    assert reaches(0.4999999999999999, 0.5)
    assert reaches(0.5, 0.5)
    assert not reaches(0.4999, 0.5)


def test_exceeds_allows_for_float_rounding_only() -> None:
    assert not exceeds(0.010000000000000005, 0.01)
    assert not exceeds(0.01, 0.01)
    assert exceeds(0.0101, 0.01)


def _brute_force_with_priority(
    quality: list[list[Quality]], priority: list[bool]
) -> tuple[int, int, float]:
    best = (0, 0, 0.0)
    for chosen in product([None, *range(len(quality[0]))], repeat=len(quality)):
        picked = [(r, c) for r, c in enumerate(chosen) if c is not None]
        if len({c for _, c in picked}) < len(picked):
            continue
        if any(quality[r][c] is None for r, c in picked):
            continue
        total = round(sum(quality[r][c] or 0.0 for r, c in picked), 9)
        best = max(best, (sum(priority[r] for r, _ in picked), len(picked), total))
    return best


def test_priority_rows_match_brute_force_on_random_matrices() -> None:
    rng = random.Random(159)
    for _ in range(300):
        rows, columns = rng.randint(1, 4), rng.randint(1, 4)
        quality: list[list[Quality]] = [
            [rng.choice([None, 0.0, 0.5, 0.55, 0.9, 1.0]) for _ in range(columns)]
            for _ in range(rows)
        ]
        priority = [rng.random() < 0.5 for _ in range(rows)]
        pairs = best_pairing(quality, priority_rows=priority)
        assert len({r for r, _ in pairs}) == len({c for _, c in pairs}) == len(pairs)
        total = round(sum(quality[r][c] or 0.0 for r, c in pairs), 9)
        got = (sum(priority[r] for r, _ in pairs), len(pairs), total)
        assert got == _brute_force_with_priority(quality, priority)


def test_priority_rows_need_one_flag_per_row() -> None:
    with pytest.raises(ValueError, match="one flag for each of the 2 rows"):
        best_pairing([[0.5], [0.5]], priority_rows=[True])
