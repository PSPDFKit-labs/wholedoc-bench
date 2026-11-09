from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    variant: int
    distance: int
    length: int

    @property
    def similarity(self) -> float:
        return 1 - self.distance / self.length

    @property
    def rank(self) -> tuple[Fraction, int, int, int, int]:
        return (
            Fraction(self.distance, self.length),
            self.distance,
            self.start,
            self.end,
            self.variant,
        )


class _Occupied:
    def __init__(self, length: int, claimed: Sequence[tuple[int, int]]) -> None:
        self.length = length
        self.ranges: list[tuple[int, int]] = []
        for start, end in sorted(claimed):
            start, end = max(0, start), min(length, end)
            if start >= end:
                continue
            if self.ranges and start <= self.ranges[-1][1]:
                self.ranges[-1] = (self.ranges[-1][0], max(end, self.ranges[-1][1]))
            else:
                self.ranges.append((start, end))
        self.ends = [end for _, end in self.ranges]
        self.prefix: list[int] | None = None if self.ranges else []

    def prepare_windows(self) -> None:
        if self.prefix is not None:
            return
        changes = [0] * (self.length + 1)
        for start, end in self.ranges:
            changes[start] += 1
            changes[end] -= 1
        self.prefix = [0, *accumulate(accumulate(changes[:-1]))]

    def available(self, start: int, end: int) -> bool:
        if self.prefix is not None:
            return not self.prefix or self.prefix[start] == self.prefix[end]
        index = bisect_right(self.ends, start)
        return index == len(self.ranges) or end <= self.ranges[index][0]

    def mask(self, text: str) -> str:
        if not self.ranges:
            return text
        parts: list[str] = []
        previous = 0
        for start, end in self.ranges:
            parts.extend((text[previous:start], "\x00" * (end - start)))
            previous = end
        parts.append(text[previous:])
        return "".join(parts)


def _boundary(
    needle: str, text: str, start: int, end: int, boundary_start: bool, boundary_end: bool
) -> bool:
    return not (
        boundary_start
        and needle[0].isalnum()
        and start
        and text[start - 1].isalnum()
        or boundary_end
        and needle[-1].isalnum()
        and end < len(text)
        and text[end].isalnum()
    )


def exact_candidate(
    variants: Sequence[str],
    haystack: str,
    claimed: Sequence[tuple[int, int]] = (),
    *,
    boundary_start: bool = True,
    boundary_end: bool = True,
) -> Match | None:
    return _exact_candidate(
        variants, haystack, _Occupied(len(haystack), claimed), boundary_start, boundary_end
    )


def _exact_candidate(
    variants: Sequence[str],
    haystack: str,
    occupied: _Occupied,
    boundary_start: bool,
    boundary_end: bool,
) -> Match | None:
    candidates = []
    for index, needle in enumerate(variants):
        if not needle:
            continue
        start = haystack.find(needle)
        while start >= 0:
            end = start + len(needle)
            if occupied.available(start, end) and _boundary(
                needle, haystack, start, end, boundary_start, boundary_end
            ):
                candidates.append(Match(start, end, index, 0, len(needle)))
                break
            start = haystack.find(needle, start + 1)
    return min(candidates, key=lambda c: c.rank, default=None)


def _seed_starts(needle: str, haystack: str, k: int) -> set[int]:
    starts: set[int] = set()
    for index in range(k + 1):
        a, b = index * len(needle) // (k + 1), (index + 1) * len(needle) // (k + 1)
        seed = needle[a:b]
        occurrence = haystack.find(seed)
        while occurrence >= 0:
            nominal = occurrence - a
            starts.update(range(max(0, nominal - k), min(len(haystack), nominal + k) + 1))
            occurrence = haystack.find(seed, occurrence + 1)
    return starts


def find_candidate(
    variants: Sequence[str],
    haystack: str,
    claimed: Sequence[tuple[int, int]] = (),
    *,
    reference: bool = False,
    boundary_start: bool = True,
    boundary_end: bool = True,
) -> Match | None:
    occupied = _Occupied(len(haystack), claimed)
    exact = _exact_candidate(variants, haystack, occupied, boundary_start, boundary_end)
    if exact is not None:
        return exact
    best: Match | None = None
    if not reference:
        masked = occupied.mask(haystack)
        for variant, needle in enumerate(variants):
            if not needle:
                continue
            proposal = fuzz.partial_ratio_alignment(needle, masked)
            if proposal is None:
                continue
            start, end = proposal.dest_start, proposal.dest_end
            length = max(len(needle), end - start)
            if not occupied.available(start, end) or not _boundary(
                needle, haystack, start, end, boundary_start, boundary_end
            ):
                continue
            distance = Levenshtein.distance(needle, haystack[start:end])
            if 10 * distance <= length:
                candidate = Match(start, end, variant, distance, length)
                if best is None or candidate.rank < best.rank:
                    best = candidate
    for variant, needle in enumerate(variants):
        m = len(needle)
        if not m:
            continue
        minimum, maximum = (9 * m + 9) // 10, 10 * m // 9
        k = maximum // 10
        if best is not None:
            k = min(k, best.distance * maximum // best.length)
            minimum, maximum = max(minimum, m - k), min(maximum, m + k)
        starts = (
            range(max(0, len(haystack) - minimum + 1))
            if reference
            else _seed_starts(needle, haystack, k)
        )
        if len(starts) * (maximum - minimum + 1) >= len(haystack):
            occupied.prepare_windows()
        for start in starts:
            if boundary_start and needle[0].isalnum() and start and haystack[start - 1].isalnum():
                continue
            for length in range(minimum, min(maximum, len(haystack) - start) + 1):
                end = start + length
                if not occupied.available(start, end):
                    break
                if not _boundary(needle, haystack, start, end, boundary_start, boundary_end):
                    continue
                limit = max(m, length) // 10
                if best is not None:
                    limit = min(limit, best.distance * max(m, length) // best.length)
                distance = Levenshtein.distance(needle, haystack[start:end], score_cutoff=limit)
                if distance > limit:
                    continue
                candidate = Match(start, end, variant, distance, max(m, length))
                if best is None or candidate.rank < best.rank:
                    best = candidate
    return best


def allocate(
    variants_by_source: Sequence[Sequence[str]], haystack: str, *, reference: bool = False
) -> dict[int, Match]:
    order = sorted(
        range(len(variants_by_source)),
        key=lambda i: (-max(map(len, variants_by_source[i]), default=0), i),
    )
    claimed: list[tuple[int, int]] = []
    found: dict[int, Match] = {}
    for exact in (True, False):
        for index in order:
            if index in found:
                continue
            candidate = (
                exact_candidate(variants_by_source[index], haystack, claimed)
                if exact
                else find_candidate(
                    variants_by_source[index], haystack, claimed, reference=reference
                )
            )
            if candidate is not None:
                found[index] = candidate
                claimed.append((candidate.start, candidate.end))
    return found


def align(source: str, output: str, distance: int) -> list[tuple[int, int]]:
    width = distance
    previous = {j: j for j in range(min(len(output), width) + 1)}
    traces: list[tuple[int, bytearray]] = [(0, bytearray([3] * (len(previous))))]
    infinity = len(source) + len(output) + 1
    for i, a in enumerate(source, 1):
        start, end = max(0, i - width), min(len(output), i + width)
        current: dict[int, int] = {}
        trace = bytearray()
        for j in range(start, end + 1):
            diagonal = previous.get(j - 1, infinity) + (a != output[j - 1]) if j else infinity
            deletion = previous.get(j, infinity) + 1
            insertion = current.get(j - 1, infinity) + 1
            cost = min(diagonal, deletion, insertion)
            direction = 1 if diagonal == cost else 2 if deletion == cost else 3
            current[j] = cost
            trace.append(direction)
        traces.append((start, trace))
        previous = current
    if previous.get(len(output)) != distance:
        raise ValueError("alignment band does not contain the supplied edit distance")
    i, j = len(source), len(output)
    pairs: list[tuple[int, int]] = []
    while i or j:
        start, trace = traces[i]
        direction = trace[j - start] if i else 3
        if direction == 1:
            i -= 1
            j -= 1
            pairs.append((i, j))
        elif direction == 2:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))
