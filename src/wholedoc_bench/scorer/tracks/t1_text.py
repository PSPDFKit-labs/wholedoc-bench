from collections import Counter
from collections.abc import Sequence

from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
from scipy.optimize import linear_sum_assignment

from wholedoc_bench.schema import ElementClass, Groundtruth, Planted
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.barcodes import without_payloads
from wholedoc_bench.scorer.utils.blocks import BlockKind, comparable_text
from wholedoc_bench.scorer.utils.groundtruth import body_text_variants, body_texts
from wholedoc_bench.scorer.utils.prose import (
    joined_words,
    phrase_count,
    sentence_language,
    sentences,
    word_f1,
    words,
)

NOT_SCORED = frozenset({ElementClass.TABLE, ElementClass.CHART})


def truth_prose(truth: Groundtruth) -> list[str]:
    return body_texts(truth, skip=NOT_SCORED)


def truth_prose_variants(truth: Groundtruth) -> list[tuple[str, ...]]:
    return body_text_variants(truth, skip=NOT_SCORED)


def output_prose(ctx: DocContext) -> list[str]:
    prose = [b.text for b in ctx.blocks if b.kind is not BlockKind.TABLE]
    return without_payloads(ctx.truth, prose)


class WordBagF1(SubMetric):
    name = "word_bag_f1"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return any(words(text) for text in truth_prose(truth))

    def score(self, ctx: DocContext) -> float:
        found = Counter(w for text in output_prose(ctx) for w in words(text))
        expected: Counter[str] = Counter()
        for variants in truth_prose_variants(ctx.truth):
            bags = [Counter(words(text)) for text in variants]
            expected += max(bags, key=lambda bag: (bag & found).total())
        return word_f1(expected, found)


class SentenceIntegrity(SubMetric):
    name = "sentence_integrity"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        language = sentence_language(truth.document.language)
        return any(_sentence_words(text, language) for text in truth_prose(truth))

    def score(self, ctx: DocContext) -> float:
        language = sentence_language(ctx.truth.document.language)
        blocks = [joined_words(words(text)) for text in output_prose(ctx)]
        passed = total = 0
        for variants in truth_prose_variants(ctx.truth):
            results = []
            for text in variants:
                expected = _sentence_words(text, language)
                found = sum(any(joined_words(s) in block for block in blocks) for s in expected)
                results.append((found, len(expected)))
            best = max(results, key=lambda r: r[0])
            passed, total = passed + best[0], total + best[1]
        return passed / total


def _sentence_words(text: str, language: str) -> list[list[str]]:
    return [w for s in sentences(text, language) if (w := words(s))]


class OrthographyDiacritics(SubMetric):
    name = "orthography_diacritics"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return bool(planted_in_prose(truth))

    def score(self, ctx: DocContext) -> float:
        expected = [words(text) for text in truth_prose(ctx.truth)]
        found = [words(text) for text in output_prose(ctx)]
        items = planted_in_prose(ctx.truth)
        passed = sum(
            phrase_count(words(item.printed), found) >= phrase_count(words(item.printed), expected)
            for item in items
        )
        return passed / len(items)


def planted_in_prose(truth: Groundtruth) -> list[Planted]:
    prose = [words(text) for text in truth_prose(truth)]
    return [p for p in truth.planted if p.kind == "exact" and phrase_count(words(p.printed), prose)]


MIN_SIMILARITY = 0.5


class BlockEditSimilarity(SubMetric):
    name = "block_edit_similarity"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return any(comparable_text(text) for text in truth_prose(truth))

    def score(self, ctx: DocContext) -> float:
        expected = [
            spellings
            for variants in truth_prose_variants(ctx.truth)
            if (spellings := tuple(t for text in variants if (t := comparable_text(text))))
        ]
        found = [t for text in output_prose(ctx) if (t := comparable_text(text))]
        best = best_similarities(expected, found)
        weights = [len(spellings[0]) for spellings in expected]
        return sum(w * s for w, s in zip(weights, best, strict=True)) / sum(weights)


def best_similarities(expected: Sequence[str | tuple[str, ...]], found: list[str]) -> list[float]:
    rows = [(e,) if isinstance(e, str) else e for e in expected]
    if not rows or not found:
        return [0.0] * len(rows)
    flat = [text for spellings in rows for text in spellings]
    matrix = cdist(flat, found, scorer=Levenshtein.normalized_similarity).tolist()
    similarities: list[list[float]] = []
    start = 0
    for spellings in rows:
        block = matrix[start : start + len(spellings)]
        similarities.append([max(column) for column in zip(*block, strict=True)])
        start += len(spellings)
    return best_pairing(similarities, [len(spellings[0]) for spellings in rows])


def best_pairing(similarities: list[list[float]], weights: list[int]) -> list[float]:
    kept = [[s if s >= MIN_SIMILARITY else 0.0 for s in row] for row in similarities]
    gains = [[w * s for s in row] for w, row in zip(weights, kept, strict=True)]
    result = [0.0] * len(similarities)
    rows, cols = linear_sum_assignment(gains, maximize=True)
    for r, c in zip(rows, cols, strict=True):
        result[r] = kept[r][c]
    return result


TRACK = Track(
    id="T1",
    name="Text fidelity",
    parts=(
        (WordBagF1(), 1.0),
        (SentenceIntegrity(), 1.0),
        (OrthographyDiacritics(), 1.0),
        (BlockEditSimilarity(), 0.5),
    ),
)
