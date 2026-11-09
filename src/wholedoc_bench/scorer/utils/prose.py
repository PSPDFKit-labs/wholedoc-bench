import re
import sys
import unicodedata
from collections import Counter
from functools import cache
from itertools import groupby

import pysbd
from pysbd.languages import LANGUAGE_CODES
from rapidfuzz.distance import Levenshtein

_SAME_MEANING = str.maketrans(
    {
        "\N{LEFT SINGLE QUOTATION MARK}": "'",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{SINGLE LOW-9 QUOTATION MARK}": "'",
        "\N{SINGLE HIGH-REVERSED-9 QUOTATION MARK}": "'",
        "\N{LEFT DOUBLE QUOTATION MARK}": '"',
        "\N{RIGHT DOUBLE QUOTATION MARK}": '"',
        "\N{DOUBLE LOW-9 QUOTATION MARK}": '"',
        "\N{DOUBLE HIGH-REVERSED-9 QUOTATION MARK}": '"',
        "\N{FIGURE DASH}": "-",
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{MINUS SIGN}": "-",
        "\N{SOFT HYPHEN}": None,
        "\N{LATIN SMALL LETTER S WITH CEDILLA}": "\N{LATIN SMALL LETTER S WITH COMMA BELOW}",
        "\N{LATIN CAPITAL LETTER S WITH CEDILLA}": "\N{LATIN CAPITAL LETTER S WITH COMMA BELOW}",
        "\N{LATIN SMALL LETTER T WITH CEDILLA}": "\N{LATIN SMALL LETTER T WITH COMMA BELOW}",
        "\N{LATIN CAPITAL LETTER T WITH CEDILLA}": "\N{LATIN CAPITAL LETTER T WITH COMMA BELOW}",
    }
)
_WHITESPACE = re.compile(r"\s+")
_COMBINING_MARKS = "".join(
    f"{re.escape(chr(run[0][1]))}-{re.escape(chr(run[-1][1]))}"
    for run in (
        list(group)
        for _, group in groupby(
            enumerate(
                code
                for code in range(sys.maxunicode + 1)
                if unicodedata.category(chr(code)).startswith("M")
            ),
            key=lambda pair: pair[1] - pair[0],
        )
    )
)
_WORD = re.compile(rf"\d+(?:[.,/:-]\d+)+|[^\W_]+(?:[{_COMBINING_MARKS}]+[^\W_]*)*")
_LETTER_DIGIT_RUN = re.compile(r"[^\W_]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_SAME_MEANING)
    return _WHITESPACE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def letter_digit_runs(text: str) -> list[str]:
    return _LETTER_DIGIT_RUN.findall(normalize(text))


def phrase_count(phrase: list[str], blocks: list[list[str]]) -> int:
    n = len(phrase)
    if n == 0:
        return 0
    return sum(block[i : i + n] == phrase for block in blocks for i in range(len(block) - n + 1))


def joined_words(words_: list[str]) -> str:
    return "\x1f" + "\x1f".join(words_) + "\x1f"


def word_f1(expected: Counter[str], found: Counter[str]) -> float:
    overlap = (expected & found).total()
    if overlap == 0:
        return 0.0
    precision = overlap / found.total()
    recall = overlap / expected.total()
    return 2 * precision * recall / (precision + recall)


def edit_similarity(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(a, b)


_EXTRA_ABBREVIATIONS = {
    "ro": frozenset(
        "dl dna dra dnii dr ing prof conf lect asist str bd bdul nr ap sc et jud mun com"
        " sat tel fax etc pag vol cap art alin lit pct".split()
    ),
}


def sentences(text: str, language: str = "en") -> list[str]:
    text = normalize(text)
    if not text:
        return []
    result = [s.strip() for s in _segmenter(language).segment(text) if s.strip()]
    extra = _EXTRA_ABBREVIATIONS.get(language)
    return _merge_after(result, extra) if extra else result


def sentence_language(codes: list[str]) -> str:
    known = LANGUAGE_CODES.keys() | _EXTRA_ABBREVIATIONS.keys()
    return next((c for c in codes if c in known), "en")


def _merge_after(pieces: list[str], abbreviations: frozenset[str]) -> list[str]:
    merged: list[str] = []
    for piece in pieces:
        if merged and _last_word(merged[-1]) in abbreviations:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _last_word(sentence: str) -> str:
    if not sentence.endswith("."):
        return ""
    return sentence[:-1].rsplit(" ", 1)[-1].lower()


@cache
def _segmenter(language: str) -> pysbd.Segmenter:
    return pysbd.Segmenter(language=language if language in LANGUAGE_CODES else "en", clean=False)
