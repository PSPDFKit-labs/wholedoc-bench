import re
from collections import Counter

from wholedoc_bench.schema import ElementClass, Groundtruth
from wholedoc_bench.scorer.utils.mrz import is_mrz

_CUT = "\0"


def payloads(truth: Groundtruth) -> list[str]:
    return [
        spaced(e.payload)
        for e in truth.elements
        if e.cls is ElementClass.MACHINE_READABLE and e.payload is not None
    ]


def code_payloads(truth: Groundtruth) -> list[str]:
    return [
        spaced(e.payload)
        for e in truth.elements
        if e.cls is ElementClass.MACHINE_READABLE and e.payload is not None and not is_mrz(e.format)
    ]


def copies(payload: str, text: str) -> int:
    return len(_whole_code(payload).findall(text))


def without_payloads(truth: Groundtruth, texts: list[str]) -> list[str]:
    expected = Counter(payloads(truth))
    pieces = [spaced(text) for text in texts]
    joined = " ".join(pieces)
    for payload in sorted(expected, key=len, reverse=True):
        for match in list(_whole_code(payload).finditer(joined))[: expected[payload]]:
            start, end = match.span()
            joined = joined[:start] + _CUT * (end - start) + joined[end:]
    result: list[str] = []
    offset = 0
    for text, piece in zip(texts, pieces, strict=True):
        kept = joined[offset : offset + len(piece)]
        offset += len(piece) + 1
        if _CUT not in kept:
            result.append(text)
        elif rest := spaced(kept.replace(_CUT, " ")):
            result.append(rest)
    return result


def spaced(text: str) -> str:
    return " ".join(text.split())


def _whole_code(payload: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(payload)}(?!\w)")
