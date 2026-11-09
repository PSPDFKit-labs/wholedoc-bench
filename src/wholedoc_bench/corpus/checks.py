import warnings
from collections.abc import Callable
from itertools import islice

from wholedoc_bench.corpus.traps import chart_texts, found_traps, hidden_traps, trap_phrase
from wholedoc_bench.schema import (
    Channel,
    Element,
    Groundtruth,
    is_style_element,
    is_typography_element,
)
from wholedoc_bench.scorer.heading_tree import is_heading_element
from wholedoc_bench.scorer.tracks.t1_text import planted_in_prose
from wholedoc_bench.scorer.tracks.t7_grounding import cells_overlap
from wholedoc_bench.scorer.utils.barcodes import copies, payloads, spaced
from wholedoc_bench.scorer.utils.groundtruth import all_texts, cell_texts
from wholedoc_bench.scorer.utils.tracks.table_links import cell_tokens

MIN_TRAP_CHARACTERS = 4


class CorpusError(ValueError): ...


class CorpusWarning(UserWarning):
    pass


def check_groundtruth(truth: Groundtruth) -> None:
    _check_planted_exact(truth)
    _check_hidden_traps(truth)
    _check_barcode_payloads(truth)
    _check_heading_levels(truth)
    _check_cells_apart(truth)
    _warn_table_text_with_every_cell_word(truth)
    check_style_annotations(truth)
    check_font_annotations(truth)


def _check_planted_exact(truth: Groundtruth) -> None:
    found = {id(p) for p in planted_in_prose(truth)}
    for p in truth.planted:
        if p.kind == "exact" and id(p) not in found:
            raise CorpusError(
                f"{truth.document.id}: planted exact {p.printed!r} is not printed in body "
                "text outside tables"
            )


def _check_hidden_traps(truth: Groundtruth) -> None:
    phrases: set[tuple[str, ...]] = set()
    hidden = _channel_texts(truth, lambda e: e.channel is Channel.HIDDEN)
    body = _channel_texts(truth, lambda e: e.channel is Channel.BODY)
    other = _channel_texts(truth, lambda e: e.channel not in (Channel.BODY, Channel.HIDDEN))
    visible = [*body, *other, " ".join(body + other)]
    for trap in hidden_traps(truth):
        where = f"{truth.document.id}: planted {trap.kind} {trap.printed!r}"
        phrase = tuple(trap_phrase(trap))
        if sum(map(len, phrase)) < MIN_TRAP_CHARACTERS:
            raise CorpusError(f"{where} has fewer than {MIN_TRAP_CHARACTERS} letters and digits")
        if phrase in phrases:
            raise CorpusError(f"{where} repeats another planted string")
        phrases.add(phrase)
        if not found_traps([trap], hidden):
            raise CorpusError(f"{where} is not in the text of a hidden element")
        if found_traps([trap], visible):
            raise CorpusError(f"{where} is printed in visible text")


def _check_barcode_payloads(truth: Groundtruth) -> None:
    body = _channel_texts(truth, lambda e: e.channel is Channel.BODY)
    other = _channel_texts(truth, lambda e: e.channel not in (Channel.BODY, Channel.HIDDEN))
    visible = [spaced(text) for text in [*body, *other, " ".join(body + other)]]
    for payload in dict.fromkeys(payloads(truth)):
        if any(copies(payload, text) for text in visible):
            raise CorpusError(
                f"{truth.document.id}: barcode payload {payload!r} is printed in visible text"
            )


def _channel_texts(truth: Groundtruth, keep: Callable[[Element], bool]) -> list[str]:
    charts = [text for e in truth.elements if keep(e) for text in chart_texts(e.chart)]
    return all_texts(truth, keep) + charts


def _check_heading_levels(truth: Groundtruth) -> None:
    for e in truth.elements:
        if is_heading_element(e) and e.level is None:
            raise CorpusError(f"{truth.document.id}: heading {e.id} has no level")


def _check_cells_apart(truth: Groundtruth) -> None:
    for e in truth.elements:
        boxes = [(cell.box, i) for i, cell in enumerate(e.cells or []) if cell.box is not None]
        boxes.sort(key=lambda pair: pair[0][0])
        for n, (a, i) in enumerate(boxes):
            for b, j in islice(boxes, n + 1, None):
                if b[0] >= a[0] + a[2]:
                    break
                if cells_overlap(a, b):
                    first, second = sorted((i, j))
                    raise CorpusError(
                        f"{truth.document.id}: cells {first} and {second} of table {e.id} overlap"
                    )


def _warn_table_text_with_every_cell_word(truth: Groundtruth) -> None:
    for e in truth.elements:
        cells = cell_tokens([cell_texts(e)])
        if cells and cells <= cell_tokens([[e.text]]):
            warnings.warn(
                f"{truth.document.id}: the text of table {e.id} holds every word of its cells; "
                "a table's text holds only printed text outside the grid",
                CorpusWarning,
                stacklevel=2,
            )


def check_style_annotations(truth: Groundtruth) -> bool:
    return any(e.styles is not None for e in truth.elements if is_style_element(e))


FONT_FIELDS = ("family", "font_class", "size")


def check_font_annotations(truth: Groundtruth) -> frozenset[str]:
    return font_fields(truth)


def font_fields(truth: Groundtruth) -> frozenset[str]:
    fonts = [e.font for e in truth.elements if is_typography_element(e) and e.font is not None]
    return frozenset(f for f in FONT_FIELDS if any(getattr(x, f) is not None for x in fonts))
