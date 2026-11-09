from collections import Counter
from collections.abc import Callable, Sequence

from wholedoc_bench.schema import Element, Groundtruth, LayoutItem, Prediction, RelationKind
from wholedoc_bench.scorer.utils.boxes import MIN_IOU, Place, box_iou, item_place
from wholedoc_bench.scorer.utils.groundtruth import cell_texts
from wholedoc_bench.scorer.utils.pairing import Quality, pair_in_order, reaches
from wholedoc_bench.scorer.utils.prose import edit_similarity
from wholedoc_bench.scorer.utils.reading_order import in_reading_order

SCORED_KINDS = (RelationKind.CAPTION_OF, RelationKind.FOOTNOTE_OF)
TEXT_THRESHOLD = 0.9
TEXT_ONLY_QUALITY = MIN_IOU / 2

Normalize = Callable[[str], str]
Link = tuple[RelationKind, str, str]


def _unchanged(text: str) -> str:
    return text


def truth_links(truth: Groundtruth) -> set[Link]:
    return {(r.kind, r.source, r.target) for r in truth.relations if r.kind in SCORED_KINDS}


def has_links(truth: Groundtruth) -> bool:
    return bool(truth_links(truth))


def element_of_item(
    truth: Groundtruth, layout: Sequence[LayoutItem], *, normalize: Normalize = _unchanged
) -> dict[str, str]:
    ids = Counter(item.id for item in layout)
    places = Counter(_place(item, normalize) for item in layout)
    items = [item for item in layout if ids[item.id] == 1 and places[_place(item, normalize)] == 1]
    elements = {e.id: e for e in truth.elements if e.box is not None or _match_text(e)}
    found: dict[str, str] = {}
    for page in sorted({item.page for item in items}):
        rows = [elements[i] for i in in_reading_order(truth, elements) if elements[i].page == page]
        columns = [item for item in items if item.page == page]
        for element, item in _pair_on_page(rows, columns, normalize):
            found[item.id] = element.id
    return found


def _pair_on_page(
    rows: list[Element], columns: list[LayoutItem], normalize: Normalize
) -> list[tuple[Element, LayoutItem]]:
    row_texts = [normalize(_match_text(e)) for e in rows]
    column_texts = [normalize(item.text) for item in columns]
    similar = [[edit_similarity(a, b) >= TEXT_THRESHOLD for b in column_texts] for a in row_texts]
    matches = [sum(row[c] for row in similar) for c in range(len(columns))]
    pairs = pair_in_order(
        range(len(rows)),
        range(len(columns)),
        lambda r, c: _quality(
            rows[r], columns[c], similar[r][c] and matches[c] == 1 and bool(row_texts[r].strip())
        ),
        lambda c: _place(columns[c], normalize),
    )
    return [(rows[r], columns[c]) for r, c in pairs]


def score_relations(
    truth: Groundtruth, pred: Prediction, *, normalize: Normalize = _unchanged
) -> float:
    expected = truth_links(truth)
    if not expected:
        raise ValueError(f"{truth.document.id}: no caption or footnote link to score")
    element_of = element_of_item(truth, pred.layout, normalize=normalize)
    predicted = {(r.kind, r.source, r.target) for r in pred.relations if r.kind in SCORED_KINDS}
    right = sum(
        (kind, element_of.get(source), element_of.get(target)) in expected
        for kind, source, target in predicted
    )
    return 2 * right / (len(expected) + len(predicted))


def _match_text(element: Element) -> str:
    return " ".join([element.text, *cell_texts(element)]).strip()


def _quality(element: Element, item: LayoutItem, text_points_here: bool) -> Quality:
    if element.box is not None and item.box is not None:
        iou = box_iou(element.box, item.box)
        if reaches(iou, MIN_IOU):
            return min(iou, 1.0)
    return TEXT_ONLY_QUALITY if text_points_here else None


def _place(item: LayoutItem, normalize: Normalize) -> Place:
    return item_place(item, normalize(item.text))
