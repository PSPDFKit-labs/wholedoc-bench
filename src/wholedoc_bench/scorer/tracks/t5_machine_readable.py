from collections import Counter

from wholedoc_bench.schema import Element, ElementClass, Groundtruth, LayoutItem
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.utils.barcodes import code_payloads, copies, spaced
from wholedoc_bench.scorer.utils.boxes import MIN_IOU, box_iou, item_place
from wholedoc_bench.scorer.utils.mrz import check_digits_valid, is_mrz
from wholedoc_bench.scorer.utils.pairing import Quality, pair_in_order, reaches


class BarcodePayloads(SubMetric):
    name = "barcode_payloads"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return bool(code_payloads(truth))

    def score(self, ctx: DocContext) -> float:
        expected = Counter(code_payloads(ctx.truth))
        forms = [spaced(ctx.pred.markdown), spaced(ctx.document_text)]
        found = sum(
            min(count, max(copies(payload, form) for form in forms))
            for payload, count in expected.items()
        )
        return found / expected.total()


def truth_zones(truth: Groundtruth) -> list[Element]:
    return [
        e for e in truth.elements if e.cls is ElementClass.MACHINE_READABLE and is_mrz(e.format)
    ]


def zone_pairs(ctx: DocContext) -> list[tuple[Element, LayoutItem]]:
    items = [
        item
        for item in ctx.pred.layout
        if item.cls is ElementClass.MACHINE_READABLE
        and is_mrz(item.format)
        and item.box is not None
    ]

    def quality(zone: Element, item: LayoutItem) -> Quality:
        if zone.box is None or item.box is None or zone.page != item.page:
            return None
        iou = box_iou(zone.box, item.box)
        return iou if reaches(iou, MIN_IOU) else None

    def place(item: LayoutItem) -> tuple[object, ...]:
        verdict = {True: 0, False: 1, None: 2}[item.check_digits_valid]
        return (*item_place(item, item.text), verdict)

    return pair_in_order(truth_zones(ctx.truth), items, quality, place)


class _ZoneSubMetric(SubMetric):
    source = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        return bool(truth_zones(truth))

    def right(self, zone: Element, item: LayoutItem) -> bool:
        return True

    def score(self, ctx: DocContext) -> float:
        right = sum(self.right(zone, item) for zone, item in zone_pairs(ctx))
        return right / len(truth_zones(ctx.truth))


class MrzIdentified(_ZoneSubMetric):
    name = "mrz_identified"


class MrzText(_ZoneSubMetric):
    name = "mrz_text"

    def right(self, zone: Element, item: LayoutItem) -> bool:
        return zone.payload is not None and spaced(item.text) == spaced(zone.payload)


class MrzValidity(_ZoneSubMetric):
    name = "mrz_validity"

    def right(self, zone: Element, item: LayoutItem) -> bool:
        if zone.format is None or zone.payload is None or item.check_digits_valid is None:
            return False
        return item.check_digits_valid == check_digits_valid(zone.format, zone.payload)


TRACK = Track(
    id="T5",
    name="Machine-readable content",
    parts=(
        (BarcodePayloads(), 1.0),
        (MrzIdentified(), 0.5),
        (MrzText(), 1.0),
        (MrzValidity(), 0.2),
    ),
)
