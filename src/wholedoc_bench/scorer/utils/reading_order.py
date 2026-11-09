from collections.abc import Collection

from wholedoc_bench.schema import Groundtruth


def in_reading_order(truth: Groundtruth, ids: Collection[str]) -> list[str]:
    wanted = set(ids)
    listed = [ref for ref in truth.reading_order if ref in wanted]
    seen = set(listed)
    return listed + [e.id for e in truth.elements if e.id in wanted and e.id not in seen]
