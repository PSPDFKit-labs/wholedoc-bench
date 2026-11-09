from wholedoc_bench.schema import Box, LayoutItem

MIN_IOU = 0.5

Place = tuple[int, bool, tuple[float, ...], str, str]


def _shared_length(a_start: float, a_length: float, b_start: float, b_length: float) -> float:
    shared = min(a_start + a_length, b_start + b_length) - max(a_start, b_start)
    return min(max(0.0, shared), a_length, b_length)


def _overlap(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return _shared_length(ax, aw, bx, bw) * _shared_length(ay, ah, by, bh)


def _area(box: Box) -> float:
    return box[2] * box[3]


def box_iou(a: Box, b: Box) -> float:
    overlap = _overlap(a, b)
    return overlap / (_area(a) + _area(b) - overlap)


def box_share_inside(inner: Box, outer: Box) -> float:
    return _overlap(inner, outer) / _area(inner)


def box_place(box: Box | None) -> tuple[bool, tuple[float, ...]]:
    return (True, ()) if box is None else (False, (box[1], box[0], box[3], box[2]))


def item_place(item: LayoutItem, text: str) -> Place:
    return (item.page, *box_place(item.box), text, item.cls.value)
