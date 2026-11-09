from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import pytest

from tests.builders import (
    CELL_HEIGHT,
    EMPTY_AREA,
    FIGURE,
    FOOTER,
    FULL_DOCUMENT,
    GRID,
    GRID_LEFT,
    GRID_TABLE,
    GRID_TOP,
    HEADING,
    HEADING_TEXT,
    OTHER_TEXT,
    PARAGRAPH,
    PARAGRAPH_TEXT,
    TABLE,
    THREE_ELEMENTS,
    beside_grid,
    boxed,
    boxed_item,
    cell,
    element_as_item,
    grid_area,
    grid_box,
    grid_cells,
    grid_table,
    prediction,
    table_item,
    truth,
    truth_as_items,
)
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.schema import Box, Groundtruth, LayoutItem, Prediction, TableCell
from wholedoc_bench.scorer.base import SubMetric
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t7_grounding import (
    TRACK,
    ElementBoxes,
    ElementClasses,
    TableCellBoxes,
)

Case = tuple[Groundtruth, list[LayoutItem], float]
Group = Mapping[str, tuple[list[LayoutItem], float]]
Cells = list[dict[str, Any]]

BOXES, CLASSES, CELLS = ElementBoxes(), ElementClasses(), TableCellBoxes()
SUB_METRICS = [BOXES, CLASSES, CELLS]

NARROWER_PARAGRAPH: Box = (0.1, 0.15, 0.55, 0.1)
SHIFTED_HEADING: Box = (0.1, 0.06, 0.5, 0.05)
TEXT_LAYER: Box = (0.1, 0.65, 0.8, 0.2)
SQUARE: Box = (0.25, 0.25, 0.5, 0.5)
SLIVER: Box = (0.3, 0.0, 2.0**-55, 0.1)
FIVE_WORDS = "one two three four five"
FOUR_OF_FIVE_WORDS = "one two three four six"
HALF_OF_FOUR_WORDS = "one two nine ten"
SEVEN_WORDS = "one two three four five six seven"
STRAY_TEXT = "Stray words"
TEXT_LAYER_WORDS = "scanned page text layer"
TABLE_TITLE = "Sales by region today"
TABLE_CELLS_TEXT = "Name Value alpha 1 beta 2"
REVENUE_TITLE = "Revenue in million euros"
SHARED_TITLE = "beta three sales one"
SHARED_HEADER = ["three beta", "one beta"]

SCORED_PARAGRAPH = boxed("p", "paragraph", PARAGRAPH, PARAGRAPH_TEXT)
SECOND_PARAGRAPH = boxed("q", "paragraph", (0.1, 0.3, 0.6, 0.1), "Profits rose in new markets.")
HEADING_ELEMENT = boxed("h", "heading", HEADING, HEADING_TEXT, level=1)
PAGE_NUMBER = boxed("f", "footer", FOOTER, "1", channel="furniture")
ONE_PARAGRAPH = truth(SCORED_PARAGRAPH)
HEADING_AND_PARAGRAPH = truth(HEADING_ELEMENT, SCORED_PARAGRAPH)
HEADING_AND_TWO_PARAGRAPHS = truth(HEADING_ELEMENT, SCORED_PARAGRAPH, SECOND_PARAGRAPH)
FOUR_ELEMENTS = truth(HEADING_ELEMENT, SCORED_PARAGRAPH, SECOND_PARAGRAPH, PAGE_NUMBER)
SECOND_HEADING = boxed("k", "heading", (0.1, 0.45, 0.5, 0.05), "Outlook for next year", level=1)
TWO_HEADINGS_TWO_PARAGRAPHS = truth(
    HEADING_ELEMENT, SCORED_PARAGRAPH, SECOND_HEADING, SECOND_PARAGRAPH
)
TWO_PAGES_WITH_FOOTERS = truth(
    HEADING_ELEMENT,
    PAGE_NUMBER,
    boxed("p", "paragraph", PARAGRAPH, PARAGRAPH_TEXT, page=1),
    boxed("g", "footer", FOOTER, "2", page=1, channel="furniture"),
)
THREE_PAGES = truth(
    boxed("a", "paragraph", PARAGRAPH, PARAGRAPH_TEXT),
    boxed("b", "paragraph", PARAGRAPH, PARAGRAPH_TEXT, page=1),
    boxed("c", "paragraph", TABLE, OTHER_TEXT, page=2),
)
PARAGRAPH_COUNT = 5
FIVE_PARAGRAPHS_AND_A_CAPTION = truth(
    *(
        boxed(f"p{i}", "paragraph", (0.1, 0.02 + 0.12 * i, 0.6, 0.1), f"Section {i} of the report")
        for i in range(PARAGRAPH_COUNT)
    ),
    boxed("c", "caption", (0.1, 0.8, 0.6, 0.05), "Figure 1 shows the sales per region"),
)
TABLE_ROWS = [["alpha", "1"], ["beta", "2"]]
TITLED_TABLE = truth(
    boxed("t", "table", TABLE, TABLE_TITLE, header=["Name", "Value"], rows=TABLE_ROWS)
)
TEXT_LAYER_OVER = partial(boxed, "o", "paragraph", text=TEXT_LAYER_WORDS, channel="hidden")
FIGURE_WITH_TEXT_LAYER = truth(boxed("g", "figure", FIGURE), TEXT_LAYER_OVER(TEXT_LAYER))
FIGURE_WITH_HIDDEN_TEXT = truth(
    boxed("g", "figure", FIGURE, TEXT_LAYER_WORDS), TEXT_LAYER_OVER(FIGURE)
)

OTHER_GRID = [["City", "Size", "Rank"], ["gamma", "3", "km"], ["delta", "4", "s"]]
SMALL_GRID = [["Name", "Value"], ["alpha", "1"]]
ZEROS = [["A", "B", "C"], ["0", "0", "0"], ["0", "0", "0"]]
NAME, ALPHA, ONE, KG, LAST = 0, 3, 4, 5, 8
LOWER_ROWS = 4
AREA = grid_area(GRID)
LOWER_AREA: Box = (GRID_LEFT, GRID_TOP + LOWER_ROWS * CELL_HEIGHT, *grid_area(OTHER_GRID)[2:])
GRID_CELLS = grid_cells(GRID)
LOWER_CELLS = [
    cell(c["text"], grid_box(i // 3 + LOWER_ROWS, i % 3))
    for i, c in enumerate(grid_cells(OTHER_GRID))
]
CELL_TABLE = truth(GRID_TABLE)
CELL_DOCUMENT = truth(HEADING_ELEMENT, GRID_TABLE, PAGE_NUMBER)
EVERY_GRID_WORD = " ".join(text for row in GRID for text in row)
MERGED_HEADER = [cell(text, grid_box(0, c)) for c, text in enumerate(["Region", "Sales", "Share"])]
MERGED_TABLE = truth(
    boxed(
        "t",
        "table",
        (0.1, 0.3, 0.6, 0.2),
        header=["Region", "Sales", "Share"],
        rows=[["All regions", "100"]],
        cells=[
            *MERGED_HEADER,
            cell("All regions", (0.1, 0.4, 0.4, 0.1)),
            cell("100", grid_box(1, 2)),
        ],
    )
)
WIDE: Box = (GRID_LEFT, 0.4, 0.3, CELL_HEIGHT)
EMPTY: Box = (GRID_LEFT + 0.3, 0.4, 0.2, CELL_HEIGHT)
EMPTY_NEIGHBOUR_CELLS = [
    cell("Name", (GRID_LEFT, GRID_TOP, 0.3, CELL_HEIGHT)),
    cell("Note", (GRID_LEFT + 0.3, GRID_TOP, 0.2, CELL_HEIGHT)),
    cell("alpha", WIDE),
    cell("", EMPTY),
]
EMPTY_NEIGHBOUR = truth(
    boxed(
        "t",
        "table",
        (GRID_LEFT, GRID_TOP, 0.5, 0.2),
        header=["Name", "Note"],
        rows=[["alpha", ""]],
        cells=EMPTY_NEIGHBOUR_CELLS,
    )
)
KG_OVER_THE_NEUTRAL_CELL: Box = (0.5, 0.4, 0.28, 0.1)
KG_OVER_HALF_OF_THE_NOTE: Box = (0.5, 0.4, 0.3, 0.1)

with_paragraph = partial(truth, SCORED_PARAGRAPH)


def f1(pairs: int, expected: int, counted: int) -> float:
    return 2 * pairs / (expected + counted)


def score(metric: SubMetric, t: Groundtruth, pred: Prediction | list[LayoutItem]) -> float:
    layout = pred if isinstance(pred, Prediction) else prediction(layout=pred)
    return metric.score(DocContext(t, layout))


def flipped(items: list[LayoutItem]) -> list[LayoutItem]:
    return [i.model_copy(update={"cells": i.cells[::-1]}) if i.cells else i for i in items[::-1]]


def round_trip(t: Groundtruth, pred: Prediction) -> tuple[Groundtruth, Prediction]:
    truth_again = Groundtruth.model_validate_json(t.model_dump_json(by_alias=True))
    return truth_again, Prediction.model_validate_json(pred.model_dump_json(by_alias=True))


def cases(
    *groups: tuple[SubMetric, Groundtruth | None, Mapping[str, tuple[Any, ...]]],
) -> tuple[list[tuple[Any, ...]], list[str]]:
    rows = [(m, *(() if t is None else (t,)), *row) for m, t, g in groups for row in g.values()]
    return rows, [f"{m.name}: {name}" for m, _, g in groups for name in g]


def item(
    box: Box | None, text: str = PARAGRAPH_TEXT, cls: str = "paragraph", page: int = 0, **extra: Any
) -> LayoutItem:
    return boxed_item("x", cls, box, text, page, **extra)


def element(cls: str, box: Box | None, text: str = "", channel: str = "body") -> dict[str, Any]:
    return {"id": "n", "class": cls, "page": 0, "box": box, "text": text, "channel": channel}


def annotated(box: Box | None, text: str = PARAGRAPH_TEXT) -> dict[str, Any]:
    return element("paragraph", box, text, "annotation")


def only(cls: str, box: Box, text: str = PARAGRAPH_TEXT, **extra: Any) -> Groundtruth:
    return truth(boxed("e", cls, box, text, **extra))


def resized(box: Box, share: float, shift: float = 0.0) -> Box:
    x, y, w, h = box
    return (x + w * shift, y, w * share, h)


def twice(items: list[LayoutItem]) -> list[LayoutItem]:
    return [*items, *(i.model_copy(update={"id": f"{i.id}-copy"}) for i in items)]


def relabelled(found: LayoutItem, cls: str) -> LayoutItem:
    return LayoutItem.model_validate({**found.model_dump(by_alias=True), "class": cls})


def labelled(t: Groundtruth, cls: str, *ids: str) -> list[LayoutItem]:
    return [relabelled(i, cls) if not ids or i.id in ids else i for i in truth_as_items(t)]


def items_without(t: Groundtruth, *ids: str) -> list[LayoutItem]:
    return [element_as_item(e) for e in t.elements if e.id not in ids]


def stray(cls: str) -> LayoutItem:
    return item(EMPTY_AREA, STRAY_TEXT, cls)


def per_channel(channel: str, cls: str, box: Box | None, group: Group) -> dict[str, Case]:
    return {
        f"{name}, {channel}": (
            with_paragraph(element(cls, box, OTHER_TEXT, channel)),
            [PARAGRAPH_ITEM, *extra],
            score,
        )
        for name, (extra, score) in group.items()
    }


def in_both_orders(
    table: Mapping[str, tuple[dict[str, Any], dict[str, Any], list[LayoutItem]]],
) -> dict[str, Case]:
    return {
        f"{name}, {order}": (truth(*pair), items, 1)
        for name, (scored, neutral, items) in table.items()
        for order, pair in [
            ("scored first", (scored, neutral)),
            ("neutral first", (neutral, scored)),
        ]
    }


def on_table(cells: Cells, page: int = 0) -> list[LayoutItem]:
    return [table_item(cells, AREA, page)]


def grid_plus(*extra: dict[str, Any]) -> list[LayoutItem]:
    return on_table([*GRID_CELLS, *extra])


def cells_with(index: int, box: Box | None, text: str | None = None) -> Cells:
    cells = grid_cells(GRID)
    cells[index] = cell(cells[index]["text"] if text is None else text, box)
    return cells


def moved(index: int, box: Box | None, text: str | None = None) -> list[LayoutItem]:
    return on_table(cells_with(index, box, text))


def tight(box: Box) -> Box:
    x, y, w, h = box
    return (x + w * 0.1, y + h * 0.2, w * 0.3, h * 0.5)


def unboxed(cells: Cells) -> Cells:
    return [cell(c["text"], None) for c in cells]


def table_of(
    texts: list[list[str]], box: Box | None, cells: Cells | None = None, channel: str = "body"
) -> dict[str, Any]:
    fields = {"box": box, "header": texts[0], "rows": texts[1:], "cells": cells}
    return {"id": "u", "class": "table", "page": 0, "channel": channel, **fields}


def beside_table(
    texts: list[list[str]], box: Box | None, cells: Cells | None, channel: str
) -> Groundtruth:
    return beside_grid(table_of(texts, box, cells, channel))


def beside_cell(text: str, box: Box, channel: str) -> Groundtruth:
    return beside_grid(
        boxed("n", "table", box, header=[text], cells=[cell(text, box)], channel=channel)
    )


def alpha_value_cases(value: str) -> dict[str, Case]:
    texts = [GRID[0], ["alpha", value, "kg"], GRID[2]]
    t, left_out = (
        truth(grid_table("t", texts)),
        [c for c in grid_cells(texts) if c["text"] != value],
    )
    return {
        f"alpha value {value!r} left out": (t, on_table(left_out), 1),
        f"alpha value {value!r} returned": (t, on_table(grid_cells(texts)), 1),
    }


def merged_row(first: str, second: str) -> list[LayoutItem]:
    row = [cell(first, grid_box(1, 0)), cell(second, grid_box(1, 1)), cell("100", grid_box(1, 2))]
    return on_table([*MERGED_HEADER, *row])


def two_column_table(first_width: float, second_width: float) -> Groundtruth:
    texts = [["Key", "Value"], ["alpha", "one"], ["beta", "two"]]
    widths, lefts = [first_width, second_width], [GRID_LEFT, GRID_LEFT + first_width]
    cells = [
        cell(text, (lefts[c], GRID_TOP + CELL_HEIGHT * r, widths[c], CELL_HEIGHT))
        for r, row in enumerate(texts)
        for c, text in enumerate(row)
    ]
    area: Box = (GRID_LEFT, GRID_TOP, first_width + second_width, CELL_HEIGHT * len(texts))
    return truth(boxed("t", "table", area, header=texts[0], rows=texts[1:], cells=cells))


def row_box(boxes: list[Box]) -> Box:
    left, right = min(b[0] for b in boxes), max(b[0] + b[2] for b in boxes)
    return (left, boxes[0][1], right - left, boxes[0][3])


def row_box_cells(cells: list[TableCell]) -> Cells:
    rows: dict[tuple[float, float], list[Box]] = defaultdict(list)
    for c in cells:
        if c.box is not None:
            rows[(c.box[1], c.box[3])].append(c.box)
    return [cell(c.text, row_box(rows[(c.box[1], c.box[3])])) for c in cells if c.box is not None]


def row_box_items(t: Groundtruth) -> list[LayoutItem]:
    tables = [e for e in t.elements if e.cells is not None]
    return [table_item(row_box_cells(e.cells or []), e.box, e.page, id_=e.id) for e in tables]


def document_with_table(table: LayoutItem) -> list[LayoutItem]:
    heading, _, footer = truth_as_items(CELL_DOCUMENT)
    return [heading, table, footer]


PARAGRAPH_ITEM = item(PARAGRAPH)
SHIFTED_ITEM = item(resized(PARAGRAPH, 1, shift=0.2))
NARROWER_ITEM = item(NARROWER_PARAGRAPH)
HEADING_ITEM = item(HEADING, HEADING_TEXT, "heading")
UNBOXED_HEADING = item(None, HEADING_TEXT, "heading")
WRONG_HEADING = item(PARAGRAPH, cls="heading")
HEADING_AS_ITEM = element_as_item(HEADING_AND_PARAGRAPH.elements[0])
FIGURE_ITEM = item(FIGURE, "", "figure")
TALLER_FIGURE_ITEM = item((0.1, 0.5, 0.8, 0.4), "", "figure")
THREE_ITEMS = truth_as_items(THREE_ELEMENTS)
FULL_ITEMS = truth_as_items(FULL_DOCUMENT)
ALL_PARAGRAPHS = labelled(FOUR_ELEMENTS, "paragraph")
CAPTION_AND_NARROWER = [item(PARAGRAPH, cls="caption"), item(resized(PARAGRAPH, 0.8))]
NARROWER_COPY = annotated(NARROWER_PARAGRAPH)
HIDDEN_PARAGRAPH_COPY = element("paragraph", PARAGRAPH, PARAGRAPH_TEXT, "hidden")
PARAGRAPH_WITH_HIDDEN_COPY = with_paragraph(HIDDEN_PARAGRAPH_COPY)
SHIFTED_HIDDEN_HEADING = element("paragraph", SHIFTED_HEADING, HEADING_TEXT, "hidden")
HEADING_WITH_SHIFTED_COPY = truth(HEADING_ELEMENT, SHIFTED_HIDDEN_HEADING)
HIDDEN_HEADING_COPIES = {
    "boxed copy": truth(HEADING_ELEMENT, SHIFTED_HIDDEN_HEADING, SCORED_PARAGRAPH),
    "off page copy": truth(
        HEADING_ELEMENT, element("paragraph", None, HEADING_TEXT, "hidden"), SCORED_PARAGRAPH
    ),
}
CAPTION_BOXES = {"heading box": HEADING, "no box": None}

SAME_PAGE_TABLES = beside_grid(table_of(OTHER_GRID, LOWER_AREA, LOWER_CELLS))
UNBOXED_BODY_TABLE = beside_grid(table_of(OTHER_GRID, None, LOWER_CELLS))
GRID_ITEM = table_item(GRID_CELLS, AREA, id_="a")
TIGHT_CELLS = [cell(c["text"], tight(c["box"])) for c in GRID_CELLS]
TIGHT_ZEROS = [cell(c["text"], tight(c["box"])) for c in grid_cells(ZEROS)]
TIGHT_ALPHA = cell("alpha", tight(grid_box(1, 0)))
WHOLE_TABLE_CELL = cell(EVERY_GRID_WORD, AREA)
STRANGERS = [cell(f"stranger {i}", None) for i in range(len(LOWER_CELLS))]
FAR_LOWER_CELLS = [cell(c["text"], (0.8, 0.85, 0.1, 0.1)) for c in LOWER_CELLS]
JUNK = [cell(f"junk {i}", c["box"]) for i, c in enumerate(GRID_CELLS)]
TABLE_ITEM_WITHOUT_CELLS = item(AREA, EVERY_GRID_WORD, "table")
NOTE_CELL = cell("Note", grid_box(0, 0))
ONE_CELL_LOST, ONE_CELL_EXTRA, CELLS_TWICE = f1(8, 9, 9), f1(9, 9, 10), f1(9, 9, 18)
ONE_EXTRA = f1(1, 1, 2)

NEUTRAL_TABLE = partial(beside_table, OTHER_GRID, LOWER_AREA, LOWER_CELLS)
NEUTRAL_COPY = partial(beside_table, GRID, AREA, GRID_CELLS)
WITHOUT_CELLS = partial(beside_table, OTHER_GRID, LOWER_AREA, None)
NEUTRAL_TABLES_WITH_THE_BODY_WORDS: dict[str, Callable[[str], Groundtruth]] = {
    "over_the_body_table": partial(beside_table, GRID, AREA, None),
    "off_page": partial(beside_table, GRID, None, None),
}
OVER_THE_BODY = NEUTRAL_TABLES_WITH_THE_BODY_WORDS["over_the_body_table"]
WRONG_BODY_CELLS = {
    "wrong_column": [
        *GRID_CELLS,
        *(
            cell(t, grid_box(r, 1 - c))
            for r, c, t in [(1, 0, "alpha"), (1, 1, "1"), (2, 0, "beta"), (2, 1, "2")]
        ),
    ],
    "row_boxes": [
        *GRID_CELLS[:3],
        cell("alpha 1 kg", (GRID_LEFT, GRID_TOP + CELL_HEIGHT, 0.6, CELL_HEIGHT)),
        cell("beta 2 m", (GRID_LEFT, GRID_TOP + 2 * CELL_HEIGHT, 0.6, CELL_HEIGHT)),
    ],
    "whole_table_box": [*GRID_CELLS, WHOLE_TABLE_CELL],
}

ON_A_BOXED_NEUTRAL_ELEMENT: Group = {
    "neutral element left out": ([], 1),
    "item on it": ([item(TABLE, OTHER_TEXT)], 1),
    "body item on it": ([item(TABLE, OTHER_TEXT, channel="body")], 1),
    "item without a box": ([item(None, OTHER_TEXT)], 1),
    "item on it with other text": ([item(TABLE, FIVE_WORDS)], ONE_EXTRA),
    "item without a box with other text": ([item(None, FIVE_WORDS)], ONE_EXTRA),
}
ON_AN_ELEMENT_WITHOUT_A_BOX: Group = {
    "element without a box left out": ([], 1),
    "item without a box with its text": ([item(None, OTHER_TEXT)], 1),
    "item on a box with its text": ([item(TABLE, OTHER_TEXT)], 1),
    "item without a box with other text": ([item(None, FIVE_WORDS)], ONE_EXTRA),
}
ON_A_NEUTRAL_CAPTION: Group = {
    "paragraph item on it": ([item(TABLE, OTHER_TEXT)], 1),
    "paragraph item without a box": ([item(None, OTHER_TEXT)], 1),
    "item on it with the wrong text": ([item(TABLE, STRAY_TEXT)], ONE_EXTRA),
}
NEAR_NEUTRAL = {
    "item on the near neutral box": (SCORED_PARAGRAPH, NARROWER_COPY, [NARROWER_ITEM]),
    "neutral copy on the same box": (SCORED_PARAGRAPH, annotated(PARAGRAPH), [PARAGRAPH_ITEM]),
    "scored element takes its best item": (
        boxed("p", "paragraph", PARAGRAPH, FIVE_WORDS),
        annotated(NARROWER_PARAGRAPH, "one two three four five six seven eight nine ten"),
        [item(PARAGRAPH, FIVE_WORDS), item(NARROWER_PARAGRAPH, SEVEN_WORDS)],
    ),
    "neutral copy without a box": (
        SCORED_PARAGRAPH,
        element("paragraph", None, PARAGRAPH_TEXT, "hidden"),
        [PARAGRAPH_ITEM],
    ),
}

BOTH_ELEMENT_METRICS: dict[str, tuple[Groundtruth, list[LayoutItem], float, float]] = {
    "every item a paragraph": (FOUR_ELEMENTS, ALL_PARAGRAPHS, 1, f1(2, 4, 4)),
    "stray code item": (FULL_DOCUMENT, [*FULL_ITEMS, stray("code")], f1(5, 5, 6), f1(5, 5, 6)),
    "stray paragraph item": (
        FULL_DOCUMENT,
        [*FULL_ITEMS, stray("paragraph")],
        f1(5, 5, 6),
        f1(5, 5, 6),
    ),
    "neutral copy without a box": (
        with_paragraph(annotated(None)),
        CAPTION_AND_NARROWER,
        1,
        ONE_EXTRA,
    ),
    "neutral copy on the box": (
        with_paragraph(annotated(PARAGRAPH)),
        CAPTION_AND_NARROWER,
        1,
        ONE_EXTRA,
    ),
    **{
        f"{cls} item on a shifted hidden copy": (
            HEADING_WITH_SHIFTED_COPY,
            [HEADING_ITEM, item(SHIFTED_HEADING, HEADING_TEXT, cls)],
            1,
            classes,
        )
        for cls, classes in [("paragraph", 1), ("heading", 1), ("caption", ONE_EXTRA)]
    },
}
BOXES_ON_ONE_PARAGRAPH: Group = {
    "right text with the wrong box": ([item(resized(PARAGRAPH, 1, shift=0.5))], 0),
    "right box without item text": ([item(PARAGRAPH, "")], 0),
    "two items on one element": ([PARAGRAPH_ITEM, SHIFTED_ITEM], ONE_EXTRA),
    "item without a box": ([item(None)], 0),
    "item on a page the document lacks": ([PARAGRAPH_ITEM, item(PARAGRAPH, page=5)], ONE_EXTRA),
}
BOX_CASES: dict[str, Case] = {
    "two pairs, one miss, one extra": (
        THREE_ELEMENTS,
        [*THREE_ITEMS[:2], stray("paragraph")],
        f1(2, 3, 3),
    ),
    "figure item and text layer item": (
        FIGURE_WITH_TEXT_LAYER,
        [TALLER_FIGURE_ITEM, item(TEXT_LAYER, TEXT_LAYER_WORDS)],
        1,
    ),
    "items moved": (
        THREE_PAGES,
        [item(PARAGRAPH, page=1), PARAGRAPH_ITEM, item(TABLE, OTHER_TEXT)],
        f1(2, 3, 3),
    ),
    "figure without words, item text": (
        only("figure", FIGURE, ""),
        [item(FIGURE, "Chart of revenue", "figure")],
        1,
    ),
    "chart with words, box alone": (
        only("chart", FIGURE, "Revenue by quarter"),
        [item(FIGURE, "Quarter Revenue Q1 10", "chart")],
        1,
    ),
    "iou just under one half": (only("paragraph", SQUARE), [item((0.25, 0.25, 0.2499, 0.5))], 0),
    "iou of one half, uneven": (
        only("paragraph", (0.1, 0.2, 0.3, 0.3)),
        [item((0.1, 0.2, 0.15, 0.3))],
        1,
    ),
    "half the words wrong": (
        only("paragraph", PARAGRAPH, "one two three four"),
        [item(PARAGRAPH, HALF_OF_FOUR_WORDS)],
        0,
    ),
    "word f1 of four fifths, uneven": (
        only("paragraph", PARAGRAPH, SEVEN_WORDS),
        [item(PARAGRAPH, "one two three four five six eight nine")],
        1,
    ),
    "upper case printed": (
        only("heading", HEADING, "QUARTERLY REPORT", level=1),
        [item(HEADING, "Quarterly Report", "heading")],
        1,
    ),
    "table text with its cells": (
        TITLED_TABLE,
        [item(TABLE, f"{TABLE_TITLE} {TABLE_CELLS_TEXT}", "table")],
        1,
    ),
    "table item without the title": (TITLED_TABLE, [item(TABLE, TABLE_CELLS_TEXT, "table")], 0),
    "title with its heading words": (
        only("table", TABLE, REVENUE_TITLE, header=[REVENUE_TITLE], rows=[["5"]]),
        [item(TABLE, f"{REVENUE_TITLE} {REVENUE_TITLE} 5", "table")],
        1,
    ),
    "title sharing words with the cells": (
        only("table", TABLE, SHARED_TITLE, header=SHARED_HEADER, rows=[]),
        [item(TABLE, f"{SHARED_TITLE} {' '.join(SHARED_HEADER)}", "table")],
        1,
    ),
    "boxes under float rounding": (only("figure", SLIVER, ""), [item(SLIVER, "", "figure")], 1),
    "exact ties": (
        PARAGRAPH_WITH_HIDDEN_COPY,
        [PARAGRAPH_ITEM, WRONG_HEADING, item(PARAGRAPH, cls="caption"), item(None)],
        f1(1, 1, 3),
    ),
    **per_channel("annotation", "paragraph", TABLE, ON_A_BOXED_NEUTRAL_ELEMENT),
    **per_channel("body", "paragraph", None, ON_AN_ELEMENT_WITHOUT_A_BOX),
    "unboxed item for a neutral chart with a box": (
        with_paragraph(element("chart", FIGURE, FIVE_WORDS, "hidden")),
        [PARAGRAPH_ITEM, item(None, FIVE_WORDS, "chart")],
        1,
    ),
    "neutral figure without box or words": (
        with_paragraph(element("figure", None, channel="hidden")),
        [PARAGRAPH_ITEM, item(None, "", "figure")],
        ONE_EXTRA,
    ),
    **in_both_orders(NEAR_NEUTRAL),
}

CLASSES_ON_HEADING_AND_PARAGRAPH: Group = {
    "mixed labels": (
        [PARAGRAPH_ITEM, WRONG_HEADING, HEADING_ITEM, item(HEADING, HEADING_TEXT), UNBOXED_HEADING],
        f1(2, 2, 5),
    ),
    "right class on the wrong page": ([HEADING_ITEM, item(PARAGRAPH, page=1)], f1(1, 2, 2)),
    "wrong class with a closer box": (
        [
            item(resized(HEADING, 0.9), HEADING_TEXT, "heading"),
            item(HEADING, HEADING_TEXT),
            PARAGRAPH_ITEM,
        ],
        f1(2, 2, 3),
    ),
    "wrong item labelled heading": ([HEADING_AS_ITEM, WRONG_HEADING], f1(1, 2, 2)),
    "wrong item labelled formula": (
        [HEADING_AS_ITEM, relabelled(WRONG_HEADING, "formula")],
        f1(1, 2, 2),
    ),
}
CLASSES_ON_A_HIDDEN_COPY: Group = {
    "wrong label heading": ([PARAGRAPH_ITEM, WRONG_HEADING], ONE_EXTRA),
    "right label": ([item(PARAGRAPH), item(PARAGRAPH)], 1),
}
CLASSES_ON_TWO_PAGES: Group = {
    "footers swapped": (
        [
            *items_without(TWO_PAGES_WITH_FOOTERS, "f", "g"),
            item(FOOTER, "1", "footer", page=1),
            item(FOOTER, "2", "footer"),
        ],
        f1(2, 4, 4),
    ),
}
CLASS_CASES: dict[str, Case] = {
    **per_channel("hidden", "caption", TABLE, ON_A_NEUTRAL_CAPTION),
    "every element twice": (
        HEADING_AND_TWO_PARAGRAPHS,
        twice(truth_as_items(HEADING_AND_TWO_PARAGRAPHS)),
        2 / 3,
    ),
    "caption item for the text": (
        FIGURE_WITH_HIDDEN_TEXT,
        [FIGURE_ITEM, item(FIGURE, TEXT_LAYER_WORDS, "caption")],
        1,
    ),
    **{
        f"caption item with the heading words, {copy}, {where}": (
            t,
            [PARAGRAPH_ITEM, item(box, HEADING_TEXT, "caption")],
            f1(1, 2, 2),
        )
        for copy, t in HIDDEN_HEADING_COPIES.items()
        for where, box in CAPTION_BOXES.items()
    },
}

ON_A_NEUTRAL_TABLE: Group = {
    "cells on the body item": (grid_plus(*LOWER_CELLS), 1),
    "cells without a box": (grid_plus(*unboxed(LOWER_CELLS)), 1),
    "strangers without a box": (grid_plus(*STRANGERS), CELLS_TWICE),
    "cells far away": (grid_plus(*FAR_LOWER_CELLS), CELLS_TWICE),
}
ON_A_NEUTRAL_TABLE_WITHOUT_CELLS: Group = {
    "cells inside it": (grid_plus(*LOWER_CELLS), 1),
    "cells without a box": (grid_plus(*unboxed(LOWER_CELLS)), 1),
    "cells outside it": (grid_plus(*FAR_LOWER_CELLS), CELLS_TWICE),
}
UNDER_A_NEUTRAL_TABLE_WITHOUT_CELLS: Group = {
    "body cells": (grid_plus(), 1),
    "copies": (grid_plus(*TIGHT_CELLS), 1),
    "junk": (grid_plus(*JUNK), CELLS_TWICE),
    "whole table cell and copies": (grid_plus(WHOLE_TABLE_CELL, *TIGHT_CELLS), ONE_CELL_EXTRA),
}
WORDS_TAKEN_ONCE: Group = {
    "two copies of every cell": (grid_plus(*TIGHT_CELLS, *TIGHT_CELLS), CELLS_TWICE),
    "two unboxed copies of every cell": (grid_plus(*unboxed(GRID_CELLS) * 2), CELLS_TWICE),
    "three copies of one cell": (grid_plus(TIGHT_ALPHA, TIGHT_ALPHA, TIGHT_ALPHA), f1(9, 9, 11)),
}
NEUTRAL_CELLS: dict[str, tuple[Callable[[str], Groundtruth], list[LayoutItem], float]] = {
    "body cell before a closer neutral cell": (
        partial(beside_cell, "alpha", (0.12, 0.38, 0.4, 0.3)),
        moved(ALPHA, (0.14, 0.4, 0.2, 0.1)),
        1,
    ),
    "body cell over half a neutral cell": (
        partial(beside_cell, "note", (0.7, 0.4, 0.1, 0.1)),
        moved(KG, KG_OVER_THE_NEUTRAL_CELL),
        ONE_CELL_LOST,
    ),
    "body cell over a cell one percent in": (
        partial(beside_cell, "note", (0.698, 0.4, 0.2, 0.1)),
        moved(KG, KG_OVER_HALF_OF_THE_NOTE),
        ONE_CELL_LOST,
    ),
    "neutral cell over half a body cell": (
        partial(beside_cell, "note", (0.7, 0.4, 0.2, 0.1)),
        grid_plus(cell("note", (0.6, 0.4, 0.3, 0.1))),
        ONE_CELL_EXTRA,
    ),
    "neutral copy of the body table": (NEUTRAL_COPY, grid_plus(), 1),
}
NEUTRAL_TABLE_GROUPS: list[tuple[str, Callable[[str], Groundtruth], Group, str]] = [
    ("neutral table", NEUTRAL_TABLE, ON_A_NEUTRAL_TABLE, "annotation"),
    ("table without cells", WITHOUT_CELLS, ON_A_NEUTRAL_TABLE_WITHOUT_CELLS, "hidden"),
    ("under a table without cells", OVER_THE_BODY, UNDER_A_NEUTRAL_TABLE_WITHOUT_CELLS, "hidden"),
    *(
        (f"table without cells {where}", build, WORDS_TAKEN_ONCE, "hidden")
        for where, build in NEUTRAL_TABLES_WITH_THE_BODY_WORDS.items()
    ),
]
ON_THE_GRID: Group = {
    "tight boxes": (on_table(TIGHT_CELLS), 1),
    "box in the neighbour cell": (moved(ALPHA, tight(grid_box(1, 1))), ONE_CELL_LOST),
    "share inside of one half at the bottom": (moved(LAST, (0.5, 0.55, 0.2, 0.1)), 1),
    "share just under one half at the edge": (moved(NAME, (0.0, 0.3, 0.1998, 0.1)), ONE_CELL_LOST),
    "every cell twice": (on_table(GRID_CELLS * 2), CELLS_TWICE),
    "cell without a box": (moved(ALPHA, None), ONE_CELL_LOST),
    "extra cells '—'": (grid_plus(cell("—", (0.8, 0.8, 0.1, 0.1)), cell("—", None)), 1),
    "table item with a wrong box": ([table_item(GRID_CELLS, (0.0, 0.0, 0.05, 0.05))], 1),
    "extra cells on a second item": (
        [GRID_ITEM, table_item([cell("gamma", (0.1, 0.7, 0.2, 0.1))], (0.1, 0.7, 0.2, 0.1))],
        ONE_CELL_EXTRA,
    ),
    "table item without cells": ([TABLE_ITEM_WITHOUT_CELLS], 0),
    "box mostly inside a smaller cell": (moved(ONE, (0.28, 0.39, 0.26, 0.12)), 1),
    "box mostly outside its cell": (moved(ONE, (0.25, 0.38, 0.35, 0.14)), ONE_CELL_LOST),
    "box twice the cell, over a neighbour": (moved(ONE, (0.3, 0.4, 0.4, 0.1)), ONE_CELL_LOST),
    **{
        f"box over half a neighbour, {box}": (moved(ONE, box), ONE_CELL_LOST)
        for box in [(0.3, 0.4, 0.3, 0.1), (0.3, 0.4, 0.2, 0.15)]
    },
    **{
        f"box over under half a neighbour, {box}": (moved(ONE, box), 1)
        for box in [(0.3, 0.4, 0.2998, 0.1), (0.3, 0.4, 0.2, 0.1499)]
    },
    "kg box where a neutral cell would be": (moved(KG, KG_OVER_THE_NEUTRAL_CELL), 1),
    "kg box over half of a note's place": (moved(KG, KG_OVER_HALF_OF_THE_NOTE), 1),
}
ON_SAME_PAGE_TABLES: Group = {
    "swapped items": (
        [table_item(LOWER_CELLS, AREA, id_="a"), table_item(GRID_CELLS, LOWER_AREA, id_="b")],
        1,
    ),
    "second table texts in the first table boxes": (
        [GRID_ITEM, table_item(grid_cells(OTHER_GRID), LOWER_AREA, id_="b")],
        f1(9, 18, 18),
    ),
}
ON_A_MERGED_CELL: Group = {
    "text in the first column": (merged_row("All regions", ""), 1),
    "text in both columns": (merged_row("All regions", "All regions"), f1(5, 5, 6)),
    "text split over two columns": (merged_row("All", "regions"), f1(4, 5, 6)),
}
NOTE_TABLE = truth(grid_table("t", [["Note"], [FIVE_WORDS]]))
NEXT_PAGE_PARAGRAPH = beside_grid(boxed("p", "paragraph", PARAGRAPH, PARAGRAPH_TEXT, page=1))
ZEROS_TABLE = truth(grid_table("t", ZEROS))
CELL_CASES: dict[str, Case] = {
    "mix of pairs, misses and extras": (
        truth(grid_table("t", SMALL_GRID)),
        on_table([*grid_cells(SMALL_GRID)[:3], cell("delta", grid_box(1, 1))]),
        f1(3, 4, 4),
    ),
    "word f1 of four fifths": (
        NOTE_TABLE,
        on_table([NOTE_CELL, cell(FOUR_OF_FIVE_WORDS, grid_box(1, 0))]),
        1,
    ),
    "word f1 of one half": (
        NOTE_TABLE,
        on_table([NOTE_CELL, cell(HALF_OF_FOUR_WORDS, grid_box(1, 0))]),
        f1(1, 2, 2),
    ),
    "right box and text on the wrong page": (NEXT_PAGE_PARAGRAPH, on_table(GRID_CELLS, page=1), 0),
    "cells copied onto another page": (
        NEXT_PAGE_PARAGRAPH,
        [GRID_ITEM, table_item(GRID_CELLS, AREA, 1)],
        CELLS_TWICE,
    ),
    **alpha_value_cases("—"),
    "same text cells": (ZEROS_TABLE, on_table(TIGHT_ZEROS), 1),
    "same text, one box over both": (
        truth(grid_table("t", [["0", "0"]])),
        on_table([cell("0", (0.25, 0.32, 0.1, 0.05)), cell("0", tight(grid_box(0, 0)))]),
        1,
    ),
    "row boxes": (two_column_table(0.36, 0.24), row_box_items(two_column_table(0.36, 0.24)), 0),
    "row box over an empty cell": (
        EMPTY_NEIGHBOUR,
        on_table([*EMPTY_NEIGHBOUR_CELLS[:2], cell("alpha", row_box([WIDE, EMPTY]))]),
        f1(2, 3, 3),
    ),
    "body table without a box": (UNBOXED_BODY_TABLE, grid_plus(), 1),
    "body table without box or cells": (
        beside_grid(table_of([["City"], ["gamma"]], None)),
        grid_plus(cell("gamma", None)),
        1,
    ),
    **{
        f"{name}, hidden": (build("hidden"), items, expected)
        for name, (build, items, expected) in NEUTRAL_CELLS.items()
    },
    **{
        f"{label}, {name}, {ch}": (build(ch), items, expected)
        for label, build, group, ch in NEUTRAL_TABLE_GROUPS
        for name, (items, expected) in group.items()
    },
}

SCORES, SCORE_IDS = cases(
    (BOXES, None, {name: (t, i, b) for name, (t, i, b, _) in BOTH_ELEMENT_METRICS.items()}),
    (BOXES, None, BOX_CASES),
    (BOXES, ONE_PARAGRAPH, BOXES_ON_ONE_PARAGRAPH),
    (CLASSES, None, {name: (t, i, c) for name, (t, i, _, c) in BOTH_ELEMENT_METRICS.items()}),
    (CLASSES, None, CLASS_CASES),
    (CLASSES, HEADING_AND_PARAGRAPH, CLASSES_ON_HEADING_AND_PARAGRAPH),
    (CLASSES, PARAGRAPH_WITH_HIDDEN_COPY, CLASSES_ON_A_HIDDEN_COPY),
    (CLASSES, TWO_PAGES_WITH_FOOTERS, CLASSES_ON_TWO_PAGES),
    (CELLS, None, CELL_CASES),
    (CELLS, CELL_TABLE, ON_THE_GRID),
    (CELLS, SAME_PAGE_TABLES, ON_SAME_PAGE_TABLES),
    (CELLS, MERGED_TABLE, ON_A_MERGED_CELL),
)

OWN_PREDICTIONS: dict[str, tuple[SubMetric, Groundtruth]] = {
    "element_boxes: full document": (BOXES, FULL_DOCUMENT),
    "element_boxes: three pages": (BOXES, THREE_PAGES),
    "element_boxes: hidden 'Confidential draft' on a figure": (
        BOXES,
        truth(
            boxed("g", "figure", FIGURE, "Sales by region"),
            TEXT_LAYER_OVER(FIGURE, text="Confidential draft"),
        ),
    ),
    "element_classes: full document": (CLASSES, FULL_DOCUMENT),
    "element_classes: hidden text over a figure": (CLASSES, FIGURE_WITH_HIDDEN_TEXT),
    **{
        f"element_classes: {copy} of a heading": (CLASSES, t)
        for copy, t in HIDDEN_HEADING_COPIES.items()
    },
    "table_cell_boxes: cell document": (CELLS, CELL_DOCUMENT),
    "table_cell_boxes: two tables on one page": (CELLS, SAME_PAGE_TABLES),
    "table_cell_boxes: neutral table, annotation": (CELLS, NEUTRAL_TABLE("annotation")),
    "table_cell_boxes: neutral copy, hidden": (CELLS, NEUTRAL_COPY("hidden")),
}

UNBOXED_PARAGRAPH = element("paragraph", None, PARAGRAPH_TEXT)
NEUTRAL_PARAGRAPH = partial(
    boxed, "a", "paragraph", PARAGRAPH, PARAGRAPH_TEXT, channel="annotation"
)
APPLIES: dict[str, tuple[SubMetric, Groundtruth, bool]] = {
    **{
        f"{m.name}: one boxed furniture element": (
            m,
            truth(boxed("p", "paragraph", PARAGRAPH, PARAGRAPH_TEXT, channel="furniture")),
            True,
        )
        for m in [BOXES, CLASSES]
    },
    "element_boxes: every boxed element is neutral": (
        BOXES,
        truth(
            NEUTRAL_PARAGRAPH(),
            boxed("h", "paragraph", TABLE, OTHER_TEXT, channel="hidden"),
            boxed("m", "paragraph", FIGURE, FIVE_WORDS, channel="metadata"),
            UNBOXED_PARAGRAPH,
        ),
        False,
    ),
    "element_classes: every boxed element is neutral": (
        CLASSES,
        truth(
            NEUTRAL_PARAGRAPH(),
            boxed("h", "heading", HEADING, HEADING_TEXT, channel="hidden", level=1),
            boxed("m", "figure", FIGURE, channel="metadata"),
            UNBOXED_PARAGRAPH,
        ),
        False,
    ),
    "table_cell_boxes: no table": (CELLS, ONE_PARAGRAPH, False),
    "table_cell_boxes: one of two tables has no cells": (
        CELLS,
        beside_grid(boxed("u", "table", TABLE, page=1, header=["Name"], rows=[["gamma"]])),
        False,
    ),
    "table_cell_boxes: every cell is empty": (
        CELLS,
        truth(grid_table("t", [["", ""], ["", " "]])),
        False,
    ),
    "table_cell_boxes: the only table has no box": (
        CELLS,
        truth({k: v for k, v in GRID_TABLE.items() if k != "box"}),
        False,
    ),
    "table_cell_boxes: the only table is hidden": (
        CELLS,
        truth(grid_table("t", GRID, channel="hidden")),
        False,
    ),
}

DOCUMENT_MARKDOWN = f"# {HEADING_TEXT}\n\n{PARAGRAPH_TEXT}"
EMPTY_LAYOUTS = {
    "element_boxes": (BOXES, FULL_DOCUMENT, DOCUMENT_MARKDOWN),
    "element_classes": (CLASSES, FULL_DOCUMENT, DOCUMENT_MARKDOWN),
    "table_cell_boxes": (CELLS, CELL_TABLE, ""),
}

TRACK_CASES = {
    "groundtruth without cells": (FULL_DOCUMENT, FULL_ITEMS, (1.0, 1.0, None), 1.0),
    "every item a paragraph": (
        FOUR_ELEMENTS,
        ALL_PARAGRAPHS,
        (1.0, 0.5, None),
        (1.0 + 0.5 * 0.5) / 1.5,
    ),
    "table item without cells": (
        CELL_DOCUMENT,
        document_with_table(TABLE_ITEM_WITHOUT_CELLS),
        (1.0, 1.0, 0.0),
        (1.0 * 1.0 + 0.5 * 1.0 + 0.5 * 0.0) / (1.0 + 0.5 + 0.5),
    ),
}


@pytest.mark.parametrize(("metric", "t", "items", "expected"), SCORES, ids=SCORE_IDS)
def test_score_is_the_hand_count_in_either_json_order(
    metric: SubMetric, t: Groundtruth, items: list[LayoutItem], expected: float
) -> None:
    assert metric.applies(t)
    assert score(metric, t, items) == score(metric, t, flipped(items)) == pytest.approx(expected)


@pytest.mark.parametrize(("metric", "t"), OWN_PREDICTIONS.values(), ids=OWN_PREDICTIONS)
def test_groundtruth_as_prediction_scores_one(metric: SubMetric, t: Groundtruth) -> None:
    assert metric.applies(t)
    assert score(metric, t, truth_as_items(t)) == 1.0


@pytest.mark.parametrize(("metric", "t", "expected"), APPLIES.values(), ids=APPLIES)
def test_applies_reads_the_groundtruth(metric: SubMetric, t: Groundtruth, expected: bool) -> None:
    assert metric.applies(t) is expected


@pytest.mark.parametrize(("metric", "t", "markdown"), EMPTY_LAYOUTS.values(), ids=EMPTY_LAYOUTS)
def test_an_empty_layout_scores_zero(metric: SubMetric, t: Groundtruth, markdown: str) -> None:
    assert score(metric, t, Prediction(markdown=markdown)) == 0.0


@pytest.mark.parametrize("metric", SUB_METRICS, ids=[m.name for m in SUB_METRICS])
def test_the_demo_groundtruth_and_nutrient_output_score_one_after_a_json_round_trip(
    metric: SubMetric, demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    result = TRACK.evaluate(demo_groundtruth, DocContext(demo_groundtruth, pred))
    assert metric.applies(demo_groundtruth)
    assert score(metric, demo_groundtruth, truth_as_items(demo_groundtruth)) == 1.0
    assert score(metric, *round_trip(demo_groundtruth, pred)) == 1.0
    assert score(metric, demo_groundtruth, pred) == result.sub_scores[metric.name] == 1.0
    assert result.not_implemented == ()


@pytest.mark.parametrize(
    ("t", "items", "sub_scores", "expected"), TRACK_CASES.values(), ids=TRACK_CASES
)
def test_the_track_weighs_the_sub_metrics_that_apply(
    t: Groundtruth, items: list[LayoutItem], sub_scores: tuple[float | None, ...], expected: float
) -> None:
    result = TRACK.evaluate(t, DocContext(t, prediction(layout=items)))
    assert result.not_implemented == ()
    assert result.sub_scores == dict(zip([m.name for m in SUB_METRICS], sub_scores, strict=True))
    assert result.score == pytest.approx(expected)


def test_one_box_for_the_whole_page_with_all_the_text_scores_zero() -> None:
    all_text = " ".join(e.text for e in THREE_ELEMENTS.elements)
    assert score(BOXES, THREE_ELEMENTS, [item((0.0, 0.0, 1.0, 1.0), all_text)]) == 0.0


def test_a_box_three_times_the_element_area_with_the_right_text_does_not_pair() -> None:
    assert (
        score(BOXES, only("paragraph", (0.1, 0.1, 0.2, 0.2)), [item((0.1, 0.1, 0.6, 0.2))]) == 0.0
    )


def test_a_missing_element_lowers_the_recall_of_its_class() -> None:
    items = items_without(TWO_HEADINGS_TWO_PARAGRAPHS, "k")
    assert score(CLASSES, TWO_HEADINGS_TWO_PARAGRAPHS, items) == pytest.approx(f1(3, 4, 3))


def test_a_missing_element_of_a_rare_class_costs_one_element() -> None:
    items = items_without(FIVE_PARAGRAPHS_AND_A_CAPTION, "c")
    expected = f1(PARAGRAPH_COUNT, PARAGRAPH_COUNT + 1, PARAGRAPH_COUNT)
    assert score(CLASSES, FIVE_PARAGRAPHS_AND_A_CAPTION, items) == pytest.approx(expected)
    assert score(BOXES, FIVE_PARAGRAPHS_AND_A_CAPTION, items) == pytest.approx(expected)


def test_a_caption_labelled_paragraph_on_the_demo_costs_one_class_pair(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    layout = [relabelled(i, "paragraph") if i.cls == "caption" else i for i in pred.layout]
    assert score(CLASSES, demo_groundtruth, layout) == pytest.approx(f1(9, 10, 10))


def test_box_across_the_whole_row_with_one_cell_text_does_not_pair() -> None:
    assert score(CELLS, CELL_TABLE, moved(ALPHA, (0.1, 0.4, 0.6, 0.1))) == pytest.approx(
        ONE_CELL_LOST
    )


def test_one_cell_for_the_whole_table_with_all_the_text_scores_zero() -> None:
    assert score(CELLS, CELL_TABLE, on_table([WHOLE_TABLE_CELL])) == 0.0


@pytest.mark.parametrize(
    "neutral_table",
    NEUTRAL_TABLES_WITH_THE_BODY_WORDS.values(),
    ids=NEUTRAL_TABLES_WITH_THE_BODY_WORDS,
)
@pytest.mark.parametrize("cells", WRONG_BODY_CELLS.values(), ids=WRONG_BODY_CELLS)
def test_wrong_cells_on_the_body_table_count_as_they_do_without_a_neutral_table(
    neutral_table: Callable[[str], Groundtruth], cells: Cells
) -> None:
    without_neutral_table = score(CELLS, CELL_TABLE, on_table(cells))
    assert without_neutral_table < 1.0
    assert score(CELLS, neutral_table("hidden"), on_table(cells)) == pytest.approx(
        without_neutral_table
    )


def test_json_round_trip_keeps_the_cells_and_the_score() -> None:
    pred = prediction(layout=document_with_table(table_item(cells_with(ALPHA, None), AREA)))
    t, p = round_trip(CELL_DOCUMENT, pred)
    assert (t, p) == (CELL_DOCUMENT, pred)
    assert score(CELLS, t, p) == score(CELLS, CELL_DOCUMENT, pred) == pytest.approx(ONE_CELL_LOST)


def test_row_boxes_with_the_right_text_on_the_demo_tables_score_zero(
    demo_groundtruth: Groundtruth,
) -> None:
    assert score(CELLS, demo_groundtruth, row_box_items(demo_groundtruth)) == 0.0
