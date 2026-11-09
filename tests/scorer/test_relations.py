from typing import Any

import pytest

from tests.builders import para, truth
from wholedoc_bench.schema import Box, Groundtruth, LayoutItem, Prediction, Relation
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t3_structure import Relations
from wholedoc_bench.scorer.utils.boxes import box_iou

METRIC = Relations()

FIGURE: Box = (0.1, 0.1, 0.8, 0.3)
CAPTION: Box = (0.3, 0.42, 0.4, 0.04)
FIGURE_2: Box = (0.1, 0.5, 0.8, 0.3)
CAPTION_2: Box = (0.3, 0.82, 0.4, 0.04)


def element(id_: str, cls: str, box: Box | None, text: str = "", page: int = 0) -> dict[str, Any]:
    return {"id": id_, "class": cls, "page": page, "box": box, "text": text}


def link(kind: str, source: str, target: str) -> dict[str, str]:
    return {"kind": kind, "from": source, "to": target}


def item(id_: str, cls: str, box: Box | None, text: str = "", page: int = 0) -> LayoutItem:
    return LayoutItem.model_validate(
        {"id": id_, "class": cls, "page": page, "box": box, "text": text}
    )


def prediction(items: list[LayoutItem], links: list[dict[str, str]]) -> Prediction:
    relations = [Relation.model_validate(r) for r in links]
    return Prediction(markdown="", layout=items, relations=relations)


def score(t: Groundtruth, items: list[LayoutItem], links: list[dict[str, str]]) -> float:
    return METRIC.score(DocContext(t, prediction(items, links)))


def shifted(box: Box, share: float) -> Box:
    x, y, w, h = box
    return (x + w * share, y, w, h)


ONE = truth(
    element("f", "figure", FIGURE),
    element("c", "caption", CAPTION, "Figure 1: Revenue by quarter"),
    relations=[link("caption_of", "c", "f")],
)
TWO = truth(
    element("f1", "figure", FIGURE),
    element("c1", "caption", CAPTION, "Figure 1: Revenue"),
    element("f2", "figure", FIGURE_2),
    element("c2", "caption", CAPTION_2, "Figure 2: Costs"),
    relations=[link("caption_of", "c1", "f1"), link("caption_of", "c2", "f2")],
)
BOTH_FIGURES = [
    item("x1", "figure", FIGURE),
    item("y1", "caption", CAPTION, "Figure 1: Revenue"),
    item("x2", "figure", FIGURE_2),
    item("y2", "caption", CAPTION_2, "Figure 2: Costs"),
]


def test_captions_linked_to_the_wrong_figures_score_zero() -> None:
    links = [link("caption_of", "y1", "x2"), link("caption_of", "y2", "x1")]
    assert score(TWO, BOTH_FIGURES, links) == 0.0


def test_one_link_of_two() -> None:
    assert score(TWO, BOTH_FIGURES, [link("caption_of", "y1", "x1")]) == pytest.approx(2 / 3)


def test_a_markdown_only_parser_scores_zero() -> None:
    ctx = DocContext(ONE, Prediction(markdown="Figure 1: Revenue by quarter"))
    assert METRIC.score(ctx) == 0.0


@pytest.mark.parametrize(
    ("share", "text", "expected"),
    [
        (0.375, "", 0.0),
        (0.375, "Figure 1: Revenue by quarter", 1.0),
    ],
    ids=["IoU 0.45, no text", "IoU 0.45, its text"],
)
def test_a_box_that_overlaps_too_little_is_found_by_its_text(
    share: float, text: str, expected: float
) -> None:
    caption = shifted(CAPTION, share)
    assert box_iou(CAPTION, caption) == pytest.approx((1 - share) / (1 + share))
    items = [item("x", "figure", FIGURE), item("y", "caption", caption, text)]
    assert score(ONE, items, [link("caption_of", "y", "x")]) == expected


def test_an_item_without_a_box_pairs_by_its_text() -> None:
    items = [
        item("x", "figure", FIGURE),
        item("y", "caption", None, "Figure 1: Revenue by quarter"),
    ]
    assert score(ONE, items, [link("caption_of", "y", "x")]) == 1.0


TABLE_NO_BOX = truth(
    element("c", "caption", None, "Table 1: Annual allowances"),
    {
        **element("t", "table", None),
        "header": ["Item", "Allowance (EUR)"],
        "rows": [["Home internet", "360"], ["Office chair", "250"]],
    },
    relations=[link("caption_of", "c", "t")],
)


@pytest.mark.parametrize(
    ("table_text", "expected"),
    [
        ("Item Allowance (EUR) Home internet 360 Office chair 250", 1.0),
        ("Item Allowance (EUR) Home internet 360", 0.0),
    ],
    ids=["its cells", "half its rows"],
)
def test_a_table_without_a_box_pairs_by_its_cells(table_text: str, expected: float) -> None:
    items = [
        item("y", "caption", CAPTION, "Table 1: Annual allowances"),
        item("x", "table", FIGURE, table_text),
    ]
    assert score(TABLE_NO_BOX, items, [link("caption_of", "y", "x")]) == expected


def test_text_that_points_to_two_elements_pairs_with_neither() -> None:
    t = truth(
        element("f1", "figure", FIGURE),
        element("c1", "caption", CAPTION, "Source: company data"),
        element("f2", "figure", FIGURE_2),
        element("c2", "caption", CAPTION_2, "Source: company data"),
        relations=[link("caption_of", "c1", "f1"), link("caption_of", "c2", "f2")],
    )
    items = [
        item("x1", "figure", FIGURE),
        item("x2", "figure", FIGURE_2),
        item("y", "caption", None, "Source: company data"),
    ]
    assert score(t, items, [link("caption_of", "y", "x1")]) == 0.0


EDGE_FIGURE: Box = (0.125, 0.125, 0.25, 0.25)
EDGE_CAPTION: Box = (0.5, 0.5, 0.25, 0.125)


@pytest.mark.parametrize(
    ("text_box", "reverse"),
    [((0.125, 0.0, 0.125, 0.125), False), (None, True)],
    ids=["box, as listed", "no box, reversed"],
)
def test_a_box_match_at_exactly_half_beats_a_text_match(
    text_box: Box | None, reverse: bool
) -> None:
    t = truth(
        element("f", "figure", EDGE_FIGURE),
        element("c", "caption", EDGE_CAPTION, "Figure 1: Revenue by quarter"),
        relations=[link("caption_of", "c", "f")],
    )
    y_box: Box = (0.5, 0.5, 0.125, 0.125)
    assert box_iou(EDGE_CAPTION, y_box) == 0.5
    items = [
        item("x", "figure", EDGE_FIGURE),
        item("y", "caption", y_box, "Figure 1: Revenue by quarter"),
        item("z", "caption", text_box, "Figure 1: Revenue by quarter"),
    ]
    assert score(t, items[::-1] if reverse else items, [link("caption_of", "y", "x")]) == 1.0


@pytest.mark.parametrize(
    "wrong", [link("caption_of", "x", "y"), link("footnote_of", "y", "x")], ids=["reversed", "kind"]
)
def test_direction_and_kind_must_match(wrong: dict[str, str]) -> None:
    items = [item("x", "figure", FIGURE), item("y", "caption", CAPTION, "Figure 1")]
    assert score(ONE, items, [wrong]) == 0.0


def test_a_repeated_link_counts_once() -> None:
    items = [item("x", "figure", FIGURE), item("y", "caption", CAPTION, "Figure 1")]
    assert score(ONE, items, [link("caption_of", "y", "x")] * 2) == 1.0


def test_a_link_to_an_unknown_item_is_wrong() -> None:
    items = [item("x", "figure", FIGURE), item("y", "caption", CAPTION, "Figure 1")]
    assert score(ONE, items, [link("caption_of", "y", "nowhere")]) == 0.0


def test_items_that_cannot_be_told_apart_find_nothing() -> None:
    twins = [item(k, "caption", CAPTION, "Figure 1") for k in ("y", "z")]
    assert score(ONE, [item("x", "figure", FIGURE), *twins], [link("caption_of", "y", "x")]) == 0.0
    same_id = [item("y", "caption", CAPTION, "Figure 1"), item("y", "caption", CAPTION_2, "Other")]
    assert (
        score(ONE, [item("x", "figure", FIGURE), *same_id], [link("caption_of", "y", "x")]) == 0.0
    )


def test_the_order_of_items_in_the_json_does_not_change_the_score() -> None:
    figure_box: Box = (0.125, 0.0625, 0.75, 0.375)
    caption_box: Box = (0.25, 0.5, 0.5, 0.0625)
    t = truth(
        element("f", "figure", figure_box),
        element("c", "caption", caption_box, "Figure 1"),
        relations=[link("caption_of", "c", "f")],
    )
    left_box, right_box = shifted(caption_box, -0.25), shifted(caption_box, 0.25)
    assert box_iou(caption_box, left_box) == box_iou(caption_box, right_box)
    left = item("l", "caption", left_box, "Figure 1")
    right = item("r", "caption", right_box, "Figure 1")
    figure = item("x", "figure", figure_box)
    links = [link("caption_of", "l", "x"), link("footnote_of", "r", "x")]
    orders = ([figure, left, right], [right, figure, left], [left, right, figure])
    assert [score(t, order, links) for order in orders] == [pytest.approx(2 / 3)] * 3


def test_items_on_another_page_never_pair() -> None:
    t = truth(
        element("f", "figure", FIGURE),
        element("c", "caption", CAPTION, "Figure 1"),
        para("p", "Page two.", page=1),
        relations=[link("caption_of", "c", "f")],
    )
    items = [item("x", "figure", FIGURE, page=1), item("y", "caption", CAPTION, "Figure 1", page=1)]
    assert score(t, items, [link("caption_of", "y", "x")]) == 0.0


def test_applies_only_with_a_caption_or_footnote_link() -> None:
    assert METRIC.applies(ONE)
    unlinked = truth(element("f", "figure", FIGURE), element("c", "caption", CAPTION, "Figure 1"))
    assert not METRIC.applies(unlinked)


def test_a_box_one_float_step_under_one_half_iou_still_pairs() -> None:
    caption_box: Box = (0.2, 0.42, 0.15, 0.04)
    item_box: Box = (0.25, 0.42, 0.15, 0.04)
    assert box_iou(caption_box, item_box) < 0.5
    t = truth(
        element("f", "figure", FIGURE),
        element("c", "caption", caption_box, "Figure 1: Revenue by quarter"),
        relations=[link("caption_of", "c", "f")],
    )
    items = [item("x", "figure", FIGURE), item("y", "caption", item_box, "Fig. 1")]
    assert score(t, items, [link("caption_of", "y", "x")]) == 1.0


def test_relations_layout_text_is_plain_printed_content():
    figure = {"id": "f", "class": "figure", "page": 0, "text": "<b>x</b>"}
    caption = {"id": "c", "class": "caption", "page": 0, "text": "Caption"}
    doc = truth(figure, caption, relations=[{"kind": "caption_of", "from": "c", "to": "f"}])

    def score(text):
        pred = Prediction.model_validate(
            {
                "markdown": "",
                "layout": [
                    {"id": "x", "class": "figure", "page": 0, "text": text},
                    {"id": "y", "class": "caption", "page": 0, "text": "Caption"},
                ],
                "relations": [{"kind": "caption_of", "from": "y", "to": "x"}],
            }
        )
        return Relations().score(DocContext(doc, pred))

    assert score("<b>x</b>") == 1
    assert score("x") == 0


def test_a_table_with_a_merged_cell_pairs_by_its_cells_written_once() -> None:
    t = truth(
        element("c", "caption", None, "Table 1: Revenue"),
        {
            **element("t", "table", None),
            "header": ["Revenue", None],
            "rows": [["2025", "2026"], ["100", "120"]],
            "spans": [{"row": 0, "column": 0, "columns": 2}],
        },
        relations=[link("caption_of", "c", "t")],
    )
    items = [
        item("y", "caption", CAPTION, "Table 1: Revenue"),
        item("x", "table", FIGURE, "Revenue 2025 2026 100 120"),
    ]
    assert score(t, items, [link("caption_of", "y", "x")]) == 1.0
