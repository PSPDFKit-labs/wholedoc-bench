from typing import Any

import pytest

from tests.builders import boxed, heading, para, truth
from wholedoc_bench.corpus.checks import check_font_annotations
from wholedoc_bench.schema import Groundtruth, LayoutItem, Prediction
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.run import score_document
from wholedoc_bench.scorer.tracks.t3_structure import TRACK, Typography
from wholedoc_bench.scorer.typography import (
    LOOK_ALIKE_GENRES,
    family_key,
    same_family,
    same_size,
    score_typography,
)

METRIC = Typography()

TITLE = "Annual report"
BODY = (
    "Revenue grew in every region this year.",
    "Costs stayed flat against the plan.",
    "The board approved the new budget.",
)
SERIF = {"family": "Times New Roman", "class": "serif", "size": 10.0}
SANS = {"family": "Arial", "class": "sans", "size": 18.0}


def font(family: str | None = None, cls: str | None = None, size: float | None = None) -> dict:
    return {k: v for k, v in {"family": family, "class": cls, "size": size}.items() if v}


def report(**fonts: Any) -> Groundtruth:
    return truth(
        heading("h", TITLE, 1, font=fonts.get("heading", SANS)),
        *(para(f"p{i}", text, font=fonts.get("body", SERIF)) for i, text in enumerate(BODY)),
    )


def item(item_id: str, text: str, font_: dict | None = None, page: int = 0) -> LayoutItem:
    fields: dict[str, Any] = {"id": item_id, "class": "paragraph", "page": page, "text": text}
    if font_ is not None:
        fields["font"] = font_
    return LayoutItem.model_validate(fields)


def items(heading_font: dict | None = SANS, body_font: dict | None = SERIF) -> list[LayoutItem]:
    return [
        item("h", TITLE, heading_font),
        *(item(f"p{i}", text, body_font) for i, text in enumerate(BODY)),
    ]


def score(t: Groundtruth, layout: list[LayoutItem], markdown: str = "x") -> float:
    return METRIC.score(DocContext(t, Prediction(markdown=markdown, layout=layout)))


def test_groundtruth_as_prediction_scores_one() -> None:
    assert score(report(), items()) == 1.0


def test_applies_only_with_font_annotations() -> None:
    assert METRIC.applies(report())
    plain = truth(para("p", "Plain text."))
    assert not METRIC.applies(plain)
    assert "typography" in dict((m.name, w) for m, w in TRACK.parts)


def test_one_font_for_every_block_scores_half_on_each_part() -> None:
    assert score(report(), items(heading_font=SERIF)) == 0.5


def test_a_block_without_a_family_counts_as_wrong() -> None:
    result = score_typography(
        report(), items(heading_font=font(cls="sans", size=18.0)), check_font_annotations(report())
    )
    assert result.parts == {"family": 0.5, "font_class": 1.0, "size": 1.0}


def test_a_missing_layout_item_counts_as_wrong_and_is_listed() -> None:
    layout = items()[1:]
    result = score_typography(report(), layout, check_font_annotations(report()))
    assert result.parts == {"family": 0.5, "font_class": 0.5, "size": 0.5}
    assert result.unpaired == ("h",)


def test_fonts_on_extra_items_earn_nothing() -> None:
    layout = [
        item("x1", "Unrelated words on the page.", SANS),
        item("x2", "Other unrelated words.", SERIF),
    ]
    assert score(report(), layout) == 0.0


def test_markdown_only_parser_scores_zero() -> None:
    assert score(report(), [], markdown=f"# {TITLE}\n\n" + "\n\n".join(BODY)) == 0.0


def test_items_pair_on_their_page_only() -> None:
    t = truth(para("a", BODY[0], font=SERIF), para("b", BODY[1], page=1, font=SERIF))
    layout = [item("a", BODY[0], SERIF, page=1), item("b", BODY[1], SERIF, page=1)]
    assert score(t, layout) == 0.5


def test_metric_compatible_and_postscript_names_are_the_same_family() -> None:
    assert family_key("Helvetica") == family_key("Arial")
    assert family_key("ABCDEF+Arial-BoldMT") == family_key("Arial")
    assert family_key("TimesNewRomanPSMT") == family_key("Times New Roman")
    assert family_key("Liberation Serif") == family_key("Times")
    assert family_key("Courier New Bold Italic") == family_key("Courier")
    assert family_key("Verdana") != family_key("Arial")
    assert family_key("Arial Narrow") != family_key("Arial")
    assert family_key("Arial Black") != family_key("Arial")
    helvetica = {**SANS, "family": "Helvetica-Bold"}
    assert score(report(), items(heading_font=helvetica)) == 1.0


def test_a_wrong_family_in_the_right_class_keeps_the_class() -> None:
    verdana = {**SANS, "family": "Verdana"}
    result = score_typography(report(), items(verdana), check_font_annotations(report()))
    assert result.parts == {"family": 0.5, "font_class": 1.0, "size": 1.0}


def test_size_is_right_within_ten_percent() -> None:
    assert same_size(10.0, 11.0)
    assert same_size(10.0, 9.0)
    assert not same_size(10.0, 11.2)
    assert not same_size(10.0, None)


def test_only_annotated_fields_are_scored() -> None:
    t = report(heading=font(cls="sans"), body=font(cls="serif"))
    assert check_font_annotations(t) == {"font_class"}
    wrong = {"family": "Comic Sans", "size": 40.0}
    assert score(t, items({**wrong, "class": "sans"}, {**wrong, "class": "serif"})) == 1.0


def test_a_block_without_a_font_is_not_graded() -> None:
    t = truth(para("p0", BODY[0], font=SERIF), para("p1", BODY[1]))
    assert check_font_annotations(t) == {"family", "font_class", "size"}
    assert score(t, [item("x", BODY[0], SERIF), item("y", BODY[1], SANS)]) == 1.0
    assert score(t, [item("x", BODY[0], SERIF)]) == 1.0
    assert score(t, [item("x", BODY[0]), item("y", BODY[1], SERIF)]) == 0.0


def test_a_field_is_graded_only_on_the_blocks_that_give_it() -> None:
    t = truth(
        para("p0", BODY[0], font=SERIF), para("p1", BODY[1], font=font(cls="serif", size=10.0))
    )
    layout = [item("x", BODY[0], SERIF), item("y", BODY[1], {**SERIF, "family": "Verdana"})]
    result = score_typography(t, layout, check_font_annotations(t))
    assert result.parts == {"family": 1.0, "font_class": 1.0, "size": 1.0}
    assert result.blocks[1] == {"id": "p1", "items": ["y"], "font_class": True, "size": True}


def test_tables_and_furniture_need_no_font() -> None:
    t = truth(
        para("p", BODY[0], font=SERIF),
        {"id": "f", "class": "footer", "page": 0, "text": "Page 1", "channel": "furniture"},
        {"id": "t", "class": "table", "page": 0, "header": [["A"]], "rows": [["1"]]},
    )
    assert check_font_annotations(t) == {"family", "font_class", "size"}


def test_a_failed_parse_scores_zero() -> None:
    result = score_document(report(), None, tracks=(TRACK,))
    assert result.sub_scores["T3"]["typography"] == 0.0


def test_a_look_alike_of_the_same_genre_is_the_right_family() -> None:
    assert same_family("Montserrat", "Century Gothic")
    assert same_family("Myriad Pro", "Trebuchet MS")
    assert same_family("Roboto", "Helvetica-Bold")
    assert same_family("Garamond", "Minion Pro")
    assert same_family("Consolas", "Menlo")
    assert not same_family("Arial", "Verdana")
    assert not same_family("Minion Pro", "Times New Roman")
    assert not same_family("Courier New", "Consolas")
    roboto = {**SANS, "family": "Roboto"}
    assert score(report(), items(roboto)) == 1.0


def test_script_and_unlisted_families_count_only_as_themselves() -> None:
    assert not same_family("Brush Script MT", "Snell Roundhand")
    assert not same_family("Maxis", "Arial")
    assert same_family("Brush Script MT", "BrushScriptMT-Italic")


def test_the_genre_table_lists_each_family_once_by_its_normalized_name() -> None:
    names = [name for names in LOOK_ALIKE_GENRES.values() for name in names]
    assert len(names) == len(set(names))
    assert all(family_key(name) == name for name in names)


def test_blocks_joined_into_one_item_keep_their_font() -> None:
    joined = item("all", " ".join(BODY), SERIF)
    assert score(report(), [item("h", TITLE, SANS), joined]) == 1.0


def test_a_split_block_takes_the_item_with_most_of_its_words() -> None:
    words = BODY[0].split()
    t = truth(para("a", BODY[0], font=SERIF))
    head, tail = " ".join(words[:4]), " ".join(words[4:])
    assert score(t, [item("x", head, SERIF), item("y", tail, SANS)]) == 1.0
    assert score(t, [item("x", head, SANS), item("y", tail, SERIF)]) == 0.0
    assert score(t, [item("x", " ".join(words[:3]), SERIF)]) == 0.0
    assert score(t, [item("a", BODY[0] + " " + BODY[1], SERIF)]) == 1.0


def boxed_item(item_id: str, text: str, font_: dict, box: tuple[float, ...]) -> LayoutItem:
    return LayoutItem.model_validate(
        {"id": item_id, "class": "paragraph", "page": 0, "text": text, "box": box, "font": font_}
    )


def test_blocks_with_the_same_text_take_their_own_items_in_order() -> None:
    t = truth(para("a", "Summary", font=SANS), para("b", "Summary", font=SERIF))
    assert score(t, [item("x", "Summary", SANS), item("y", "Summary", SERIF)]) == 1.0
    assert score(t, [item("x", "Summary", SERIF), item("y", "Summary", SANS)]) == 0.0


def test_blocks_with_the_same_text_take_the_nearest_item_by_box() -> None:
    top, bottom = (0.1, 0.1, 0.3, 0.05), (0.1, 0.8, 0.3, 0.05)
    t = truth(
        boxed("b", "paragraph", bottom, "Summary", font=SERIF),
        boxed("a", "paragraph", top, "Summary", font=SANS),
    )
    layout = [boxed_item("x", "Summary", SANS, top), boxed_item("y", "Summary", SERIF, bottom)]
    assert score(t, layout) == 1.0


def test_a_block_returned_as_lines_takes_the_lines_together() -> None:
    nine = "one two three four five six seven eight nine".split()
    t = truth(para("p", " ".join(nine), font=SERIF))
    lines = [" ".join(nine[i : i + 3]) for i in (0, 3, 6)]
    assert score(t, [item(f"l{i}", line, SERIF) for i, line in enumerate(lines)]) == 1.0
    wrong = [item("l0", lines[0], SERIF), item("l1", lines[1], SANS), item("l2", lines[2], SERIF)]
    assert score(t, wrong) == 0.0
    neighbour = item("n", "nine reasons to stay with the plan this year", SANS)
    assert score(t, [item("l0", lines[0], SERIF), neighbour]) == 0.0


def test_a_block_without_words_takes_the_item_with_the_same_symbols() -> None:
    t = truth(para("p", BODY[0], font=SERIF), para("s", "$", font=SANS))
    assert score(t, [item("x", BODY[0], SERIF), item("y", "$", SANS)]) == 1.0
    assert score(t, [item("x", BODY[0], SERIF), item("y", "$", SERIF)]) == 0.5
    assert score(t, [item("x", BODY[0], SERIF)]) == 0.5


def test_short_urw_postscript_names_are_their_metric_family() -> None:
    assert same_family("Arial", "NimbusSanL-Regu")
    assert same_family("Times New Roman", "NimbusRomNo9L-Medi")
    assert same_family("Courier New", "NimbusMonL-Bold")
    assert same_family("Palatino Linotype", "URWPalladioL-Roma")
    assert same_family("Century Schoolbook", "CenturySchL-Roma")
    assert same_family("ITC Avant Garde", "URWGothicL-Book")
    assert same_family("ITC Bookman", "URWBookmanL-Ligh")


def test_details_give_each_part_and_block() -> None:
    wrong_family = {**SANS, "family": "Verdana"}
    layout = items(heading_font=wrong_family)[:-1]
    result = score_document(report(), Prediction(markdown="x", layout=layout), tracks=(TRACK,))
    details = result.details["T3"]["typography"]
    assert details["status"] == "scored"
    assert details["parts"] == pytest.approx({"family": 1 / 3, "font_class": 5 / 6, "size": 5 / 6})
    assert details["unpaired"] == ["p2"]
    assert details["blocks"][0] == {
        "id": "h",
        "items": ["h"],
        "family": False,
        "font_class": True,
        "size": True,
    }


def test_details_of_a_failed_parse_list_every_block() -> None:
    details = score_document(report(), None, tracks=(TRACK,)).details["T3"]["typography"]
    assert details["status"] == "parse_failed"
    assert details["unpaired"] == ["h", "p0", "p1", "p2"]
    assert set(details["parts"].values()) == {0.0}
