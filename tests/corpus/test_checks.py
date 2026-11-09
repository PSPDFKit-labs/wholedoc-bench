import warnings
from typing import Any

import pytest

from tests.builders import (
    URL,
    VISIBLE,
    boxed,
    cell,
    chart,
    code,
    element,
    grid_area,
    grid_box,
    groundtruth,
    heading,
    incomplete_truth,
    invoice,
    memo,
    para,
    secret,
    span,
    truth,
    typo,
)
from wholedoc_bench.corpus.checks import (
    CorpusError,
    CorpusWarning,
    check_groundtruth,
    check_style_annotations,
)
from wholedoc_bench.schema import TableCell
from wholedoc_bench.scorer.text_styles import unknown_styles


@pytest.mark.parametrize("styles", [None, []])
def test_all_unknown_is_inactive_and_all_annotated_unstyled_is_active(styles: Any) -> None:
    truth = groundtruth([element(styles=styles), element(id="p2", styles=styles)])
    assert check_style_annotations(truth) is (styles is not None)


def test_mixed_styled_and_explicitly_unstyled_elements_are_complete() -> None:
    truth = groundtruth([element(styles=[span()]), element(id="p2", styles=[])])
    assert check_style_annotations(truth)
    check_groundtruth(truth)


def test_an_element_without_styles_is_left_out_not_rejected() -> None:
    truth = incomplete_truth()
    assert check_style_annotations(truth)
    check_groundtruth(truth)
    assert unknown_styles(truth) == {"unknown"}


@pytest.mark.parametrize(
    "excluded",
    [{"channel": "hidden"}, {"class": "chart"}, {"text": " "}],
)
def test_excluded_annotations_neither_activate_nor_satisfy_completeness(
    excluded: dict[str, Any],
) -> None:
    excluded_element = element(id="excluded", styles=[], **excluded)
    assert not check_style_annotations(groundtruth([excluded_element]))
    assert not check_style_annotations(groundtruth([excluded_element, element()]))
    assert check_style_annotations(groundtruth([excluded_element, element(styles=[])]))
    truth = groundtruth([excluded_element, element(styles=[])])
    truth = truth.model_copy(update={"elements": [*truth.elements, element(id="unknown")]})
    assert check_style_annotations(truth)


def test_the_corpus_rejects_a_string_with_fewer_than_four_letters_and_digits() -> None:
    hidden = {**para("h", "A-1"), "channel": "hidden"}
    trap = {"kind": "hidden", "printed": "A-1"}
    with pytest.raises(CorpusError, match="has fewer than 4 letters and digits"):
        truth(para("p", VISIBLE), hidden, planted=[trap])


def test_the_corpus_rejects_the_same_string_planted_twice() -> None:
    first = {**para("h1", "acct 4417"), "channel": "hidden"}
    second = {**para("h2", "ACCT-4417"), "channel": "hidden"}
    planted = [
        {"kind": "hidden", "printed": "acct 4417"},
        {"kind": "redacted", "printed": "ACCT-4417"},
    ]
    with pytest.raises(CorpusError, match="repeats another planted string"):
        truth(para("p", VISIBLE), first, second, planted=planted)


def test_the_corpus_finds_the_string_in_hidden_text_without_accents() -> None:
    held = {**para("s", "Signed, Élodie Roe."), "channel": "hidden"}
    truth(
        para("p", VISIBLE),
        held,
        planted=[{"kind": "redacted", "printed": "Elodie Roe"}],
    )


def test_the_corpus_finds_a_string_held_in_chart_data() -> None:
    held = {**chart(title="Revenue"), "channel": "hidden"}
    held["chart"]["series"][0]["name"] = "acct 4417"
    truth(
        para("p", VISIBLE),
        held,
        planted=[{"kind": "hidden", "printed": "acct 4417"}],
    )


def test_the_corpus_rejects_a_string_missing_from_hidden_text() -> None:
    match = r"planted redacted 'acct 9999' is not in the text of a hidden element"
    with pytest.raises(CorpusError, match=match):
        memo({"kind": "redacted", "printed": "acct 9999"})


@pytest.mark.parametrize(
    "visible",
    [
        [para("p", "Jane Roe joined the board.")],
        [
            {
                "id": "t",
                "class": "table",
                "page": 0,
                "header": ["First", "Last"],
                "rows": [["Jane", "Roe"]],
            }
        ],
        [{**para("f", "Jane Roe"), "channel": "furniture"}],
        [chart(title="Jane Roe")],
        [
            para("p1", "Signed by Jane"),
            {**para("f", "Page 1"), "channel": "furniture"},
            para("p2", "Roe"),
        ],
    ],
)
def test_the_corpus_rejects_a_string_printed_in_visible_text(visible: list[dict[str, Any]]) -> None:
    with pytest.raises(CorpusError, match="planted redacted 'Jane Roe' is printed in visible text"):
        secret("Jane Roe", *visible)


def test_the_corpus_rejects_a_string_printed_as_a_chart_value() -> None:
    sales = chart()
    sales["chart"]["series"][0]["points"] = [{"label": "North", "value": 4417}]
    with pytest.raises(CorpusError, match="is printed in visible text"):
        secret("4417", para("p", VISIBLE), sales)


@pytest.mark.parametrize(("overlap", "rejected"), [(0.01, False), (0.02, True)])
def test_cells_of_one_table_may_overlap_by_at_most_one_percent(
    overlap: float, rejected: bool
) -> None:
    x, y, w, h = grid_box(0, 1)
    cells = [cell("Name", grid_box(0, 0)), cell("Value", (x - w * overlap, y, w, h))]
    texts = [["Name", "Value"]]
    table = boxed("t", "table", grid_area(texts), header=texts[0], rows=[], cells=cells)
    if rejected:
        with pytest.raises(CorpusError, match="cells 0 and 1 of table t overlap"):
            truth(table)
    else:
        assert truth(table).elements[0].cells == [TableCell.model_validate(c) for c in cells]


def test_a_wide_cell_over_a_later_column_is_rejected() -> None:
    cells = [
        cell("Name", (0.1, 0.3, 0.4, 0.1)),
        cell("Gap", (0.15, 0.5, 0.05, 0.1)),
        cell("Value", (0.3, 0.3, 0.2, 0.1)),
    ]
    table = boxed("t", "table", (0.1, 0.3, 0.4, 0.3), header=["Name"], rows=[], cells=cells)
    with pytest.raises(CorpusError, match="cells 0 and 2 of table t overlap"):
        truth(table)


def test_a_table_text_with_every_cell_word_loads_with_a_warning() -> None:
    text = "Quarterly figures Name Value alpha 1"
    table = boxed(
        "t", "table", (0.1, 0.3, 0.4, 0.2), text, header=["Name", "Value"], rows=[["alpha", "1"]]
    )
    with pytest.warns(CorpusWarning, match="text of table t holds every word of its cells"):
        truth(table)


def test_a_table_text_without_every_cell_word_loads_without_a_warning() -> None:
    text = "Revenue in million euros"
    table = boxed("t", "table", (0.1, 0.3, 0.4, 0.2), text, header=[text], rows=[["5"]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorpusWarning)
        truth(table)


def test_a_payload_printed_as_page_text_is_rejected() -> None:
    with pytest.raises(CorpusError, match="printed in visible text"):
        truth(para("p", f"Scan or visit {URL}."), code("q"))


def test_a_payload_printed_across_two_elements_is_rejected() -> None:
    with pytest.raises(CorpusError, match="printed in visible text"):
        truth(para("p1", "Ref INV-"), para("p2", "4417 due."), code("q", payload="INV- 4417"))


def test_a_payload_printed_only_in_another_form_is_accepted() -> None:
    t = truth(para("p", "Order INV-44170 shipped."), code("q", payload="INV-4417"))
    check_groundtruth(t)
    check_groundtruth(invoice())


def test_trap_not_printed_in_the_document_is_rejected() -> None:
    with pytest.raises(CorpusError, match="not printed in body text outside tables"):
        truth(para("p", "Please recieve the parcel."), planted=[typo("absent")])


def test_trap_inside_a_longer_word_is_not_printed() -> None:
    with pytest.raises(CorpusError, match="not printed in body text outside tables"):
        truth(para("p", "Please recieved it."), planted=[typo("recieve")])


def test_trap_only_in_a_table_is_rejected() -> None:
    with pytest.raises(CorpusError, match="not printed in body text outside tables"):
        truth(
            para("p", "Delivery note."),
            {"id": "t", "class": "table", "page": 0, "header": ["Name"], "rows": [["Muller"]]},
            planted=[typo("Muller")],
        )


def test_trap_only_in_furniture_is_rejected() -> None:
    with pytest.raises(CorpusError, match="not printed in body text outside tables"):
        truth(
            para("p", "Delivery note."),
            {"id": "f", "class": "footer", "page": 0, "text": "Muller", "channel": "furniture"},
            planted=[typo("Muller")],
        )


def test_trap_in_decomposed_text_is_accepted() -> None:
    truth(para("p", "Herr Mu\u0308ller."), planted=[typo("Müller")])


def test_the_corpus_checks_reject_a_heading_without_a_level() -> None:
    with pytest.raises(CorpusError, match="heading r has no level"):
        check_groundtruth(truth(heading("r", "Report", None)))
    check_groundtruth(truth(heading("h", "Annual report 2025", None, channel="furniture")))


def test_a_merged_cell_is_not_printed_twice_in_a_row() -> None:
    merged = {
        "id": "t",
        "class": "table",
        "page": 0,
        "header": ["North", None],
        "rows": [["2025", "2026"]],
        "spans": [{"row": 0, "column": 0, "columns": 2}],
    }
    secret("North North", para("p", VISIBLE), merged)
