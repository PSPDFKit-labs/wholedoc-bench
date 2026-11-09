import pytest

from tests.builders import (
    ICAO_TD3,
    URL,
    code,
    context,
    invoice,
    layout_item,
    mrz,
    mrz_item,
    para,
    truth,
)
from wholedoc_bench.schema import ElementClass, Groundtruth, LayoutItem
from wholedoc_bench.scorer.tracks.t1_text import BlockEditSimilarity, WordBagF1
from wholedoc_bench.scorer.tracks.t3_structure import BlockIntegrity
from wholedoc_bench.scorer.tracks.t5_machine_readable import (
    BarcodePayloads,
    MrzIdentified,
    MrzText,
    MrzValidity,
)


def score(t: Groundtruth, markdown: str, layout: tuple[LayoutItem, ...] = ()) -> float:
    return BarcodePayloads().score(context(t, markdown, layout))


def test_applies_only_to_a_barcode_with_a_payload() -> None:
    assert BarcodePayloads().applies(invoice())
    assert not BarcodePayloads().applies(invoice(code("q", payload=None)))
    assert not BarcodePayloads().applies(truth(para("p", "Scan to pay.")))


def test_the_exact_payload_anywhere_in_the_markdown_passes() -> None:
    assert score(invoice(), f"Scan to pay.\n\n{URL}\n") == 1.0
    assert score(invoice(), f"Scan to pay at {URL} today.") == 1.0


def test_a_changed_case_or_character_fails() -> None:
    outputs = {
        "case": f"Scan to pay.\n\n{URL.upper()}\n",
        "one wrong character": "Scan to pay.\n\nhttps://example.com/inv/4471\n",
    }
    found = {name: score(invoice(), markdown) for name, markdown in outputs.items()}
    assert found == dict.fromkeys(outputs, 0.0)


@pytest.mark.parametrize(
    ("payload", "markdown"),
    [
        ("INV-4417", "INV-44170"),
        ("INV-4417", "XINV-4417"),
        ("INV_4417", "INV_4417_B"),
    ],
)
def test_a_longer_code_does_not_hold_the_payload(payload: str, markdown: str) -> None:
    assert score(invoice(code("q", payload=payload)), f"Scan to pay.\n\n{markdown}\n") == 0.0


@pytest.mark.parametrize("markdown", ["Order INV-4417.", "<INV-4417>", "**INV-4417**"])
def test_spaces_brackets_and_punctuation_may_touch_the_payload(markdown: str) -> None:
    assert score(invoice(code("q", payload="INV-4417")), f"Scan to pay.\n\n{markdown}\n") == 1.0


def test_a_longer_code_that_adds_other_characters_still_holds_a_copy() -> None:
    assert score(invoice(), f"Scan to pay.\n\n{URL}/extra\n") == 1.0


def test_markdown_escapes_do_not_count() -> None:
    t = invoice(code("q", payload="INV_4417*A"))
    assert score(t, "Scan to pay.\n\nINV\\_4417\\*A\n") == 1.0


def test_whitespace_runs_count_as_one_space() -> None:
    t = invoice(code("q", payload="Jane Roe\nACME  Corp"))
    assert score(t, "Scan to pay.\n\nJane Roe ACME\nCorp\n") == 1.0


def test_a_payload_of_two_codes_must_be_found_twice() -> None:
    t = invoice(code("q1"), code("q2"))
    assert score(t, f"Scan to pay.\n\n{URL}\n") == 0.5
    assert score(t, f"Scan to pay.\n\n{URL}\n\n{URL}\n") == 1.0


def test_extra_copies_earn_nothing() -> None:
    t = invoice(code("q1"), code("q2", payload="INV-9"))
    assert score(t, f"{URL}\n\n{URL}\n\n{URL}\n") == 0.5


def test_a_code_without_a_payload_is_not_counted() -> None:
    t = invoice(code("q1"), code("q2", payload=None))
    assert score(t, f"Scan to pay.\n\n{URL}\n") == 1.0


def test_the_layout_json_is_not_read() -> None:
    item = LayoutItem(id="x", cls=ElementClass.MACHINE_READABLE, page=0, text=URL)
    assert score(invoice(), "Scan to pay.\n", (item,)) == 0.0


def _scores(markdown: str, t: Groundtruth | None = None) -> dict[str, float]:
    ctx = context(t or invoice(), markdown)
    metrics = (WordBagF1(), BlockEditSimilarity(), BlockIntegrity(), BarcodePayloads())
    return {metric.name: metric.score(ctx) for metric in metrics}


def test_a_decoded_payload_costs_nothing_in_t1_and_t3() -> None:
    assert _scores(f"Scan to pay.\n\n{URL}\n") == {
        "word_bag_f1": 1.0,
        "block_edit_similarity": 1.0,
        "block_integrity": 1.0,
        "barcode_payloads": 1.0,
    }


def test_a_payload_written_as_two_paragraphs_costs_nothing_in_t1_and_t3() -> None:
    t = invoice(code("q", payload="Jane Roe\nACME Corp"))
    assert _scores("Scan to pay.\n\nJane Roe\n\nACME Corp\n", t) == {
        "word_bag_f1": 1.0,
        "block_edit_similarity": 1.0,
        "block_integrity": 1.0,
        "barcode_payloads": 1.0,
    }


def test_an_invented_label_still_costs_in_t1() -> None:
    assert _scores(f"Scan to pay.\n\nQR code: {URL}\n")["word_bag_f1"] < 1.0


def test_a_wrong_decode_costs_in_t1_and_t3_too() -> None:
    scores = _scores("Scan to pay.\n\nhttps://example.com/inv/4471\n")
    assert scores["barcode_payloads"] == 0.0
    assert scores["word_bag_f1"] < 1.0
    assert scores["block_integrity"] < 1.0


TD3_FAILING = ICAO_TD3[:-1] + "1"


def passport(*zones: dict[str, object]) -> Groundtruth:
    return truth(para("p", "Passport"), *(zones or [mrz("m")]))


def mrz_scores(
    t: Groundtruth, layout: tuple[LayoutItem, ...], markdown: str = "Passport\n"
) -> dict[str, float]:
    ctx = context(t, markdown, layout)
    return {m.name: m.score(ctx) for m in (MrzIdentified(), MrzText(), MrzValidity())}


def grades(identified: float, text: float, validity: float) -> dict[str, float]:
    return {"mrz_identified": identified, "mrz_text": text, "mrz_validity": validity}


def test_the_mrz_grades_apply_only_to_a_zone_and_barcode_payloads_leaves_it_to_them() -> None:
    assert all(m.applies(passport()) for m in (MrzIdentified(), MrzText(), MrzValidity()))
    assert not MrzIdentified().applies(invoice())
    assert not BarcodePayloads().applies(passport())


def test_a_zone_identified_read_and_judged_right_scores_every_grade() -> None:
    assert mrz_scores(passport(), (mrz_item(),)) == grades(1, 1, 1)


def test_a_misread_line_loses_only_the_text() -> None:
    misread = ICAO_TD3.replace("L898902C3", "L898902O3")
    assert mrz_scores(passport(), (mrz_item(text=misread),)) == grades(1, 0, 1)


def test_a_wrong_or_missing_verdict_loses_only_the_validity() -> None:
    assert mrz_scores(passport(), (mrz_item(valid=False),)) == grades(1, 1, 0)
    assert mrz_scores(passport(), (mrz_item(valid=None),)) == grades(1, 1, 0)


def test_a_zone_whose_check_digits_fail_wants_the_verdict_false() -> None:
    t = passport(mrz("m", payload=TD3_FAILING))
    assert mrz_scores(t, (mrz_item(text=TD3_FAILING, valid=False),)) == grades(1, 1, 1)
    assert mrz_scores(t, (mrz_item(text=TD3_FAILING, valid=True),)) == grades(1, 1, 0)


@pytest.mark.parametrize(
    "item",
    [
        layout_item("z", ElementClass.PARAGRAPH, text=ICAO_TD3),
        mrz_item(format_=None),
        mrz_item(box=(0.05, 0.1, 0.9, 0.1)),
        mrz_item(page=1),
        mrz_item(box=None),
    ],
)
def test_a_zone_no_item_identifies_scores_nothing_at_all(item: LayoutItem) -> None:
    assert mrz_scores(passport(), (item,)) == grades(0, 0, 0)


def test_the_right_lines_in_the_markdown_alone_score_nothing() -> None:
    markdown = f"Passport\n\n```\n{ICAO_TD3}\n```\n"
    assert mrz_scores(passport(), (), markdown) == grades(0, 0, 0)


def test_every_zone_counts_whether_identified_or_not() -> None:
    second = mrz("n", box=(0.05, 0.1, 0.9, 0.1))
    assert mrz_scores(passport(mrz("m"), second), (mrz_item(),)) == grades(0.5, 0.5, 0.5)


def test_the_order_of_items_alike_but_for_their_verdict_does_not_change_the_grades() -> None:
    items = (mrz_item("a", valid=True), mrz_item("b", valid=False), mrz_item("c", valid=None))
    orders = [items, items[::-1], (items[2], items[0], items[1])]
    assert [mrz_scores(passport(), order) for order in orders] == [grades(1, 1, 1)] * 3


def test_an_extra_zone_item_costs_nothing() -> None:
    extra = mrz_item("x", box=(0.05, 0.1, 0.9, 0.1))
    assert mrz_scores(passport(), (mrz_item(), extra)) == grades(1, 1, 1)


def test_zone_lines_in_the_markdown_cost_nothing_in_t1() -> None:
    ctx = context(passport(), f"Passport\n\n```\n{ICAO_TD3}\n```\n")
    assert WordBagF1().score(ctx) == 1.0
