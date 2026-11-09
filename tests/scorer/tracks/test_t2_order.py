from tests.builders import para, truth
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.base import SubMetric
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t2_order import (
    JOIN,
    TRACK,
    BlockOrderSimilarity,
    OrderTests,
    locating_sentences,
    order_tests,
)

ORDER_TESTS = OrderTests()
BLOCK_ORDER = BlockOrderSimilarity()

TWO_COLUMNS = truth(
    {"id": "e1", "class": "heading", "page": 0, "text": "Annual results", "level": 1},
    para("e2", "The board met in March to review the budget. It was approved by the board."),
    para("e3", "Costs fell for the third year in a row."),
    para("e4", "Revenue grew in every region, led by the north."),
    para("e5", "Revenue grew in every region last year as well."),
    para("e6", "The next meeting is planned for June.", page=1),
)
E1 = "# Annual results"
E2 = "The board met in March to review the budget. It was approved by the board."
E3 = "Costs fell for the third year in a row."
E4 = "Revenue grew in every region, led by the north."
E5 = "Revenue grew in every region last year as well."
E6 = "The next meeting is planned for June."

PAGE = truth(
    {"id": "e1", "class": "heading", "page": 0, "text": "Annual results", "level": 1},
    para("e2", "The board met in March to review the budget. It was approved by the board."),
    para("e3", "Costs fell for the third year in a row."),
    para("e4", "See the notes below. Revenue grew in every region, led by the north."),
    para("e5", "See the notes below. Staff numbers stayed the same."),
    para("e6", "The next meeting is planned for June.", page=1),
)
U1, U2, U3, U4, U5, U6 = (
    "# Annual results",
    "The board met in March to review the budget. It was approved by the board.",
    "Costs fell for the third year in a row.",
    "See the notes below. Revenue grew in every region, led by the north.",
    "See the notes below. Staff numbers stayed the same.",
    "The next meeting is planned for June.",
)


def score(metric: SubMetric, t: Groundtruth, *blocks: str) -> float:
    return metric.score(DocContext(t, Prediction(markdown="\n\n".join(blocks))))


def lines(t: Groundtruth) -> list[tuple[str, str]]:
    return [(_plain(test.before), _plain(test.after)) for test in order_tests(t)]


def _plain(side: str) -> str:
    return " ".join(w for w in side.split() if w != JOIN)


def test_a_document_without_tests_or_two_placed_units_does_not_apply() -> None:
    one_paragraph = truth(para("a", "One paragraph. Two sentences."))
    assert not ORDER_TESTS.applies(one_paragraph)
    assert not BLOCK_ORDER.applies(one_paragraph)
    assert ORDER_TESTS.applies(TWO_COLUMNS)
    assert BLOCK_ORDER.applies(PAGE)


def test_groundtruth_order_scores_one() -> None:
    assert score(ORDER_TESTS, TWO_COLUMNS, E1, E2, E3, E4, E5, E6) == 1.0
    assert score(BLOCK_ORDER, PAGE, U1, U2, U3, U4, U5, U6) == 1.0


def test_nutrient_output_for_the_fixture_scores_one(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    ctx = DocContext(demo_groundtruth, pred)
    assert len(order_tests(demo_groundtruth)) == 3
    assert len(locating_sentences(demo_groundtruth)) == 3
    assert ORDER_TESTS.score(ctx) == 1.0
    assert BLOCK_ORDER.score(ctx) == 1.0
    assert TRACK.evaluate(demo_groundtruth, ctx).score == 1.0


def test_near_empty_output_scores_zero() -> None:
    assert score(ORDER_TESTS, TWO_COLUMNS, "x") == 0.0
    assert score(BLOCK_ORDER, PAGE, "x") == 0.0


def test_each_pair_of_neighbouring_units_gives_one_test_page_breaks_included() -> None:
    assert lines(TWO_COLUMNS) == [
        ("annual results", "the board met in march to review the budget"),
        ("it was approved by the board", "costs fell for the third year in a row"),
        ("costs fell for the third year in a row", "revenue grew in every region led by the north"),
        (
            "revenue grew in every region led by the north",
            "revenue grew in every region last year as well",
        ),
        ("revenue grew in every region last year as well", "the next meeting is planned for june"),
    ]


def test_a_break_inside_a_paragraph_tests_the_whole_sentences_around_it() -> None:
    def continued(first: str, second: str, hyphenated: bool) -> Groundtruth:
        return truth(
            para("a", first),
            para("b", second, page=1),
            relations=[{"kind": "continues_in", "from": "a", "to": "b", "hyphenated": hyphenated}],
        )

    documents = {
        "cut sentence": continued(
            "The plan was approved. Work starts", "in June. Costs are fixed.", False
        ),
        "sentence end": continued("The plan was approved.", "Costs are fixed.", False),
        "hyphenated word": continued(
            "The plan was approved. The work is comp-", "lex. Costs are fixed.", True
        ),
    }
    found = {name: lines(t) for name, t in documents.items()}
    assert found == dict.fromkeys(documents, [("the plan was approved", "costs are fixed")])


def test_units_follow_reading_order_not_file_order() -> None:
    t = truth(para("a", "Second part."), para("b", "First part."))
    t = t.model_copy(update={"reading_order": ["b", "a"]})
    assert lines(t) == [("first part", "second part")]


def test_captions_tables_and_furniture_are_not_units() -> None:
    t = truth(
        para("a", "The results are below."),
        {"id": "c", "class": "caption", "page": 0, "text": "Table 1: Results."},
        {"id": "t", "class": "table", "page": 0, "rows": [["Q3", "18.2"]]},
        {"id": "f", "class": "footer", "page": 0, "text": "Page 1", "channel": "furniture"},
        para("b", "The table shows growth."),
    )
    assert lines(t) == [("the results are below", "the table shows growth")]


def test_a_repeated_sentence_is_glued_to_the_one_before_it() -> None:
    t = truth(
        para("a", "Costs fell. See the notes."),
        para("b", "Revenue grew. See the notes."),
        para("c", "Staff stayed the same."),
    )
    assert lines(t) == [
        ("costs fell see the notes", "revenue grew"),
        ("revenue grew see the notes", "staff stayed the same"),
    ]


def test_a_one_word_heading_is_glued_to_its_neighbour_sentence() -> None:
    t = truth(
        para("p1", "All figures are unaudited."),
        {"id": "h", "class": "heading", "page": 0, "text": "Revenue", "level": 2},
        para("p2", "Sales rose in every region, and revenue too."),
    )
    assert lines(t) == [
        ("all figures are unaudited", "revenue sales rose in every region and revenue too"),
        ("all figures are unaudited revenue", "sales rose in every region and revenue too"),
    ]
    p1, p2 = "All figures are unaudited.", "Sales rose in every region, and revenue too."
    assert score(ORDER_TESTS, t, p1, "## Revenue", p2) == 1.0
    assert score(ORDER_TESTS, t, p1, p2, "## Revenue") == 0.0


def test_a_glued_heading_is_searched_on_its_own() -> None:
    t = truth(
        para("p1", "Ébénisterie du Lauragais, 9 chemin de la Saudrune."),
        {"id": "h1", "class": "heading", "page": 0, "text": "Facture", "level": 1},
        {"id": "h2", "class": "heading", "page": 0, "text": "Facturé à", "level": 2},
        para("p2", "Gardez cette facture."),
    )
    p1, p2 = "Ébénisterie du Lauragais, 9 chemin de la Saudrune.", "Gardez cette facture."
    assert score(ORDER_TESTS, t, p1, "# Facture", "## Facturé à", p2) == 1.0
    assert score(ORDER_TESTS, t, p1, "## Facturé à", p2, "# Facture") == 1 / 3


def test_unused_edits_of_a_glued_sentence_may_lie_before_the_next_one() -> None:
    t = truth(
        para("p1", "We approved the plan."),
        {"id": "h1", "class": "heading", "page": 0, "text": "Sales", "level": 1},
        para("p2", "Sales grew."),
    )
    assert score(ORDER_TESTS, t, "We approved the plann.", "# Sales", "Sales grew.") == 1.0
    assert score(ORDER_TESTS, t, "We approved the plann.", "Sales grew.", "# Sales") < 1.0


def test_a_near_copy_counts_as_a_repeat() -> None:
    t = truth(
        para("a", "The committee approved the plan on 4 March 2024."),
        para("b", "The committee approved the plan on 5 March 2024."),
        para("c", "Nothing else changed."),
    )
    assert lines(t) == [
        (
            "4 march 2024 the committee approved the plan on 5 march 2024",
            "nothing else changed",
        )
    ]


def test_a_sentence_also_printed_as_furniture_gives_no_test() -> None:
    t = truth(
        {"id": "h", "class": "header", "page": 0, "text": "Annual results", "channel": "furniture"},
        {"id": "e1", "class": "heading", "page": 0, "text": "Annual results", "level": 1},
        para("e2", "The board met in March."),
    )
    assert lines(t) == []


def test_a_long_sentence_is_searched_by_the_words_at_the_jump() -> None:
    long_a = (
        "The board met in March to review the budget for the coming year and approved it "
        "after a long discussion about the cost of energy."
    )
    long_b = (
        "Revenue grew in every region during the second half of the year, led by strong "
        "export orders from the north and the east."
    )
    t = truth(para("a", long_a), para("b", long_b))
    assert lines(t) == [
        (
            "and approved it after a long discussion about the cost of energy",
            "revenue grew in every region during the second half of the year",
        )
    ]
    assert score(ORDER_TESTS, t, long_a, long_b) == 1.0
    assert score(ORDER_TESTS, t, long_b, long_a) == 0.0


def test_a_page_number_inside_a_paragraph_does_not_fail_a_glued_test() -> None:
    t = truth(
        para("a", "Revenue grew. See the notes."),
        para("b", "Costs fell.", page=1),
        para("c", "Staff stayed the same.", page=1),
        {"id": "n", "class": "footnote", "page": 1, "text": "Costs fell."},
        relations=[{"kind": "continues_in", "from": "a", "to": "b"}],
    )
    joined = "Revenue grew. See the notes. Costs fell.\n\nStaff stayed the same.\n\nCosts fell."
    with_page_number = (
        "Revenue grew. See the notes.\n\n2\n\nCosts fell.\n\nStaff stayed the same.\n\nCosts fell."
    )
    assert score(ORDER_TESTS, t, joined) == 1.0
    assert score(ORDER_TESTS, t, with_page_number) == 1.0


def test_right_column_first_fails_the_column_jump() -> None:
    assert score(ORDER_TESTS, TWO_COLUMNS, E1, E4, E5, E2, E3, E6) == 0.8


def test_lines_read_straight_across_both_columns_fail_most_tests() -> None:
    across = (
        "# Annual results\n\n"
        "The board met in March to review Revenue grew in every region, led by\n"
        "the budget. It was approved by the board. the north. Revenue grew in every\n"
        "Costs fell for the third year in a row. region last year as well.\n\n"
        "The next meeting is planned for June."
    )
    assert score(ORDER_TESTS, TWO_COLUMNS, across) == 0.2


def test_the_fixture_with_the_page_3_fragment_first_fails_the_break_test(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult
) -> None:
    pred = NutrientProvider().normalize(demo_nutrient_raw)
    head, cut, tail = pred.markdown.partition("lex example.")
    moved = f"{cut}{tail}\n\n{head}"
    assert ORDER_TESTS.score(DocContext(demo_groundtruth, Prediction(markdown=moved))) == 2 / 3


def test_a_missing_sentence_fails_its_tests() -> None:
    assert score(ORDER_TESTS, TWO_COLUMNS, E1, E2, E4, E5, E6) == 0.6


COMMITTEE = truth(
    para("a", "The committee approved the budget."),
    para("b", "Costs fell for the third year in a row."),
)
A, B = "The committee approved the budget.", "Costs fell for the third year in a row."


def test_case_punctuation_markup_and_typos_are_not_order_errors() -> None:
    grew = truth(para("a", "We met the board."), para("b", "Costs fell."))
    outputs = {
        "case, punctuation and markup": (
            TWO_COLUMNS,
            [
                "## ANNUAL RESULTS",
                "the board met in march, to review the budget; it was **approved** by the board",
                "- Costs fell for the third year in a row",
                E4,
                E5,
                E6,
            ],
        ),
        "typo in a long sentence": (
            TWO_COLUMNS,
            [E1, E2.replace("approved", "aproved"), E3, E4, E5, E6],
        ),
        "first letter lost": (COMMITTEE, ["he committee approved the budget.", B]),
        "last letter lost": (COMMITTEE, ["The committee approved the budge.", B]),
        "first letter doubled": (COMMITTEE, ["TThe committee approved the budget.", B]),
        "last word grew": (grew, ["We met the boardroom.", "Costs fell."]),
    }
    found = {name: score(ORDER_TESTS, t, *blocks) for name, (t, blocks) in outputs.items()}
    assert found == dict.fromkeys(outputs, 1.0)


def test_a_late_copy_with_a_typo_still_makes_the_order_unclear() -> None:
    assert score(ORDER_TESTS, COMMITTEE, A, B, "he committee approved the budget.") == 0.0


CUT = truth(
    para("p0", "Revenue grew in every region this year."),
    para("a", "The committee approved"),
    para("b", "the budget after a lengthy discussion.", page=1),
    para("c", "Staff numbers stayed the same.", page=1),
    {"id": "h", "class": "header", "page": 1, "text": "Annual report 2026", "channel": "furniture"},
    relations=[{"kind": "continues_in", "from": "a", "to": "b"}],
)
REVENUE, PIECE_1 = "Revenue grew in every region this year.", "The committee approved"
PIECE_2, STAFF = "the budget after a lengthy discussion.", "Staff numbers stayed the same."


def test_a_sentence_cut_by_a_page_break_is_searched_as_its_pieces_in_order() -> None:
    assert lines(CUT) == [
        (
            "revenue grew in every region this year",
            "the committee approved \N{HORIZONTAL ELLIPSIS} the budget after a lengthy discussion",
        ),
        (
            "the committee approved \N{HORIZONTAL ELLIPSIS} the budget after a lengthy discussion",
            "staff numbers stayed the same",
        ),
    ]
    assert score(ORDER_TESTS, CUT, REVENUE, f"{PIECE_1} {PIECE_2}", STAFF) == 1.0


def test_a_running_header_between_the_pieces_is_not_an_order_error() -> None:
    blocks = (REVENUE, PIECE_1, "Annual report 2026", PIECE_2, STAFF)
    assert score(ORDER_TESTS, CUT, *blocks) == 1.0
    assert score(BLOCK_ORDER, CUT, *blocks) == 1.0


def test_pieces_swapped_or_far_apart_fail() -> None:
    assert score(ORDER_TESTS, CUT, REVENUE, PIECE_2, PIECE_1, STAFF) == 0.0
    filler = " ".join(["Other words in between."] * 4)
    assert score(ORDER_TESTS, CUT, REVENUE, PIECE_1, STAFF, filler, PIECE_2) == 0.0


def test_a_split_word_at_the_cut_may_be_joined_or_in_two_halves() -> None:
    t = truth(
        para("p0", "Revenue grew."),
        para("a", "The work is comp-"),
        para("b", "lex and costly.", page=1),
        para("c", "Costs are fixed.", page=1),
        relations=[{"kind": "continues_in", "from": "a", "to": "b", "hyphenated": True}],
    )
    joined = ["Revenue grew.", "The work is complex and costly.", "Costs are fixed."]
    assert score(ORDER_TESTS, t, *joined) == 1.0
    halves = ["Revenue grew.", "The work is comp-", "2", "lex and costly.", "Costs are fixed."]
    assert score(ORDER_TESTS, t, *halves) == 1.0


def test_a_number_joined_at_the_cut_may_be_joined_or_in_two_halves() -> None:
    t = truth(
        para("p0", "Budget notes begin here."),
        para("a", "The summary table lists pages 10-"),
        para("b", "12 as the key results.", page=1),
        para("c", "Closing words end the document.", page=1),
        relations=[{"kind": "continues_in", "from": "a", "to": "b"}],
    )
    start, end = "Budget notes begin here.", "Closing words end the document."
    joined = [start, "The summary table lists pages 10-12 as the key results.", end]
    halves = [start, "The summary table lists pages 10-", "Page 4", "12 as the key results.", end]
    for blocks in (joined, halves):
        assert score(ORDER_TESTS, t, *blocks) == 1.0
        assert score(BLOCK_ORDER, t, *blocks) == 1.0


def test_a_sentence_cut_by_a_page_break_counts_as_a_copy_of_the_same_sentence() -> None:
    t = truth(
        {"id": "h", "class": "heading", "page": 0, "text": "Market update", "level": 1},
        para("a", "Costs fell again in the north."),
        para("b", "Revenue rose. Costs fell"),
        para("c", "again in the north.", page=1),
        relations=[{"kind": "continues_in", "from": "b", "to": "c"}],
    )
    blocks = [
        "# Market update",
        "Costs fell again in the north.",
        "Revenue rose. Costs fell again in the north.",
    ]
    assert score(ORDER_TESTS, t, *blocks) == 1.0
    assert score(BLOCK_ORDER, t, *blocks) == 1.0


def test_a_word_split_at_a_page_break_counts_as_a_copy_of_a_heading() -> None:
    t = truth(
        {"id": "h", "class": "heading", "page": 0, "text": "Outlook", "level": 1},
        para("a", "Costs fell in the north."),
        para("b", "The board expects a better out-"),
        para("c", "look next year.", page=1),
        relations=[{"kind": "continues_in", "from": "b", "to": "c", "hyphenated": True}],
    )
    blocks = [
        "# Outlook",
        "Costs fell in the north.",
        "The board expects a better outlook next year.",
    ]
    assert score(ORDER_TESTS, t, *blocks) == 1.0
    assert score(BLOCK_ORDER, t, *blocks) == 1.0


def test_a_paragraph_written_as_a_table_still_counts() -> None:
    table = "| Costs fell for the third year in a row. |\n|---|"
    assert score(ORDER_TESTS, TWO_COLUMNS, E1, E2, table, E4, E5, E6) == 1.0


def test_a_copy_before_and_after_makes_the_order_unclear() -> None:
    assert score(ORDER_TESTS, TWO_COLUMNS, E1, E2, E3, E4, E5, E6, E2) == 0.8


def test_extra_copies_before_the_next_sentence_keep_the_order_clear() -> None:
    assert score(ORDER_TESTS, TWO_COLUMNS, E1, E2, E2, E3, E4, E5, E6) == 1.0


def test_the_whole_output_twice_scores_zero() -> None:
    once = [E1, E2, E3, E4, E5, E6]
    assert score(ORDER_TESTS, TWO_COLUMNS, *once, *once) == 0.0


def test_the_whole_output_in_both_orders_scores_zero() -> None:
    once = [E1, E2, E3, E4, E5, E6]
    assert score(ORDER_TESTS, TWO_COLUMNS, *once, *reversed(once)) == 0.0


def test_each_unit_is_found_by_its_first_unique_sentence() -> None:
    assert [_plain(a) for a in locating_sentences(PAGE)] == [
        "annual results",
        "the board met in march to review the budget",
        "costs fell for the third year in a row",
        "revenue grew in every region led by the north",
        "staff numbers stayed the same",
        "the next meeting is planned for june",
    ]


def test_a_unit_with_no_unique_sentence_is_found_with_the_next_sentence() -> None:
    t = truth(para("a", "See the notes."), para("b", "See the notes."), para("c", "Costs fell."))
    assert [_plain(a) for a in locating_sentences(t)] == [
        "see the notes see the notes",
        "see the notes costs fell",
        "costs fell",
    ]


def test_a_unit_that_stays_unclear_is_left_out() -> None:
    note = "See the detailed notes at the end of this report."
    t = truth(
        para("a", note),
        {"id": "t", "class": "table", "page": 0, "rows": [["Q3", "18.2"]]},
        para("b", note),
        para("c", "Costs fell.", page=1),
    )
    assert [_plain(a) for a in locating_sentences(t)] == ["costs fell"]
    assert lines(t) == []


def test_right_column_first_moves_four_units() -> None:
    assert score(BLOCK_ORDER, PAGE, U1, U4, U5, U2, U3, U6) == 1 - 4 / 6


def test_one_paragraph_moved_to_the_end_costs_two_edits() -> None:
    assert score(BLOCK_ORDER, PAGE, U1, U3, U4, U5, U6, U2) == 1 - 2 / 6


def test_a_missing_unit_costs_one_edit() -> None:
    assert score(BLOCK_ORDER, PAGE, U1, U2, U4, U5, U6) == 1 - 1 / 6


def test_markup_case_and_a_typo_do_not_move_a_unit() -> None:
    blocks = ["## ANNUAL RESULTS", U2.replace("budget", "budgt"), U3, U4, U5, U6]
    assert score(BLOCK_ORDER, PAGE, *blocks) == 1.0


def test_a_break_inside_a_paragraph_is_left_to_the_order_tests() -> None:
    t = truth(
        para("a", "The plan was approved. Work starts"),
        para("b", "in June. Costs are fixed.", page=1),
        para("c", "Nothing else changed."),
        relations=[{"kind": "continues_in", "from": "a", "to": "b"}],
    )
    blocks = [
        "in June. Costs are fixed.",
        "The plan was approved. Work starts",
        "Nothing else changed.",
    ]
    pred = Prediction(markdown="\n\n".join(blocks))
    ctx = DocContext(t, pred)
    assert BLOCK_ORDER.score(ctx) == 1.0
    assert ORDER_TESTS.score(ctx) == 0.5


def test_a_unit_printed_twice_counts_as_missing() -> None:
    once = [U1, U2, U3, U4, U5, U6]
    outputs = {
        "late copy": ([*once, U2], 1 - 1 / 6),
        "copy side by side": ([U1, U2, U2, U3, U4, U5, U6], 1 - 1 / 6),
        "late copy with a typo": ([*once, U2[1:]], 1 - 1 / 6),
        "whole output twice": ([*once, *once], 0.0),
    }
    found = {name: score(BLOCK_ORDER, PAGE, *blocks) for name, (blocks, _) in outputs.items()}
    assert found == {name: expected for name, (_, expected) in outputs.items()}
