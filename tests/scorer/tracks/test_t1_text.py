import pytest

from tests.builders import para, truth, typo
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.base import SubMetric
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t1_text import (
    BlockEditSimilarity,
    OrthographyDiacritics,
    SentenceIntegrity,
    WordBagF1,
    best_pairing,
    truth_prose,
)

WORD_BAG = WordBagF1()
SENTENCE = SentenceIntegrity()
ORTHOGRAPHY = OrthographyDiacritics()
BLOCK_EDIT = BlockEditSimilarity()
PROSE_METRICS: tuple[SubMetric, ...] = (WORD_BAG, SENTENCE, BLOCK_EDIT)
ALL_METRICS: tuple[SubMetric, ...] = (*PROSE_METRICS, ORTHOGRAPHY)


def score(metric: SubMetric, t: Groundtruth, markdown: str) -> float:
    return metric.score(DocContext(t, Prediction(markdown=markdown)))


def scores(
    t: Groundtruth, markdown: str, metrics: tuple[SubMetric, ...] = PROSE_METRICS
) -> dict[str, float]:
    ctx = DocContext(t, Prediction(markdown=markdown))
    return {metric.name: metric.score(ctx) for metric in metrics}


def every(metrics: tuple[SubMetric, ...], value: float) -> dict[str, float]:
    return {metric.name: value for metric in metrics}


def split(first: str, second: str, *, hyphenated: bool) -> Groundtruth:
    return truth(
        para("a", first, page=0),
        para("b", second, page=1),
        relations=[{"kind": "continues_in", "from": "a", "to": "b", "hyphenated": hyphenated}],
    )


TABLE_ONLY = truth({"id": "t", "class": "table", "page": 0, "header": ["Q1"], "rows": [["10"]]})
FURNITURE_ONLY = truth({**para("n", "2"), "class": "footer", "channel": "furniture"})
CAT = truth(para("p", "The cat sat"), {**para("n", "2"), "class": "footer", "channel": "furniture"})
SALES = truth(
    para("p", "Sales rose"),
    {"id": "t", "class": "table", "page": 0, "header": ["Q1"], "rows": [["10"]]},
)
BUDGET = truth(para("p", "The budget was approved in March. It passed."))
LETTER = truth(
    para("p", "Please recieve the parcel from Herr Müller."),
    planted=[typo("recieve"), typo("Müller")],
)
COMMITTEE = truth(para("p", "The committee approved the budget."))
TWO = truth(para("a", "A" * 100), para("b", "B" * 50))
WELL_KNOWN = split("It is well-", "known here.", hyphenated=False)
COMPLEX = split("A comp-", "lex case.", hyphenated=True)


def test_groundtruth_as_output_scores_one(demo_groundtruth: Groundtruth) -> None:
    markdown = "\n\n".join(truth_prose(demo_groundtruth))
    assert scores(demo_groundtruth, markdown) == every(PROSE_METRICS, 1.0)


def test_nutrient_fixture_loses_only_the_page_numbers(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult
) -> None:
    ctx = DocContext(demo_groundtruth, NutrientProvider().normalize(demo_nutrient_raw))
    assert {metric.name: metric.score(ctx) for metric in PROSE_METRICS} == {
        "word_bag_f1": pytest.approx(418 / 421),
        "sentence_integrity": 1.0,
        "block_edit_similarity": 1.0,
    }


def test_near_empty_output_scores_zero() -> None:
    assert scores(LETTER, "x", ALL_METRICS) == every(ALL_METRICS, 0.0)


def test_prose_inside_a_table_does_not_count() -> None:
    markdown = "| Please recieve the parcel from Herr Müller. |\n|---|"
    assert scores(LETTER, markdown, ALL_METRICS) == every(ALL_METRICS, 0.0)


def test_applies_only_with_body_prose() -> None:
    applies = {
        metric.name: (
            metric.applies(CAT),
            metric.applies(TABLE_ONLY),
            metric.applies(FURNITURE_ONLY),
        )
        for metric in PROSE_METRICS
    }
    assert applies == {metric.name: (True, False, False) for metric in PROSE_METRICS}


def test_page_number_in_the_body_is_an_extra_word() -> None:
    assert score(WORD_BAG, CAT, "The cat sat 2") == pytest.approx(6 / 7)


def test_missing_words_lower_recall() -> None:
    t = truth(para("p", "The cat sat on the mat"))
    assert score(WORD_BAG, t, "The cat sat") == pytest.approx(2 / 3)


def test_case_and_diacritics_count() -> None:
    t = truth(para("p", "Müller Weg"))
    assert score(WORD_BAG, t, "Müller Weg") == 1.0
    assert score(WORD_BAG, t, "Muller Weg") == pytest.approx(0.5)
    assert score(WORD_BAG, t, "müller Weg") == pytest.approx(0.5)
    assert score(WORD_BAG, truth(para("p", "Müller")), "Mu\u0308ller") == 1.0


def test_a_dropped_combining_mark_changes_the_word() -> None:
    t = truth(para("p", "हिन्दी भाषा"))
    assert score(WORD_BAG, t, "हिन्दी भाषा") == 1.0
    assert score(WORD_BAG, t, "हिन्द भाषा") == pytest.approx(0.5)


def test_superscript_is_formatting_not_a_different_word() -> None:
    t = truth(para("p", "area m²"))
    assert score(WORD_BAG, t, "area m2") == 1.0
    assert score(WORD_BAG, t, "area m<sup>2</sup>") == 1.0
    assert score(WORD_BAG, t, "area m²") == 1.0


def test_order_punctuation_markdown_and_html_are_not_words() -> None:
    assert score(WORD_BAG, CAT, "sat, cat. The!") == 1.0
    markdown = "## **The** <b>cat</b> [sat](https://example.com/table/td)\n\n![a dog](dog.png)"
    assert score(WORD_BAG, CAT, markdown) == 1.0


@pytest.mark.parametrize(
    "markdown",
    [
        "Sales rose\n\nQ1 | 2\n---|---\n10 | 3",
        "Sales rose\n\n<TABLE>\n<tr><td><table><tr><td>Q1</td></tr></table></td></tr>\n</TABLE>",
    ],
)
def test_tables_are_left_out_on_both_sides(markdown: str) -> None:
    assert score(WORD_BAG, SALES, markdown) == 1.0


def test_table_written_as_plain_lines_adds_words() -> None:
    assert score(WORD_BAG, SALES, "Sales rose\n\nQ1\n10") == pytest.approx(2 / 3)


def test_hidden_text_in_the_output_is_an_extra_word() -> None:
    t = truth(para("p", "Total due"), {**para("h", "secret"), "channel": "hidden"})
    assert score(WORD_BAG, t, "Total due secret") == pytest.approx(0.8)


def test_repeating_the_text_halves_precision() -> None:
    assert score(WORD_BAG, CAT, "The cat sat The cat sat") == pytest.approx(2 / 3)


def test_a_broken_sentence_fails() -> None:
    outputs = {
        "page number inside": "The budget was 2 approved in March. It passed.",
        "blank line inside": "The budget was\n\napproved in March. It passed.",
        "running header between pages": (
            "The budget was\n\nAnnual Report 2025\n\napproved in March. It passed."
        ),
        "one wrong word": "The budjet was approved in March. It passed.",
        "case changed": "The budget was approved in March. it passed.",
    }
    found = {name: score(SENTENCE, BUDGET, markdown) for name, markdown in outputs.items()}
    assert found == dict.fromkeys(outputs, 0.5)


def test_line_breaks_markers_and_moved_sentences_keep_sentences_whole() -> None:
    outputs = {
        "line break inside a block": "The budget was\napproved in March.\nIt passed.",
        "markers and punctuation": "## The **budget** was _approved_ in March; It passed!",
        "sentences moved between blocks": "It passed.\n\nThe budget was approved in March.",
    }
    found = {name: score(SENTENCE, BUDGET, markdown) for name, markdown in outputs.items()}
    assert found == dict.fromkeys(outputs, 1.0)


def test_split_word_fails_the_sentence() -> None:
    t = truth(
        para("a", "The comp-", page=0),
        para("b", "lex case works.", page=1),
        relations=[{"kind": "continues_in", "from": "a", "to": "b", "hyphenated": True}],
    )
    assert score(SENTENCE, t, "The complex case works.") == 1.0
    assert score(SENTENCE, t, "The comp- lex case works.") == 0.0
    assert score(SENTENCE, t, "The comp-\nlex case works.") == 0.0


def test_whole_words_only() -> None:
    t = truth(para("p", "cat sat."))
    assert score(SENTENCE, t, "the cats sat.") == 0.0


def test_split_after_an_abbreviation_fails_the_sentence() -> None:
    t = truth(para("p", "Dr. Smith approved the plan."))
    assert score(SENTENCE, t, "Dr. Smith approved the plan.") == 1.0
    assert score(SENTENCE, t, "Dr.\n\nSmith approved the plan.") == 0.0


def test_copying_the_page_passes_every_item() -> None:
    outputs = {
        "copied": "Please recieve the parcel from Herr Müller.",
        "decomposed diacritic": "Please recieve the parcel from Herr Mu\u0308ller.",
        "markdown markers": "Please **recieve** the parcel from _Herr Müller_.",
    }
    found = {name: score(ORTHOGRAPHY, LETTER, markdown) for name, markdown in outputs.items()}
    assert found == dict.fromkeys(outputs, 1.0)


def test_each_changed_item_fails() -> None:
    outputs = {
        "typo fixed": ("Please receive the parcel from Herr Müller.", 0.5),
        "diacritic dropped": ("Please recieve the parcel from Herr Muller.", 0.5),
        "both fixed": ("Please receive the parcel from Herr Muller.", 0.0),
        "word dropped": ("Please take the parcel from Herr Müller.", 0.5),
        "case changed": ("Please Recieve the parcel from Herr Müller.", 0.5),
        "inside a longer word": ("Please recieved the parcel from Herr Müller.", 0.5),
    }
    found = {name: score(ORTHOGRAPHY, LETTER, markdown) for name, (markdown, _) in outputs.items()}
    assert found == {name: expected for name, (_, expected) in outputs.items()}


def test_fixing_one_of_two_occurrences_fails() -> None:
    t = truth(para("p", "Herr Müller called. Müller will visit."), planted=[typo("Müller")])
    assert score(ORTHOGRAPHY, t, "Herr Müller called. Müller will visit.") == 1.0
    assert score(ORTHOGRAPHY, t, "Herr Muller called. Müller will visit.") == 0.0
    assert score(ORTHOGRAPHY, t, "Herr Müller called. Müller will visit. Müller") == 1.0


def test_near_miss_name_over_two_words_needs_one_block() -> None:
    t = truth(para("p", "Signed, Jon Smiht"), planted=[typo("Jon Smiht")])
    assert score(ORTHOGRAPHY, t, "Signed, Jon Smiht") == 1.0
    assert score(ORTHOGRAPHY, t, "Signed, Jon Smith") == 0.0
    assert score(ORTHOGRAPHY, t, "Signed, Jon\n\nSmiht") == 0.0


def test_applies_only_with_planted_typos() -> None:
    assert ORTHOGRAPHY.applies(LETTER)
    assert not ORTHOGRAPHY.applies(truth(para("p", "No traps here.")))


def test_trap_split_over_a_page_break_is_accepted_and_scored() -> None:
    t = truth(
        para("a", "Please ask Herr Mül-", page=0),
        para("b", "ler for the parcel.", page=1),
        relations=[{"kind": "continues_in", "from": "a", "to": "b", "hyphenated": True}],
        planted=[typo("Müller")],
    )
    assert score(ORTHOGRAPHY, t, "Please ask Herr Müller for the parcel.") == 1.0
    assert score(ORTHOGRAPHY, t, "Please ask Herr Muller for the parcel.") == 0.0


def test_one_missing_letter() -> None:
    assert score(BLOCK_EDIT, COMMITTEE, "The comittee approved the budget.") == pytest.approx(
        1 - 1 / 34
    )


def test_missing_block_is_weighted_by_its_length() -> None:
    assert score(BLOCK_EDIT, TWO, "A" * 100) == pytest.approx(2 / 3)


def test_split_paragraph_pairs_with_one_half_only() -> None:
    t = truth(para("p", "one two three four five six seven eight nine ten"))
    split_output = score(BLOCK_EDIT, t, "one two three four five six\n\n2\n\nseven eight nine ten")
    assert split_output == pytest.approx(27 / 48)
    assert score(BLOCK_EDIT, t, "one two three four five six seven eight nine ten") == 1.0


def test_two_paragraphs_merged_into_one_block() -> None:
    merged = score(BLOCK_EDIT, TWO, "A" * 100 + " " + "B" * 50)
    assert merged == pytest.approx(100 * (100 / 151) / 150)


def test_below_the_minimum_similarity_is_missing() -> None:
    assert score(BLOCK_EDIT, COMMITTEE, "Budget.") == 0.0


def test_markdown_markers_are_not_edits() -> None:
    t = truth(
        {"id": "h", "class": "heading", "page": 0, "text": "Budget 2025", "level": 2},
        para("p", "The committee approved the budget."),
        {"id": "l", "class": "list_item", "page": 0, "text": "First item"},
    )
    markdown = "## Budget 2025\n\nThe **committee** approved the _budget_.\n\n- First item"
    assert score(BLOCK_EDIT, t, markdown) == 1.0


def test_block_order_does_not_count() -> None:
    assert score(BLOCK_EDIT, TWO, "B" * 50 + "\n\n" + "A" * 100) == 1.0


def test_curly_quotes_are_not_edits() -> None:
    t = truth(para("p", 'He said "yes" and left.'))
    assert score(BLOCK_EDIT, t, "He said \u201cyes\u201d and left.") == 1.0


def test_pair_below_the_minimum_does_not_steer_the_pairing() -> None:
    assert best_pairing([[0.9, 0.6], [0.6, 0.45]], [1, 1]) == pytest.approx([0.6, 0.6])


def test_long_blocks_steer_the_pairing() -> None:
    assert best_pairing([[0.9, 0.6], [1.0, 0.5]], [100, 10]) == pytest.approx([0.9, 0.5])


def test_list_items_on_consecutive_lines_pair_one_by_one() -> None:
    t = truth(
        {"id": "h", "class": "heading", "page": 0, "text": "Steps", "level": 2},
        {"id": "a", "class": "list_item", "page": 0, "text": "Open the valve slowly"},
        {"id": "b", "class": "list_item", "page": 0, "text": "Check the pressure gauge"},
    )
    markdown = "## Steps\n- Open the valve slowly\n- Check the pressure gauge"
    assert score(BLOCK_EDIT, t, markdown) == 1.0


def test_each_output_block_has_one_partner_at_most() -> None:
    t = truth(para("a", "Same text here."), para("b", "Same text here."))
    assert score(BLOCK_EDIT, t, "Same text here.") == pytest.approx(0.5)


def test_compound_kept_or_joined_scores_one() -> None:
    assert scores(WELL_KNOWN, "It is well-known here.") == every(PROSE_METRICS, 1.0)
    assert scores(WELL_KNOWN, "It is wellknown here.") == every(PROSE_METRICS, 1.0)


def test_compound_split_at_the_page_still_fails_the_sentence() -> None:
    word_bag, sentence, block = scores(WELL_KNOWN, "It is well-\n\n2\n\nknown here.").values()
    assert sentence == 0.0
    assert word_bag == pytest.approx(10 / 11)
    assert block == pytest.approx(11 / 21)


def test_syllable_break_accepts_only_the_joined_word() -> None:
    assert scores(COMPLEX, "A complex case.") == every(PROSE_METRICS, 1.0)
    word_bag, sentence, block = scores(COMPLEX, "A comp-lex case.").values()
    assert sentence == 0.0
    assert word_bag < 1.0
    assert block < 1.0


def test_two_compounds_give_every_combination() -> None:
    t = truth(
        para("a", "a well-", page=0),
        para("b", "known and self-", page=1),
        para("c", "made one", page=2),
        relations=[
            {"kind": "continues_in", "from": "a", "to": "b"},
            {"kind": "continues_in", "from": "b", "to": "c"},
        ],
    )
    combinations = [
        "a well-known and self-made one",
        "a wellknown and self-made one",
        "a well-known and selfmade one",
        "a wellknown and selfmade one",
    ]
    found = {text: score(WORD_BAG, t, text) for text in combinations}
    assert found == dict.fromkeys(combinations, 1.0)
