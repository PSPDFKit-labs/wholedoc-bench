from typing import Any

import pytest

from tests.builders import A, B, C, D, E, without_marks
from wholedoc_bench.schema import Groundtruth
from wholedoc_bench.scorer.block_integrity import (
    BlockIntegrityResult,
    JunctionOutcome,
    OutputBlock,
    has_blocks,
    score_block_integrity,
    truth_blocks,
)
from wholedoc_bench.scorer.tracks.t3_structure import BlockIntegrity
from wholedoc_bench.scorer.utils.prose import edit_similarity

LONG = f"{A} {B} {C}"
SHORT = "Minutes were approved without changes."


def element(
    id_: str, page: int, text: str, cls: str = "paragraph", channel: str = "body"
) -> dict[str, Any]:
    return {"id": id_, "class": cls, "page": page, "text": text, "channel": channel}


def link(kind: str, source: str, target: str) -> dict[str, str]:
    return {"kind": kind, "from": source, "to": target}


def truth(
    elements: list[dict[str, Any]],
    relations: list[dict[str, str]] | None = None,
    pages: int = 1,
    reading_order: list[str] | None = None,
) -> Groundtruth:
    return Groundtruth.model_validate(
        {
            "schema_version": "0.1",
            "document": {
                "id": "doc",
                "source": "born-digital",
                "groundtruth_source": "generator",
                "language": ["en"],
                "pages": [{"index": i, "width": 595.0, "height": 842.0} for i in range(pages)],
            },
            "elements": elements,
            "relations": relations or [],
            "reading_order": reading_order or [],
        }
    )


def blocks(*texts: str) -> list[OutputBlock]:
    return [OutputBlock(text) for text in texts]


FOUR = truth([element(name, 0, text) for name, text in (("a", A), ("b", B), ("c", C), ("d", D))])


def demo_texts(demo: Groundtruth) -> dict[str, str]:
    return {e.id: e.text for e in demo.elements}


def test_split_on_one_page_is_a_miss_f1_0_667() -> None:
    half = len(B) // 2
    result = score_block_integrity(FOUR, blocks(A, B[:half], B[half:], C, D))
    assert result.precision == pytest.approx(0.6)
    assert result.recall == pytest.approx(0.75)
    assert result.f1 == pytest.approx(2 / 3)


def test_joined_across_the_page_break_scores_one(demo_groundtruth: Groundtruth) -> None:
    t = demo_texts(demo_groundtruth)
    joined = t["e7"].removesuffix("-") + t["e9"]
    result = score_block_integrity(demo_groundtruth, blocks(t["e2"], joined))
    assert result.f1 == 1.0
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.JOINED]


def test_one_block_per_page_is_a_merge_f1_0() -> None:
    page = truth([element("a", 0, A), element("b", 0, B), element("c", 0, C)])
    result = score_block_integrity(page, blocks(f"{A} {B} {C}"))
    assert result.f1 == 0.0
    assert result.merged_blocks == (0,)


def test_long_and_short_joined_across_a_keep_apart_break_both_miss() -> None:
    merged = f"{LONG} {SHORT}"
    assert edit_similarity(LONG, merged) >= 0.9
    doc = truth(
        [element("long", 0, LONG), element("short", 1, SHORT)],
        [link("stays_apart", "long", "short")],
        pages=2,
    )
    result = score_block_integrity(doc, blocks(merged))
    assert result.f1 == 0.0
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.JOINED]


def test_always_merge_across_every_page_break() -> None:
    doc = truth(
        [
            element("a", 0, A),
            element("b", 0, B),
            element("c", 1, C),
            element("d", 1, D),
            element("e", 2, E),
        ],
        [link("continues_in", "b", "c"), link("stays_apart", "d", "e")],
        pages=3,
    )
    right = score_block_integrity(doc, blocks(A, f"{B} {C}", D, E))
    assert right.f1 == 1.0
    assert [j.outcome for j in right.junctions] == [
        JunctionOutcome.JOINED,
        JunctionOutcome.KEPT_APART,
    ]
    always = score_block_integrity(doc, blocks(A, f"{B} {C}", f"{D} {E}"))
    assert always.f1 == pytest.approx(4 / 7)
    assert [j.outcome for j in always.junctions] == [
        JunctionOutcome.JOINED,
        JunctionOutcome.JOINED,
    ]


@pytest.mark.parametrize(
    ("elements", "applies"),
    [
        ([element("a", 0, A)], True),
        ([element("i", 0, A, cls="list_item")], True),
        ([element("h", 0, "Budget", cls="heading"), element("t", 0, "", cls="table")], False),
        ([element("n", 0, "1", cls="footer", channel="furniture")], False),
    ],
)
def test_applies_when_there_is_a_body_paragraph_or_list_item(
    elements: list[dict[str, Any]], applies: bool
) -> None:
    assert BlockIntegrity().applies(truth(elements)) is applies


def test_short_paragraph_words_inside_a_long_one_are_not_a_merge() -> None:
    long = (
        "Revenue grew in every region; see Table 1. Costs also rose, mainly because of new "
        "hires in the second half of the year."
    )
    doc = truth([element("long", 0, long), element("short", 0, "See Table 1.")])
    result = score_block_integrity(doc, blocks(long))
    assert result.merged_blocks == ()
    assert result.f1 == pytest.approx(2 / 3)


REGIONS = "Sales in the North rose while sales in the South fell sharply."


@pytest.mark.parametrize(
    "written",
    [
        pytest.param(REGIONS.removesuffix("."), id="final_period_dropped"),
        pytest.param(REGIONS.replace("sharply", "sharpy"), id="one_letter_dropped"),
    ],
)
def test_block_slightly_shorter_than_its_paragraph_still_hides_short_items_inside(
    written: str,
) -> None:
    doc = truth(
        [
            element("p", 0, REGIONS),
            element("n", 0, "North", cls="list_item"),
            element("s", 0, "South", cls="list_item"),
        ]
    )
    result = score_block_integrity(doc, blocks(written, "North", "South"))
    assert result.merged_blocks == ()
    assert result.f1 == 1.0


def test_block_holding_a_small_part_of_a_much_longer_paragraph_does_not_hide_a_merge() -> None:
    first, second = "Sales in the North rose.", "Sales in the South fell."
    long = f"{first} {second} Costs also rose, mainly because of new hires late in the year."
    doc = truth([element("long", 0, long), element("a", 0, first), element("b", 0, second)])
    result = score_block_integrity(doc, blocks(long, f"{first} {second}"))
    assert result.merged_blocks == (1,)


def test_identical_paragraphs_match_identical_blocks() -> None:
    doc = truth([element("a1", 0, A), element("a2", 0, A)])
    result = score_block_integrity(doc, blocks(A, A))
    assert result.f1 == 1.0
    assert result.merged_blocks == ()


def test_truth_blocks_join_the_hyphenated_continuation(demo_groundtruth: Groundtruth) -> None:
    found = truth_blocks(demo_groundtruth)
    assert [b.element_ids for b in found] == [("e2",), ("e7", "e9")]
    assert found[1].pages == (1, 2)
    assert "deliberately complex example" in found[1].text
    assert found[1].variants == (found[1].text,)


def test_keep_apart_pair_that_is_missing_is_not_kept_apart() -> None:
    doc = truth(
        [element("c", 0, C), element("long", 0, LONG), element("short", 1, SHORT)],
        [link("stays_apart", "long", "short")],
        pages=2,
    )
    result = score_block_integrity(doc, blocks(C))
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.MISSING]


def test_same_page_column_continuation() -> None:
    doc = truth(
        [element("col1", 0, A), element("col2", 0, B)],
        [link("continues_in", "col1", "col2")],
        reading_order=["col1", "col2"],
    )
    joined = score_block_integrity(doc, blocks(f"{A} {B}"))
    assert joined.f1 == 1.0
    assert [(j.junction.same_page, j.outcome) for j in joined.junctions] == [
        (True, JunctionOutcome.JOINED)
    ]
    split = score_block_integrity(doc, blocks(A, B))
    assert split.f1 == 0.0
    assert [j.outcome for j in split.junctions] == [JunctionOutcome.SPLIT]


def test_extra_joined_copy_next_to_separate_blocks() -> None:
    doc = truth(
        [element("a", 0, A), element("b", 1, B)],
        [link("stays_apart", "a", "b")],
        pages=2,
    )
    result = score_block_integrity(doc, blocks(A, B, f"{A} {B}"))
    assert result.f1 == pytest.approx(0.8)
    assert result.merged_blocks == (2,)
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.JOINED]


NOTE = "Source: annual report of the committee, 2024."


def test_caption_written_as_a_paragraph_does_not_count_against_precision(
    demo_groundtruth: Groundtruth,
) -> None:
    t = demo_texts(demo_groundtruth)
    joined = t["e7"].removesuffix("-") + t["e9"]
    right = score_block_integrity(demo_groundtruth, blocks(t["e2"], t["e3"], joined))
    assert right.f1 == 1.0
    assert right.counted_blocks == (0, 2)
    split = score_block_integrity(demo_groundtruth, blocks(t["e2"], t["e3"], t["e7"], t["e9"]))
    assert split.f1 == pytest.approx(0.4)


def test_near_copy_of_a_caption_does_not_count() -> None:
    caption, near = "Table 1: Sample records", "Table 1. Sample records"
    assert edit_similarity(caption, near) >= 0.9
    doc = truth([element("a", 0, A), element("c", 0, caption, cls="caption")])
    assert score_block_integrity(doc, blocks(A, near)).f1 == 1.0


def test_extra_block_that_copies_nothing_still_counts() -> None:
    doc = truth([element("a", 0, A), element("c", 0, "Table 1: Sample records", cls="caption")])
    result = score_block_integrity(doc, blocks(A, "Figure 3: Travel costs by quarter"))
    assert result.f1 == pytest.approx(2 / 3)


def test_continuation_inside_a_merged_block_is_merged_not_joined(
    demo_groundtruth: Groundtruth,
) -> None:
    t = demo_texts(demo_groundtruth)
    whole = f"{t['e2']} {t['e7'].removesuffix('-')}{t['e9']}"
    result = score_block_integrity(demo_groundtruth, blocks(whole))
    assert result.f1 == 0.0
    assert result.merged_blocks == (0,)
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.MERGED]


def test_chain_split_at_its_second_break() -> None:
    doc = truth(
        [element("a", 0, A), element("b", 1, B), element("c", 2, C)],
        [link("continues_in", "a", "b"), link("continues_in", "b", "c")],
        pages=3,
    )
    result = score_block_integrity(doc, blocks(f"{A} {B}", C))
    assert result.f1 == 0.0
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.JOINED, JunctionOutcome.SPLIT]


def test_part_of_a_chain_glued_to_another_paragraph_is_merged() -> None:
    doc = truth(
        [element("a", 0, A), element("b", 1, B), element("d", 1, D), element("c", 2, C)],
        [link("continues_in", "a", "b"), link("continues_in", "b", "c")],
        pages=3,
    )
    result = score_block_integrity(doc, blocks(f"{A} {B} {D}", C))
    assert result.merged_blocks == ()
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.MERGED, JunctionOutcome.SPLIT]


def test_copy_of_text_that_is_not_printed_body_text_still_counts() -> None:
    doc = truth([element("a", 0, A), element("x", 0, NOTE, channel="hidden")])
    assert score_block_integrity(doc, blocks(A, NOTE)).f1 == pytest.approx(2 / 3)


def test_a_few_characters_of_the_next_paragraph_make_a_merge() -> None:
    a, b = "A" * 200, "B" * 100
    doc = truth([element("a", 0, a), element("b", 0, b)])
    result = score_block_integrity(doc, blocks(a + "B" * 5, "B" * 95))
    assert result.merged_blocks == (0,)
    assert result.f1 == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("output", "merged"),
    [
        ((f"{A} Several", B.removeprefix("Several ")), (0,)),
        ((A.removesuffix(" figures."), f"figures. {B}"), (1,)),
        pytest.param(
            (f"{A.replace('final ', '')} Sev", B.removeprefix("Sev")), (0,), id="a_lost_letters"
        ),
        pytest.param(
            (A.removesuffix(" figures."), f"res. {B.replace('clearer ', '')}"),
            (1,),
            id="b_lost_letters",
        ),
        pytest.param(
            (f"{A.removesuffix('es.')} Several", B.removeprefix("Several ")),
            (0,),
            id="a_lost_its_last_letters",
        ),
        pytest.param(
            (f"{A.removesuffix(' figures.')} Several", B.removeprefix("Several ")),
            (0,),
            id="a_lost_its_last_word",
        ),
        pytest.param(
            (A.removesuffix(" figures."), f"figures. {B.removeprefix('Se')}"),
            (1,),
            id="b_lost_its_first_letters",
        ),
    ],
)
def test_one_word_on_the_wrong_side_of_a_page_break(
    output: tuple[str, str], merged: tuple[int, ...]
) -> None:
    doc = truth([element("a", 0, A), element("b", 1, B)], [link("stays_apart", "a", "b")], pages=2)
    result = score_block_integrity(doc, blocks(*output))
    assert result.merged_blocks == merged
    assert result.f1 == pytest.approx(0.5)
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.JOINED]


def test_a_changed_word_before_a_shared_last_word_is_no_merge() -> None:
    opening = "The company completed its annual review and confirmed the final results. "
    after = "demand rose. Further details are provided in the attached report."
    doc = truth([element("a", 0, f"{opening}profits rose."), element("b", 0, after)])
    result = score_block_integrity(doc, blocks(f"{opening}demand rose.", after))
    assert result.merged_blocks == ()
    assert result.f1 == 1.0


@pytest.mark.parametrize("extra", ["travel costs", "S"])
def test_extra_text_that_is_not_a_neighbour_edge_is_no_merge(extra: str) -> None:
    doc = truth([element("a", 0, A), element("b", 0, B)])
    result = score_block_integrity(doc, blocks(f"{A} {extra}", B))
    assert result.merged_blocks == ()
    assert result.f1 == 1.0


def test_each_caption_excuses_one_copy() -> None:
    doc = truth(
        [element("p", 0, "Table 1: Sales"), element("c", 0, "Table 1: Sales", cls="caption")]
    )
    assert score_block_integrity(doc, blocks(*["Table 1: Sales"] * 5)).f1 == pytest.approx(0.4)


def test_one_page_number_per_page_is_excused() -> None:
    doc = truth(
        [
            element("a", 0, A),
            element("n1", 0, "1", cls="footer", channel="furniture"),
            element("b", 1, B),
            element("n2", 1, "2", cls="footer", channel="furniture"),
        ],
        pages=2,
    )
    assert score_block_integrity(doc, blocks(A, "1", B, "2")).f1 == 1.0
    assert score_block_integrity(doc, blocks(A, "1", "1", B)).f1 == pytest.approx(0.8)


def test_continuation_with_the_next_paragraphs_first_word_is_merged() -> None:
    doc = truth(
        [element("a", 0, A), element("b", 1, B), element("c", 1, C)],
        [link("continues_in", "a", "b")],
        pages=2,
    )
    result = score_block_integrity(doc, blocks(f"{A} {B} The", C.removeprefix("The ")))
    assert result.merged_blocks == (0,)
    assert result.f1 == pytest.approx(0.5)
    assert [j.outcome for j in result.junctions] == [JunctionOutcome.MERGED]


def test_exact_output_of_near_duplicate_paragraphs_scores_one() -> None:
    b, c, a = B, f"New {C}", f"{B} New"
    doc = truth([element("b", 0, b), element("c", 0, c), element("a", 0, a)])
    result = score_block_integrity(doc, blocks(b, c, a))
    assert result.merged_blocks == ()
    assert result.f1 == 1.0


def _compound(first: str, second: str) -> Groundtruth:
    return truth(
        [element("a", 0, first), element("b", 1, second)],
        [link("continues_in", "a", "b")],
        pages=2,
    )


def test_compound_word_split_at_a_page_accepts_both_spellings() -> None:
    doc = _compound("A well-", "known fact.")
    assert [b.variants for b in truth_blocks(doc)] == [("A well-known fact.", "A wellknown fact.")]
    assert score_block_integrity(doc, blocks("A well-known fact.")).f1 == 1.0
    assert score_block_integrity(doc, blocks("A wellknown fact.")).f1 == 1.0


def test_accepted_spelling_counts_in_the_merge_check() -> None:
    doc = truth(
        [element("a", 0, "co-"), element("b", 1, "op"), element("c", 1, "Short note here.")],
        [link("continues_in", "a", "b")],
        pages=2,
    )
    assert score_block_integrity(doc, blocks("coop Short note here.")).merged_blocks == (0,)


COMPOUNDS = ["A well-", "known, long-", "term, high-", "quality, low-", "cost plan."]
NEXT = "The committee reviewed the full report and approved the final findings."


def _compound_chain() -> Groundtruth:
    elements = [element(f"p{k}", k, text) for k, text in enumerate(COMPOUNDS)]
    return truth(
        [*elements, element("n", 4, NEXT)],
        [link("continues_in", f"p{k}", f"p{k + 1}") for k in range(len(COMPOUNDS) - 1)],
        pages=len(COMPOUNDS),
    )


def test_misplaced_break_is_caught_with_the_accepted_spelling() -> None:
    joined = "A wellknown, longterm, highquality, lowcost plan."
    result = score_block_integrity(
        _compound_chain(), blocks(f"{joined} The", NEXT.removeprefix("The "))
    )
    assert result.merged_blocks == (0,)
    assert result.f1 == pytest.approx(0.5)
    assert score_block_integrity(_compound_chain(), blocks(joined, NEXT)).f1 == 1.0


def test_glued_word_matches_the_neighbours_accepted_spelling() -> None:
    doc = truth(
        [element("a", 0, A), element("b", 1, "co-"), element("c", 1, "op members met.")],
        [link("continues_in", "b", "c")],
        pages=2,
    )
    result = score_block_integrity(doc, blocks(f"{A} coop", "members met."))
    assert result.merged_blocks == (0,)


def _outcomes(doc: Groundtruth, *texts: str) -> list[JunctionOutcome]:
    return [j.outcome for j in score_block_integrity(doc, blocks(*texts)).junctions]


def _word_split(first: str, second: str) -> Groundtruth:
    hyphenated = {**link("continues_in", "a", "b"), "hyphenated": True}
    return truth([element("a", 0, first), element("b", 1, second)], [hyphenated], pages=2)


@pytest.mark.parametrize(
    ("doc", "joined", "split"),
    [
        (_compound("co-", "op"), ["coop"], ["co-", "op"]),
        (_word_split("comp-", "lex"), ["complex"], ["comp-", "lex"]),
    ],
)
def test_break_results_accept_a_dropped_edge_hyphen(
    doc: Groundtruth, joined: list[str], split: list[str]
) -> None:
    assert _outcomes(doc, *joined) == [JunctionOutcome.JOINED]
    assert _outcomes(doc, *split) == [JunctionOutcome.SPLIT]


def test_break_results_of_a_compound_chain_in_its_accepted_spelling() -> None:
    joined = "A wellknown, longterm, highquality, lowcost plan."
    printed = "A well-known, long-term, high-quality, low-cost plan."
    assert _outcomes(_compound_chain(), joined, NEXT) == [JunctionOutcome.JOINED] * 4
    assert _outcomes(_compound_chain(), printed, NEXT) == [JunctionOutcome.JOINED] * 4


def _changed(text: str, positions: range) -> str:
    return "".join("#" if k in positions else ch for k, ch in enumerate(text))


def test_pairing_matches_the_most_blocks_not_the_strongest_first() -> None:
    p1 = "".join(chr(ord("a") + (k * 7) % 26) for k in range(100))
    p2 = _changed(p1, range(10, 16))
    x1 = _changed(p1, range(50, 54))
    x2 = _changed(p1, range(60, 69))
    assert edit_similarity(p1, x1) == pytest.approx(0.96)
    assert edit_similarity(p2, x1) == pytest.approx(0.90)
    assert edit_similarity(p1, x2) == pytest.approx(0.91)
    assert edit_similarity(p2, x2) < 0.9
    doc = truth([element("p1", 0, p1), element("p2", 0, p2)])
    result = score_block_integrity(doc, blocks(x1, x2))
    assert result.merged_blocks == ()
    assert result.matches == ((0, 1), (1, 0))
    assert result.f1 == 1.0


def test_document_without_blocks_is_an_error() -> None:
    doc = truth([element("h", 0, "Budget", cls="heading")])
    with pytest.raises(ValueError, match="no body paragraph or list item"):
        score_block_integrity(doc, blocks("Budget"))


def _rule(t: Groundtruth, *texts: str) -> BlockIntegrityResult:
    return score_block_integrity(t, blocks(*texts), normalize=without_marks)


def test_a_paragraph_with_no_text_after_cleaning_is_not_a_target() -> None:
    t = truth([element("a", 0, A), element("s", 0, "* * *"), element("b", 0, B)])
    result = _rule(t, A, "* * *", B)
    assert [block.text for block in result.truth_blocks] == [A, B]
    assert result.f1 == 1.0
    assert not has_blocks(truth([element("s", 0, "* * *")]), normalize=without_marks)


def test_a_link_to_a_piece_with_no_text_after_cleaning_gets_no_break_result() -> None:
    t = truth(
        [element("s", 0, "* * *"), element("b", 1, B)],
        [link("stays_apart", "s", "b")],
        pages=2,
    )
    result = _rule(t, "* * *", B)
    assert result.f1 == 1.0
    assert result.junctions == ()


def test_a_continued_paragraph_keeps_its_text_when_one_piece_has_none() -> None:
    t = truth(
        [element("a1", 0, A), element("a2", 1, "* * *"), element("b", 1, B)],
        [link("continues_in", "a1", "a2")],
        pages=2,
    )
    result = _rule(t, f"{A} * * *", B)
    assert [block.text for block in result.truth_blocks] == [A, B]
    assert result.f1 == 1.0
    assert result.junctions == ()
