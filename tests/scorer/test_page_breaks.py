from typing import Any

import pytest

from tests.builders import A, B, C, D, E, groundtruth
from wholedoc_bench.schema import Groundtruth, Relation
from wholedoc_bench.scorer.block_integrity import OutputBlock, score_block_integrity
from wholedoc_bench.scorer.page_breaks import failed_page_breaks, has_page_breaks, score_page_breaks
from wholedoc_bench.scorer.utils.prose import normalize


def piece(id_: str, text: str, page: int = 0, **extra: Any) -> dict[str, Any]:
    return {"id": id_, "class": "paragraph", "page": page, "text": text, **extra}


def relation(kind: str, before: str, after: str, **extra: Any) -> Relation:
    return Relation.model_validate({"kind": kind, "from": before, "to": after, **extra})


def joined_truth(*extra: dict[str, Any]) -> Groundtruth:
    return groundtruth(
        [piece("a", A), piece("b", B, 1), *extra], [relation("continues_in", "a", "b")]
    )


def score(truth: Groundtruth, *texts: str):
    blocks = [OutputBlock(text) for text in texts]
    block_result = score_block_integrity(truth, blocks, normalize=normalize)
    return score_page_breaks(truth, blocks, block_result)


def link(result, index: int = 0) -> dict[str, Any]:
    return result.details["links"][index]


def test_correct_join_has_two_recovered_ranges_and_no_reasons() -> None:
    result = score(joined_truth(), A + " " + B)
    assert result.score == 1
    assert result.details["status"] == "scored"
    assert result.details["continues_in"] == {"passed": 1, "total": 1, "rate": 1}
    assert link(result)["reasons"] == []
    assert [(e["start"], e["end"]) for e in link(result)["evidence"]] == [
        (0, len(A)),
        (len(A) + 1, len(A) + 1 + len(B)),
    ]


def test_split_and_missing_pieces_stay_in_the_denominator() -> None:
    split = score(joined_truth(), A, B)
    missing = score(joined_truth(), A)
    assert split.score == missing.score == 0
    assert link(split)["reasons"] == ["split"]
    assert link(missing)["reasons"] == ["missing"]
    assert len(link(missing)["evidence"]) == 1
    assert link(missing)["evidence"][0]["source_ids"] == ["a"]
    assert missing.details["continues_in"]["total"] == 1


def test_reversed_pieces_in_one_block_do_not_pass() -> None:
    result = score(joined_truth(), B + " " + A)
    assert result.score == 0
    assert "misplaced_break" in link(result)["reasons"]


def test_known_source_text_inserted_at_join_fails() -> None:
    footer = "Report footer"
    truth = joined_truth(piece("footer", footer, channel="furniture"))
    result = score(truth, A + " " + footer + " " + B)
    assert result.score == 0
    assert "extra_source_text" in link(result)["reasons"]
    assert any(e["source_ids"] == ["footer"] for e in link(result)["evidence"])


def test_source_content_from_other_pages_is_not_join_contamination() -> None:
    truth = joined_truth(piece("footer", "12", 2, channel="furniture"))
    assert score(truth, A + " 12 " + B).score == 1


def test_genuine_number_and_number_inside_larger_word_are_not_contamination() -> None:
    truth = groundtruth(
        [piece("a", A + " 12"), piece("b", B, 1), piece("footer", "12", channel="furniture")],
        [relation("continues_in", "a", "b")],
    )
    assert score(truth, A + " 12 " + B).score == 1
    assert (
        score(joined_truth(piece("footer", "12", channel="furniture")), A + " 312 " + B).score == 1
    )
    result = score(truth, A + " 12 12 " + B)
    assert result.score == 0
    assert "extra_source_text" in link(result)["reasons"]


def test_partial_footer_number_is_the_documented_whole_phrase_limit() -> None:
    truth = joined_truth(piece("footer", "Page 12 of 40", channel="furniture"))
    assert score(truth, A + " 12 " + B).score == 1


@pytest.mark.parametrize("extra", [[A, B], [A + " " + B]])
def test_correct_join_does_not_rescue_extra_split_or_joined_copy(extra: list[str]) -> None:
    result = score(joined_truth(), A + " " + B, *extra)
    assert result.score == 0
    assert "duplicate" in link(result)["reasons"]
    assert any(e["role"] == "extra" for e in link(result)["evidence"])
    if extra == [A, B]:
        assert "conflicting_copy" in link(result)["reasons"]


def test_duplicate_joined_copies_inside_one_block_are_separate_occurrences() -> None:
    joined = A + " " + B
    result = score(joined_truth(), joined + " " + joined)
    assert result.score == 0
    extras = [e for e in link(result)["evidence"] if e["role"] == "extra"]
    assert len(extras) == 1
    assert extras[0]["start"] == len(joined) + 1
    assert extras[0]["end"] == 2 * len(joined) + 1


def test_genuine_repeated_source_paragraphs_reserve_two_legitimate_occurrences() -> None:
    truth = groundtruth(
        [piece("a", A), piece("b", B, 1), piece("c", A), piece("d", B, 1)],
        [relation("continues_in", "a", "b"), relation("continues_in", "c", "d")],
    )
    joined = A + " " + B
    assert score(truth, joined, joined).score == 1
    repeated = score(truth, joined, joined, joined)
    assert repeated.score == 0
    assert all("duplicate" in outcome["reasons"] for outcome in repeated.details["links"])


def test_extra_copy_fails_only_the_indistinguishable_text_group() -> None:
    truth = groundtruth(
        [piece("a", A), piece("b", B, 1), piece("c", C), piece("d", D, 1)],
        [relation("continues_in", "a", "b"), relation("continues_in", "c", "d")],
    )
    result = score(truth, A + " " + B, A + " " + B, C + " " + D)
    assert result.score == 0.5
    assert not link(result)["passed"]
    assert link(result, 1)["passed"]


def test_three_piece_chain_can_pass_first_break_and_fail_second() -> None:
    truth = groundtruth(
        [piece("a", A), piece("b", B, 1), piece("c", C, 2)],
        [relation("continues_in", "a", "b"), relation("continues_in", "b", "c")],
    )
    result = score(truth, A + " " + B, C)
    assert result.score == 0.5
    assert link(result)["passed"]
    assert link(result, 1)["reasons"] == ["split"]


def test_whole_document_in_one_block_fails_join_and_keep_apart() -> None:
    truth = groundtruth(
        [piece("a", A), piece("b", B, 1), piece("c", C), piece("d", D, 1)],
        [relation("continues_in", "a", "b"), relation("stays_apart", "c", "d")],
    )
    result = score(truth, " ".join([A, B, C, D]))
    assert result.score == 0
    assert all("merged" in outcome["reasons"] for outcome in result.details["links"])


def test_keep_apart_requires_distinct_matches_and_rejects_joined_extra_copy() -> None:
    truth = groundtruth([piece("a", A), piece("b", B, 1)], [relation("stays_apart", "a", "b")])
    assert score(truth, A, B).score == 1
    assert score(truth, A + " " + B).score == 0
    assert score(truth, A, B, A + " " + B).score == 0
    assert score(truth, A).score == 0


def test_keep_apart_rejects_minimum_three_character_neighbour_edge() -> None:
    truth = groundtruth([piece("a", A), piece("b", B, 1)], [relation("stays_apart", "a", "b")])
    result = score(truth, A + " Sev", B)
    assert result.score == 0
    assert "misplaced_break" in link(result)["reasons"]


def test_neighbour_edge_rejected_by_block_integrity_also_fails_join() -> None:
    truth = joined_truth(piece("c", C, 1))
    result = score(truth, A + " " + B + " The", C[4:])
    assert result.score == 0
    assert "misplaced_break" in link(result)["reasons"]


def test_longer_unlinked_paragraph_owns_embedded_short_link_endpoint() -> None:
    truth = groundtruth(
        [piece("a", "Tax"), piece("b", B, 1), piece("large", "Tax " + A)],
        [relation("continues_in", "a", "b")],
    )
    result = score(truth, "Tax " + A, B)
    assert result.score == 0
    assert "missing" in link(result)["reasons"]
    assert all(e["source_ids"] != ["a"] for e in link(result)["evidence"])


def test_longer_unlinked_paragraph_does_not_look_like_extra_short_piece() -> None:
    truth = groundtruth(
        [piece("a", "Tax"), piece("b", B, 1), piece("large", "Tax " + A)],
        [relation("continues_in", "a", "b")],
    )
    assert score(truth, "Tax " + B, "Tax " + A).score == 1


def test_hyphenated_word_uses_piece_provenance() -> None:
    truth = groundtruth(
        [piece("a", "comp-"), piece("b", "lex", 1)],
        [relation("continues_in", "a", "b", hyphenated=True)],
    )
    result = score(truth, "comp-lex")
    assert result.score == 1


@pytest.mark.parametrize("output", ["well-known", "wellknown"])
def test_compound_hyphen_both_accepted_spellings_have_one_occurrence(output: str) -> None:
    truth = groundtruth(
        [piece("a", "well-"), piece("b", "known", 1)],
        [relation("continues_in", "a", "b")],
    )
    result = score(truth, output)
    assert result.score == 1
    assert link(result)["reasons"] == []


def test_partial_hyphenated_chain_uses_join_provenance_at_first_break() -> None:
    truth = groundtruth(
        [piece("a", "comp-"), piece("b", "lex " + B, 1), piece("c", C, 2)],
        [relation("continues_in", "a", "b", hyphenated=True), relation("continues_in", "b", "c")],
    )
    result = score(truth, "complex " + B, C)
    assert result.score == 0.5
    assert link(result)["passed"]


def test_small_missing_piece_is_not_recovered_by_fuzzy_whole_paragraph_alignment() -> None:
    truth = groundtruth([piece("a", A), piece("b", "Tax", 1)], [relation("continues_in", "a", "b")])
    result = score(truth, A)
    assert result.score == 0
    assert link(result)["reasons"] == ["missing"]
    assert [e["source_ids"] for e in link(result)["evidence"]] == [["a"]]


def test_overlapping_fuzzy_windows_around_one_occurrence_are_not_duplicates() -> None:
    result = score(joined_truth(), A.replace("Monday", "Mondax") + " " + B)
    assert result.score == 1
    assert link(result)["reasons"] == []


def test_near_duplicate_source_paragraphs_allocate_distinct_legitimate_occurrences() -> None:
    other = A.replace("Monday", "Tuesday")
    truth = joined_truth(piece("other", other))
    assert score(truth, A + " " + B, other).score == 1


def test_repeated_relation_entries_count_once() -> None:
    truth = groundtruth(
        [piece("a", A), piece("b", B, 1)],
        [relation("stays_apart", "a", "b"), relation("stays_apart", "a", "b")],
    )
    result = score(truth, A, B)
    assert result.score == 1
    assert result.details["stays_apart"]["total"] == 1


def test_same_page_links_are_inapplicable_and_empty_searched_piece_is_inapplicable() -> None:
    same_page = groundtruth([piece("a", A), piece("b", B)], [relation("continues_in", "a", "b")])
    empty = groundtruth(
        [piece("a", "-"), piece("b", B, 1)], [relation("continues_in", "a", "b", hyphenated=True)]
    )
    assert not has_page_breaks(same_page)
    assert not has_page_breaks(empty)
    assert score(same_page, A + " " + B).score is None
    assert failed_page_breaks(same_page).details["status"] == "not_applicable"


def test_nonparagraph_source_links_do_not_activate_metric() -> None:
    extra: dict[str, Any] = {"class": "caption"}
    truth = groundtruth(
        [piece("a", A, **extra), piece("b", B, 1, **extra)],
        [relation("continues_in", "a", "b")],
    )
    assert not has_page_breaks(truth)


def test_reordered_groundtruth_storage_preserves_reading_order_assignment() -> None:
    stored = joined_truth(piece("c", C, 1))
    reordered = stored.model_copy(
        update={
            "elements": [stored.elements[2], *stored.elements[:2]],
            "reading_order": ["a", "b", "c"],
        }
    )
    result = score(reordered, A + " " + B + " The", C[4:])
    assert result.score == 0
    assert "misplaced_break" in link(result)["reasons"]


def test_two_of_three_joins_without_keep_apart_score_two_thirds() -> None:
    truth = groundtruth(
        [piece("a", A), piece("b", B, 1), piece("c", C, 2), piece("d", D, 3)],
        [
            relation("continues_in", "a", "b"),
            relation("continues_in", "b", "c"),
            relation("continues_in", "c", "d"),
        ],
    )
    assert score(truth, A + " " + B, C + " " + D).score == pytest.approx(2 / 3)


def test_join_and_apart_rates_have_equal_weight_not_link_count_weight() -> None:
    truth = groundtruth(
        [
            piece("a", A),
            piece("b", B, 1),
            piece("c", C, 2),
            piece("d", D, 3),
            piece("e", E),
            piece("f", "Minutes approved without changes.", 1),
        ],
        [
            relation("continues_in", "a", "b"),
            relation("continues_in", "b", "c"),
            relation("continues_in", "c", "d"),
            relation("stays_apart", "e", "f"),
        ],
    )
    assert score(
        truth, A + " " + B, C + " " + D, E, "Minutes approved without changes."
    ).score == pytest.approx(5 / 6)
    result = score(truth, " ".join([A, B, C, D]), E + " Minutes approved without changes.")
    assert result.score == 0.5
    assert result.details["continues_in"]["passed"] == 3
    assert result.details["stays_apart"]["passed"] == 0


def test_extra_partial_compound_chain_is_found_using_join_provenance() -> None:
    truth = groundtruth(
        [piece("a", "co-"), piece("b", "op-", 1), piece("c", "era", 2)],
        [relation("continues_in", "a", "b"), relation("continues_in", "b", "c")],
    )
    result = score(truth, "coopera", "coop")
    assert result.score == 0
    assert all("duplicate" in outcome["reasons"] for outcome in result.details["links"])
    assert any(e["block"] == 1 for e in link(result)["evidence"] if e["role"] == "extra")


def budget_truth(*extra: dict[str, Any]) -> Groundtruth:
    return groundtruth(
        [piece("a", "Budget"), piece("b", "grew by ten percent.", 1), *extra],
        [relation("continues_in", "a", "b")],
    )


def test_genuine_heading_identical_to_piece_is_a_reserved_separate_copy() -> None:
    truth = budget_truth(piece("heading", "Budget", **{"class": "heading", "level": 1}))
    result = score(truth, "Budget", "Budget grew by ten percent.")
    assert result.score == 1
    assert link(result)["reasons"] == []


def test_genuine_repeated_captions_reserve_the_expected_number_of_separate_copies() -> None:
    caption: dict[str, Any] = {"class": "caption"}
    truth = budget_truth(
        piece("caption1", "Budget", **caption),
        piece("caption2", "Budget", 1, **caption),
    )
    assert score(truth, "Budget", "Budget", "Budget grew by ten percent.").score == 1
    extra = score(truth, "Budget", "Budget", "Budget", "Budget grew by ten percent.")
    assert extra.score == 0
    assert "duplicate" in link(extra)["reasons"]
    assert len([e for e in link(extra)["evidence"] if e["role"] == "extra"]) == 1


@pytest.mark.parametrize("cls", ["heading", "caption", "footnote"])
def test_printed_unscored_text_glued_inside_paragraph_is_not_a_reserved_copy(cls: str) -> None:
    extra: dict[str, Any] = {"class": cls, "level": 1}
    truth = budget_truth(piece("extra", "Budget", **extra))
    result = score(truth, "Budget Budget grew by ten percent.")
    assert result.score == 0
    assert "extra_source_text" in link(result)["reasons"]


def test_furniture_copy_is_reserved_only_when_it_is_a_separate_whole_block() -> None:
    truth = budget_truth(piece("footer", "Budget", channel="furniture"))
    assert score(truth, "Budget", "Budget grew by ten percent.").score == 1
    glued = score(truth, "Budget Budget grew by ten percent.")
    assert glued.score == 0
    assert "extra_source_text" in link(glued)["reasons"]


def test_unprinted_elements_never_reserve_extra_piece_copies() -> None:
    truth = budget_truth(piece("unprinted", "Budget", channel="hidden"))
    result = score(truth, "Budget", "Budget grew by ten percent.")
    assert result.score == 0
    assert "duplicate" in link(result)["reasons"]


def test_separate_unscored_copy_uses_whole_block_similarity_at_point_nine() -> None:
    caption: dict[str, Any] = {"class": "caption"}
    truth = groundtruth(
        [
            piece("a", "Annual Budget"),
            piece("b", B, 1),
            piece("caption", "Annual Budget", **caption),
        ],
        [relation("continues_in", "a", "b")],
    )
    assert score(truth, "Annual Budgef", "Annual Budget " + B).score == 1
    assert score(truth, "Annual Budget Appendix", "Annual Budget " + B).score == 0


@pytest.mark.parametrize("extra_first", [False, True])
def test_other_paragraph_copy_inside_join_fails_in_both_output_orders(extra_first):
    truth = groundtruth(
        [piece("e", E), piece("c", C), piece("a", A, 1), piece("b", B, 2), piece("d", D, 2)],
        [relation("continues_in", "a", "b")],
    )
    joined = A + " " + B + " " + E
    outputs = [E, C, joined, D] if extra_first else [C, joined, E, D]
    result = score(truth, *outputs)
    assert result.score == 0
    assert "merged" in link(result)["reasons"]
    if extra_first:
        assert any(
            e["source_ids"] == ["e"] for e in link(result)["evidence"] if e["role"] == "extra"
        )


def test_unrelated_extra_copy_outside_join_does_not_fail_that_link():
    truth = groundtruth(
        [piece("e", E), piece("c", C), piece("a", A, 1), piece("b", B, 2)],
        [relation("continues_in", "a", "b")],
    )
    assert score(truth, E, C, A + " " + B, E).score == 1


def test_short_piece_typo_can_fail_even_with_a_matching_whole_paragraph():
    truth = groundtruth(
        [piece("a", "The team reviewed the quarterly"), piece("b", "meeting.", 1)],
        [relation("continues_in", "a", "b")],
    )
    result = score(truth, "The team reviewed the quarterly meetings.")
    assert result.score == 0
    assert "missing" in link(result)["reasons"]


def test_unknown_invented_words_at_join_are_a_documented_limit():
    assert score(joined_truth(), A + " lorem ipsum " + B).score == 1


def test_combined_page_search_matches_independent_per_block_complete_search():
    import random
    from fractions import Fraction

    from wholedoc_bench.scorer.page_breaks import _best
    from wholedoc_bench.scorer.utils.text_matching import find_candidate

    rng = random.Random(1580032)
    for _ in range(300):
        needle = "".join(rng.choices("abcd ", k=rng.randrange(3, 22)))
        variants = [needle, needle.replace("a", "b", 1)]
        texts = ["".join(rng.choices("abcd ", k=rng.randrange(0, 28))) for _ in range(4)]
        texts[rng.randrange(4)] = "start " + variants[rng.randrange(2)] + " end"
        claimed = [[] for _ in texts]
        block = rng.randrange(4)
        if texts[block]:
            claimed[block] = [(0, rng.randrange(1, len(texts[block]) + 1))]
        candidates = [
            (i, match)
            for i, value in enumerate(texts)
            if (match := find_candidate(variants, value, claimed[i], reference=True))
        ]
        expected = min(
            candidates,
            key=lambda item: (
                Fraction(item[1].distance, item[1].length),
                item[1].distance,
                item[0],
                item[1].start,
                item[1].end,
                item[1].variant,
            ),
            default=None,
        )
        assert _best(variants, texts, claimed) == expected


def repeated_phrase_truth():
    first = "The financial report must reach the committee for approval"
    ending = "before the end of the month."
    prefix = "The purchasing department must also gather every outstanding invoice"
    suffix = "and submit its separate account " + ending
    truth = groundtruth(
        [piece("a", first), piece("b", ending, 1), piece("c", prefix + " " + suffix, 1)],
        [relation("continues_in", "a", "b")],
    )
    return truth, first + " " + ending, prefix, suffix, ending


def test_split_unlinked_paragraph_owns_its_repeated_phrase():
    truth, joined, prefix, suffix, _ = repeated_phrase_truth()
    assert score(truth, joined, prefix, suffix).score == 1


def test_erroneous_unlinked_paragraph_can_still_identify_repeated_phrase_context():
    truth, joined, _, suffix, _ = repeated_phrase_truth()
    assert score(truth, joined, "Several invented words replace the start " + suffix).score == 1


def test_identifying_fragment_cannot_support_a_separate_shared_phrase():
    truth, joined, prefix, suffix, ending = repeated_phrase_truth()
    assert score(truth, joined, prefix, suffix.removesuffix(ending).strip(), ending).score == 0


@pytest.mark.parametrize("extra", ["piece", "fragments"])
def test_split_source_does_not_excuse_more_copies_than_its_source_positions(extra):
    truth, joined, prefix, suffix, ending = repeated_phrase_truth()
    outputs = [joined, prefix, suffix]
    outputs.extend([ending] if extra == "piece" else [prefix, suffix])
    result = score(truth, *outputs)
    assert result.score == 0
    assert "duplicate" in link(result)["reasons"]


def test_one_identified_source_cannot_excuse_repeated_phrases_in_invented_context():
    truth, joined, prefix, _, ending = repeated_phrase_truth()
    result = score(truth, joined, prefix, "Invented " + ending, "More invented " + ending)
    assert result.score == 0
    assert "duplicate" in link(result)["reasons"]


def test_fragments_do_not_hide_an_extra_piece_inside_the_correct_join():
    truth, joined, prefix, suffix, ending = repeated_phrase_truth()
    result = score(truth, joined + " " + ending, prefix, suffix)
    assert result.score == 0
    assert "duplicate" in link(result)["reasons"]


def test_identical_repetitions_inside_one_source_have_separate_fragment_budgets():
    truth, joined, prefix, suffix, _ = repeated_phrase_truth()
    truth = truth.model_copy(
        update={
            "elements": [
                *truth.elements[:-1],
                truth.elements[-1].model_copy(
                    update={"text": prefix + " " + suffix + " " + suffix}
                ),
            ]
        }
    )
    assert score(truth, joined, prefix, suffix, suffix).score == 1
    assert score(truth, joined, prefix, suffix, suffix, suffix).score == 0


def test_keep_apart_unmatched_block_reports_the_actual_reason():
    truth = groundtruth(
        [piece("a", A), piece("b", B, 1), piece("c", C, 1)],
        [relation("stays_apart", "a", "b")],
    )
    words = C.split()
    result = score(truth, A, B + " " + " ".join(words[:3]), " ".join(words[3:]))
    assert result.score == 0
    assert link(result)["reasons"] == ["unmatched"]


def test_identical_owner_groups_do_not_expand_through_indirect_variant_matches():
    from wholedoc_bench.scorer.page_breaks import _extras, _Unit

    units = [
        _Unit(0, ("a",), ("alpha", "beta")),
        _Unit(1, ("b",), ("beta", "gamma")),
        _Unit(2, ("c",), ("gamma", "delta")),
    ]
    extras = _extras(units, ["alpha"], [[]], {0: ("a",), 1: ("b",), 2: ("c",)})
    assert len(extras) == 1
    assert extras[0].ids == ("a", "b")


def test_identical_source_paragraphs_each_allow_one_set_of_identified_fragments():
    truth, joined, prefix, suffix, _ = repeated_phrase_truth()
    truth = truth.model_copy(
        update={
            "elements": [
                *truth.elements,
                truth.elements[-1].model_copy(update={"id": "d"}),
            ]
        }
    )
    assert score(truth, joined, prefix, suffix, prefix, suffix).score == 1
    assert score(truth, joined, prefix, suffix, prefix, suffix, prefix, suffix).score == 0


def test_partial_word_reservation_cannot_hide_another_links_extra_piece():
    truth, joined, prefix, suffix, _ = repeated_phrase_truth()
    before = "The editorial board agreed that the reviewed results should be"
    after = "published in the quarterly bulletin for all members."
    other = (prefix + " " + suffix).replace("outstanding", "quarterly")
    truth = groundtruth(
        [piece("e", before), piece("f", after, 1), *truth.elements[:2], piece("c", other, 1)],
        [relation("continues_in", "e", "f"), relation("continues_in", "a", "b")],
    )
    result = score(
        truth, before + " " + after, joined, prefix.replace("outstanding ", ""), suffix, after
    )
    assert result.score == 0.5
    assert "duplicate" in link(result)["reasons"]
    assert link(result, 1)["passed"]


def test_mangled_shared_ending_cannot_license_a_complete_extra_copy():
    truth, joined, prefix, suffix, ending = repeated_phrase_truth()
    result = score(truth, joined, prefix, suffix.replace("end of", "the"), ending)
    assert result.score == 0
    assert "duplicate" in link(result)["reasons"]


@pytest.mark.parametrize("extra_first", [False, True])
def test_shared_phrase_split_inside_words_does_not_hide_extra_in_either_order(extra_first):
    truth, joined, prefix, suffix, ending = repeated_phrase_truth()
    fragments = [prefix + " " + suffix.removesuffix(ending) + "before the", "end of the month."]
    outputs = [ending, *fragments] if extra_first else [*fragments, ending]
    assert score(truth, joined, *outputs).score == 0


def test_identifying_word_elsewhere_cannot_license_a_bare_shared_phrase():
    truth, joined, _, _, ending = repeated_phrase_truth()
    assert score(truth, joined, "invoice", ending).score == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_competing_source_ranges_use_a_conservative_content_order(reverse):
    truth, joined, _, _, ending = repeated_phrase_truth()
    truth = truth.model_copy(
        update={
            "elements": [
                *truth.elements[:-1],
                truth.elements[-1].model_copy(
                    update={"text": "U " + ending + " V U " + ending + " X"}
                ),
            ]
        }
    )
    outputs = ["U " + ending, ending + " V"]
    if reverse:
        outputs.reverse()
    assert score(truth, joined, *outputs).score == 0


def test_one_identifying_run_excuses_all_whole_suspects_it_contains():
    truth, joined, prefix, suffix, _ = repeated_phrase_truth()
    before = "The editorial board agreed that the reviewed results should be"
    after = "published in the quarterly bulletin for all members."
    context = suffix + " The appendix says " + after
    truth = groundtruth(
        [
            piece("e", before),
            piece("f", after, 1),
            *truth.elements[:2],
            piece("c", prefix + " " + context, 1),
        ],
        [relation("continues_in", "e", "f"), relation("continues_in", "a", "b")],
    )
    assert score(truth, before + " " + after, joined, prefix, context).score == 1
    assert score(truth, before + " " + after, joined, prefix, context, after).score == 0.5


@pytest.mark.parametrize(
    ("embedded_first", "extra_copy"), [(False, False), (True, False), (False, True)]
)
def test_keep_apart_prefers_intact_exact_piece_over_copy_inside_split_source(
    embedded_first, extra_copy
):
    first = "The editorial board approved the accounts for publication."
    ending = "before the end of the month."
    prefix = "All supplier invoices are processed"
    tail = "by the finance office. " + ending
    truth = groundtruth(
        [piece("a", first), piece("b", ending, 1), piece("c", prefix + " " + tail, 1)],
        [relation("stays_apart", "a", "b")],
    )
    outputs = [prefix, tail, ending] if embedded_first else [ending, prefix, tail]
    if extra_copy:
        outputs.append(ending)
    result = score(truth, first, *outputs)
    assert result.score == (0 if extra_copy else 1)
    assert ("duplicate" in link(result)["reasons"]) == extra_copy
