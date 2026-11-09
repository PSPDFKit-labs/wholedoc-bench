import pytest

from tests.builders import chain, para, truth
from wholedoc_bench.schema import StyleName
from wholedoc_bench.scorer.text_styles import score_text_styles, style_variants
from wholedoc_bench.scorer.utils.blocks import split_blocks
from wholedoc_bench.scorer.utils.text_mapping import Origin
from wholedoc_bench.scorer.utils.text_matching import allocate


def styled(text, style="bold", **extra):
    return para("a", text, styles=[{"style": style, "start": 0, "end": len(text)}], **extra)


def result(elements, markdown, relations=None):
    doc = truth(*elements, relations=relations)
    doc = doc.model_copy(
        update={"reading_order": [e.id for e in doc.elements if e.channel.value == "body"]}
    )
    return score_text_styles(doc, split_blocks(markdown))


SUPPORTED_SYNTAX = {
    "bold": [
        "**Budget**",
        "__Budget__",
        "<b>Budget</b>",
        "<strong>Budget</strong>",
        "<b><b>Budget</b></b>",
        "**Bud**<strong>get</strong>",
    ],
    "italic": ["*Budget*", "_Budget_", "<i>Budget</i>", "<em>Budget</em>"],
    "strikethrough": [
        "~~Budget~~",
        "<s>Budget</s>",
        "<del>Budget</del>",
        "<strike>Budget</strike>",
    ],
    "underline": ["<u>Budget</u>", "<ins>Budget</ins>"],
    "superscript": ["<sup>Budget</sup>"],
    "subscript": ["<sub>Budget</sub>"],
}


def test_all_six_explicit_style_kinds():
    unsupported = [
        (style, markdown)
        for style, forms in SUPPORTED_SYNTAX.items()
        for markdown in forms
        if result([styled("Budget", style)], markdown).score != 1
    ]
    assert unsupported == []


@pytest.mark.parametrize(
    ("source", "markdown", "expected"),
    [
        ("Payment is due Friday.", "**Payment is due Fridax.**", 1),
        ("tax", "**taz**", 0),
        ("Budget", "**Budget** **Budget**", 2 / 3),
        ("Budget", "**Budget**\n\nExtra plain sentence.", 1),
        ("abcdefghij", "**abcdefghi**", 18 / 19),
        ("abcdefghij", "**abcdefghijx**", 20 / 21),
    ],
)
def test_character_counts_and_location_threshold(source, markdown, expected):
    assert result([styled(source)], markdown).score == pytest.approx(expected)


def test_equal_mean_small_extra_style_is_not_diluted():
    assert result([styled("A" * 100)], "**" + "A" * 100 + "** *x*").score == 0.5


def test_four_missing_bold_characters():
    r = result(
        [styled("A" * 100), para("b", "STOP", styles=[{"style": "bold", "start": 0, "end": 4}])],
        "**" + "A" * 100 + "**\n\nSTOP",
    )
    assert r.score == pytest.approx(200 / 204)


@pytest.mark.parametrize("plain_source", [False, True])
def test_small_footnote_reference_has_equal_style_weight(plain_source):
    reference = para(
        "b",
        "note1" if plain_source else "note¹",
        styles=[] if plain_source else [{"style": "superscript", "start": 4, "end": 5}],
    )
    output = "note¹" if plain_source else "note1"
    r = result([styled("A" * 100), reference], "**" + "A" * 100 + "**\n\n" + output)
    assert r.score == 0.5
    assert r.details["styles"]["superscript"]["f1"] == 0


def test_bold_over_a_whole_heading_is_not_counted():
    heading = styled("Budget", cls="heading", level=1)
    heading["class"] = "heading"
    heading.pop("cls", None)
    for markdown in ("# Budget", "# **Budget**", "<h1><strong>Budget</strong></h1>"):
        assert result([heading], markdown).score == 1
    heading["styles"] = []
    assert result([heading], "# Budget").score == 1
    assert result([heading], "# **Budget**").score == 1


def test_bold_over_part_of_a_heading_counts():
    heading = para("a", "Annual Budget", styles=[{"style": "bold", "start": 7, "end": 13}])
    heading["class"] = "heading"
    heading["level"] = 1
    assert result([heading], "# Annual **Budget**").score == 1
    assert result([heading], "# Annual Budget").score == 0
    assert result([heading], "# **Annual** Budget").score == 0


def test_bold_over_a_whole_output_heading_still_matches_a_bold_paragraph():
    assert result([styled("Total")], "## **Total**").score == 1
    assert result([para("a", "Total", styles=[])], "## **Total**").score == 1


def test_bold_over_a_whole_output_heading_does_not_stand_for_part_of_one():
    heading = para("a", "Annual Budget", styles=[{"style": "bold", "start": 7, "end": 13}])
    heading["class"] = "heading"
    heading["level"] = 1
    assert result([heading], "# **Annual Budget**").score == 0


def test_a_heading_split_over_pages_is_whole_only_when_joined():
    annual = para("a", "Annual", styles=[])
    budget = para("b", "Budget", page=1, styles=[{"style": "bold", "start": 0, "end": 6}])
    for piece in (annual, budget):
        piece["class"] = "heading"
        piece["level"] = 1
    split = [{"kind": "continues_in", "from": "a", "to": "b"}]
    assert result([annual, budget], "# Annual **Budget**", split).score == 1
    assert result([annual, budget], "# Annual Budget", split).score == 0


def test_punctuation_does_not_decide_whether_a_heading_is_bold_all_over():
    heading = para("a", "Article 1:", styles=[])
    heading["class"] = "heading"
    heading["level"] = 2
    assert result([heading], "## **Article 1:**").score == 1
    assert result([heading], "## **Article 1**:").score == 1


def test_bold_over_a_whole_paragraph_counts():
    assert result([styled("Budget")], "Budget").score == 0
    assert result([styled("Budget")], "**Budget**").score == 1


def test_furniture_is_excluded_from_expected_but_returned_styles_count():
    body = para("a", "The body text.", styles=[])
    footer = para(
        "f", "Footer", channel="furniture", styles=[{"style": "bold", "start": 0, "end": 6}]
    )
    assert result([body, footer], "The body text.").score == 1
    assert result([body, footer], "The body text.\n\n**Footer**").score == 0
    assert result([footer], "**Footer**").score is None


def test_occurrence_identity_uses_text_not_style():
    doc = [styled("Same paragraph."), para("b", "Same paragraph.", styles=[])]
    r = result(doc, "Same paragraph.\n\n**Same paragraph.**")
    assert r.score == 0
    assert [a["comparison_range"] for a in r.details["assignments"]] == [[0, 15], [16, 31]]


def test_exact_copy_precedes_fuzzy_competitor():
    targets = [["Payment is due Friday."], ["Payment is due Fridax."]]
    assert allocate(targets, "Payment is due Fridax.")[1].distance == 0
    r = result(
        [styled(targets[0][0]), para("b", targets[1][0], styles=[])], "**Payment is due Fridax.**"
    )
    assert r.score == 0


def test_normalized_diagnostic_positions_include_decoded_offsets():
    r = result([styled("fix")], "**ﬁx** **ﬁx**")
    assert r.score == pytest.approx(2 / 3)
    origins = r.details["unmatched_output_ranges"][0]["output_origins"]
    assert origins == [{"owner": "block:0", "start": 3, "end": 5}]


def test_joined_style_chain_and_accepted_compound_hyphen():
    parts = [
        styled("well-"),
        para("b", "known", page=1, styles=[{"style": "bold", "start": 0, "end": 5}]),
    ]
    relations = [{"kind": "continues_in", "from": "a", "to": "b"}]
    assert result(parts, "**wellknown**", relations).score == 1
    assert result(parts, "**well-known**", relations).score == 1
    assert result(parts, "**well-**\n\n**known**", relations).score == 1


def test_unicode_superscript_and_subscript_are_recognized_before_nfkc():
    parts = [
        styled("2", "superscript"),
        para("b", "2", styles=[{"style": "subscript", "start": 0, "end": 1}]),
    ]
    assert result(parts, "²\n\n₂").score == 1


def test_literal_code_and_escaped_emphasis_make_no_extra_style_claim():
    plain = para("a", "**name**", styles=[])
    assert result([plain], r"\*\*name\*\*").score == 1
    assert result([plain], "`**name**`").score == 1
    assert result([para("a", "2", styles=[])], "`²`").score == 1


def test_unsupported_css_does_not_supply_bold():
    assert result([styled("Budget")], '<span style="font-weight:bold">Budget</span>').score == 0


def test_source_superscript_needs_annotation_but_literal_code_does_not():
    unstyled = para("a", "m\u00b2", styles=[])
    annotated = para("a", "m\u00b2", styles=[{"style": "superscript", "start": 1, "end": 2}])
    assert result([unstyled], "m\u00b2").score == 0
    assert result([annotated], "m\u00b2").score == 1
    assert result([unstyled], "`m\u00b2`").score == 1


def test_removed_hyphen_and_trimmed_whitespace_have_no_origins():
    _, variants = style_variants(chain([" co- ", " op "], [False]))[0]
    kept, removed = variants
    assert kept.text == "co-op"
    assert kept.origins == (
        (Origin("0", 1),),
        (Origin("0", 2),),
        (Origin("0", 3),),
        (Origin("1", 1),),
        (Origin("1", 2),),
    )
    assert removed.text == "coop"
    assert removed.origins == (*kept.origins[:2], *kept.origins[3:])
    assert all(style == frozenset([StyleName.BOLD]) for style in removed.styles)


def test_same_spelling_preserves_the_existing_character_owner():
    doc = chain(["a-", "-", "b"], [False, False])
    styles = doc.elements[1].styles
    assert styles is not None
    styles[0] = styles[0].model_copy(update={"style": StyleName.ITALIC})
    _, variants = style_variants(doc)[0]
    assert variants[1].text == "a-b"
    assert variants[1].origins == ((Origin("0", 0),), (Origin("1", 0),), (Origin("2", 0),))
    assert variants[1].styles == (
        frozenset([StyleName.BOLD]),
        frozenset([StyleName.ITALIC]),
        frozenset([StyleName.BOLD]),
    )


def test_an_element_without_styles_is_not_graded_on_either_side():
    known = styled("Annual Budget")
    unknown = para("b", "Signed by the treasurer")
    graded = result([known, unknown], "**Annual Budget**\n\n**Signed by the treasurer**")
    assert graded.score == 1.0
    assert graded.details["styles"]["bold"] == {
        "expected": 12,
        "found": 12,
        "matched": 12,
        "f1": 1.0,
    }
    assert graded.details["unmatched_source_ids"] == []
    assert result([known, unknown], "Annual Budget\n\nSigned by the treasurer").score == 0.0


def test_only_unannotated_elements_leave_the_metric_out():
    assert result([para("a", "Plain"), para("b", "Text")], "Plain\n\nText").score is None
