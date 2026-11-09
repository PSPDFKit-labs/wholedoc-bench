import pytest

from tests.builders import FINDINGS, MINUTES, NO_HEADINGS, heading, para, truth
from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t3_structure import HeadingTree

METRIC = HeadingTree()


REPORT = truth(heading("r", "Report", 1), heading("b", "Budget", 2), heading("c", "Costs", 2))


def score(t: Groundtruth, markdown: str) -> float:
    return METRIC.score(DocContext(t, Prediction(markdown=markdown)))


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("## Report\n\n### Budget\n\n### Costs", 1.0),
        ("# Report\n\n# Budget\n\n# Costs", 2 / 3),
        ("# Report\n\n## Budget", (0.8 + 1 / 3) / 2),
        ("# Report\n\n## Budget\n\n## Costs\n\n### Note", (6 / 7 + 1) / 2),
        ("Report\n\nBudget\n\nCosts", 0.0),
    ],
)
def test_worked_examples(markdown: str, expected: float) -> None:
    assert score(REPORT, markdown) == pytest.approx(expected)


def test_levels_must_count() -> None:
    t = truth(heading("a", "Alpha", 1), heading("b", "Beta", 2))
    assert score(t, "###### Alpha\n\n# Beta") == 0.5


def test_order_is_left_to_t2() -> None:
    assert score(REPORT, "# Report\n\n## Costs\n\n## Budget") == 1.0


MARKET = truth(heading("m", "Market", 1), heading("r", "Revenue", 2), heading("e", "Energy", 2))


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("# Market\n\n## Revenue\n\n# Energy", (1 + 1 / 3) / 2),
        ("# Market\n\n## Revenue", (0.8 + 1 / 3) / 2),
    ],
    ids=["wrong level", "left out"],
)
def test_a_heading_at_the_wrong_level_costs_less_than_leaving_it_out(
    markdown: str, expected: float
) -> None:
    assert score(MARKET, markdown) == pytest.approx(expected)


def test_a_heading_too_high_costs_only_its_own_pairs() -> None:
    t = truth(*(heading(k, k, level) for k, level in (("T", 1), ("A", 2), ("B", 2), ("C", 3))))
    wrong = score(t, "# T\n\n# A\n\n## B\n\n### C")
    left_out = score(t, "# T\n\n## B\n\n### C")
    assert wrong == pytest.approx((1 + 4 / 6) / 2)
    assert left_out == pytest.approx((6 / 7 + 3 / 6) / 2)
    assert left_out < wrong


def test_a_flat_outline_loses_every_pair_of_different_levels() -> None:
    spec = [("r", 1)] + [
        (f"{s}.{t}" if t else f"{s}", 3 if t else 2) for s in range(1, 9) for t in range(5)
    ]
    t = truth(*(heading(f"h{k}", f"Heading {k}", level) for k, level in spec))
    flat = "\n\n".join(f"# Heading {k}" for k, _ in spec)
    assert score(t, flat) == pytest.approx((1 + 524 / 820) / 2)


SUMMARIES = truth(
    heading("r", "Revenue", 1),
    heading("rs", "Summary", 2),
    heading("c", "Costs", 1),
    heading("cs", "Summary", 2),
)


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("# Revenue\n\n# Costs\n\n## Summary", (6 / 7 + 3 / 6) / 2),
        ("# Revenue\n\n## Summary\n\n# Costs", (6 / 7 + 3 / 6) / 2),
        ("# Costs\n\n## Summary\n\n# Revenue\n\n## Summary", 1.0),
    ],
    ids=["first summary missing", "second summary missing", "sections swapped"],
)
def test_a_repeated_heading_scores_the_same_whichever_copy_is_matched(
    markdown: str, expected: float
) -> None:
    assert score(SUMMARIES, markdown) == pytest.approx(expected)


def test_a_reversed_hierarchy_is_wrong() -> None:
    t = truth(heading("c", "Company", 1), heading("f", "Finance", 2))
    assert score(t, "# Finance\n\n## Company") == 0.5


def test_a_heading_less_than_half_the_same_is_another_heading() -> None:
    assert score(REPORT, "# Report\n\n## Budget\n\n## Coasts and rivers") == pytest.approx(
        (4 / 6 + 1 / 3) / 2
    )


def test_a_typo_costs_little_and_case_nothing() -> None:
    assert score(REPORT, "# REPORT\n\n## Budgets\n\n## costs") == pytest.approx(
        ((2 + 6 / 7) / 3 + 1) / 2
    )


def test_closing_hashes_and_markup_are_not_text() -> None:
    assert score(REPORT, "# Report #\n\n## **Budget** ##\n\n## Costs ##") == 1.0


def test_an_empty_heading_line_is_left_out() -> None:
    assert score(REPORT, "# Report\n\n#\n\n## Budget\n\n## Costs") == 1.0


@pytest.mark.parametrize(
    ("expected", "markdown"),
    [
        ("Issue #", "# [Issue #](https://example.com)"),
        ("#", "# &#35;"),
    ],
)
def test_closing_markers_are_read_before_the_text_is_decoded(expected: str, markdown: str) -> None:
    assert score(truth(heading("h", expected, 1)), markdown) == 1.0


def test_groundtruth_headings_follow_reading_order() -> None:
    t = truth(heading("c", "Costs", 2), heading("r", "Report", 1), heading("b", "Budget", 2))
    t = t.model_copy(update={"reading_order": ["r", "b", "c"]})
    assert score(t, "# Report\n\n## Budget\n\n## Costs") == 1.0


def test_a_running_header_is_not_a_heading() -> None:
    t = truth(
        heading("r", "Report", 1),
        heading("h", "Annual report 2025", 1, channel="furniture"),
        para("p", "The budget grew this year."),
    )
    assert score(t, "# Report\n\nThe budget grew this year.") == 1.0


def test_a_heading_of_underscores_scores_one() -> None:
    t = truth(heading("s", "Sign here: __________", 1), para("a", FINDINGS))
    assert score(t, f"# Sign here: __________\n\n{FINDINGS}") == 1.0


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        (f"{FINDINGS}\n\n{MINUTES}", 1.0),
        (f"# {FINDINGS}\n\n{MINUTES}", 0.0),
    ],
    ids=["none written", "paragraph as heading"],
)
def test_a_document_without_headings_has_an_empty_outline(markdown: str, expected: float) -> None:
    assert score(NO_HEADINGS, markdown) == expected


def test_a_heading_without_a_level_is_a_groundtruth_error() -> None:
    t = truth(heading("r", "Report", 1))
    t = t.model_copy(update={"elements": [t.elements[0].model_copy(update={"level": None})]})
    with pytest.raises(ValueError, match="heading r has no level"):
        score(t, "# Report")
