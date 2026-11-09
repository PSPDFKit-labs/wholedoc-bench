import json
from unittest.mock import patch

import pytest

from tests.builders import FINDINGS, MINUTES, NO_HEADINGS, heading, para, truth
from wholedoc_bench.cli import main
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.schema import ElementClass, Groundtruth, LayoutItem, Prediction
from wholedoc_bench.scorer.block_integrity import score_block_integrity
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.run import score_document
from wholedoc_bench.scorer.tracks.t3_structure import (
    TRACK,
    BlockIntegrity,
    HeadingTree,
    output_blocks,
    output_headings,
)

METRIC = BlockIntegrity()


def score(t: Groundtruth, markdown: str) -> float:
    return METRIC.score(DocContext(t, Prediction(markdown=markdown)))


def nutrient_output(raw: RawResult) -> Prediction:
    return NutrientProvider().normalize(raw)


def test_nutrient_output_for_the_fixture_scores_one(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult
) -> None:
    ctx = DocContext(demo_groundtruth, nutrient_output(demo_nutrient_raw))
    assert len(output_blocks(ctx)) == 7
    assert [h.level for h in output_headings(ctx)] == [2]
    result = TRACK.evaluate(demo_groundtruth, ctx)
    scored = ("block_integrity", "heading_tree", "relations")
    assert {name: result.sub_scores[name] for name in scored} == dict.fromkeys(scored, 1.0)
    assert not set(scored) & set(result.not_implemented)
    assert result.score == 1.0


def test_paragraph_split_at_the_page_break_scores_0_4(
    demo_groundtruth: Groundtruth, demo_nutrient_raw: RawResult
) -> None:
    pred = nutrient_output(demo_nutrient_raw)
    markdown = pred.markdown.replace("deliberately complex", "deliberately comp-\n\nlex")
    assert markdown != pred.markdown
    assert score(demo_groundtruth, markdown) == pytest.approx(0.4)


def test_tables_are_not_output_blocks_but_headings_are() -> None:
    t = truth(
        heading("h", "Budget", 1),
        para("p", "The budget grew this year."),
        {"id": "t", "class": "table", "page": 0, "header": ["a", "b"], "rows": [["1", "2"]]},
    )
    markdown = "# Budget\n\nThe budget grew this year.\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert [b.text for b in output_blocks(DocContext(t, Prediction(markdown=markdown)))] == [
        "Budget",
        "The budget grew this year.",
    ]
    assert score(t, markdown) == 1.0


def test_a_paragraph_written_as_a_heading_keeps_its_boundaries() -> None:
    assert score(NO_HEADINGS, f"# {FINDINGS}\n\n{MINUTES}") == 1.0


def test_heading_blocks() -> None:
    t = truth(heading("h", "Budget", 1), para("a", FINDINGS), para("b", MINUTES))
    assert score(t, f"# Budget #\n\n{FINDINGS}\n\n{MINUTES}") == 1.0


SIGNATURE = "Sign here: __________"


@pytest.mark.parametrize(
    ("printed", "markdown"),
    [
        ("* * *", "`* * *`"),
        (SIGNATURE, f"`{SIGNATURE}`"),
        (SIGNATURE, "Sign here: **__________**"),
        ("Rating: ****", "Rating: ****"),
        ("Rating: * * *", "Rating: **_* * *_**"),
    ],
)
def test_marker_text_scores_one_plain_or_formatted(printed: str, markdown: str) -> None:
    t = truth(para("a", FINDINGS), para("s", printed), para("b", MINUTES))
    assert score(t, f"{FINDINGS}\n\n{markdown}\n\n{MINUTES}") == 1.0


def test_each_list_item_is_a_block_and_a_flattened_list_is_a_merge() -> None:
    items = ["Bring the signed form.", "Bring a photo ID.", "Arrive ten minutes early."]
    t = truth(
        *(
            {"id": f"i{k}", "class": "list_item", "page": 0, "text": text}
            for k, text in enumerate(items)
        )
    )
    assert score(t, "\n".join(f"- {text}" for text in items)) == 1.0
    assert score(t, " ".join(items)) == 0.0


@pytest.mark.parametrize(
    ("first", "second", "output"),
    [
        ("co-", "op", "co-op"),
        ("co-", "op", "coop"),
    ],
)
def test_compound_word_split_at_a_page_scores_as_in_t1(
    first: str, second: str, output: str
) -> None:
    t = truth(
        para("a", first, page=0),
        para("b", second, page=1),
        relations=[{"kind": "continues_in", "from": "a", "to": "b"}],
    )
    assert score(t, output) == 1.0


def test_applies_to_any_document_with_body_text() -> None:
    assert HeadingTree().applies(truth(heading("r", "Report", 1)))
    assert HeadingTree().applies(NO_HEADINGS)
    assert not HeadingTree().applies(truth(para("s", "\u00ad")))
    assert not HeadingTree().applies(truth({"id": "f", "class": "figure", "page": 0, "text": ""}))


def test_the_t3_score_with_an_empty_outline() -> None:
    markdown = f"{FINDINGS[: len(FINDINGS) // 2]}\n\n{FINDINGS[len(FINDINGS) // 2 :]}\n\n{MINUTES}"
    result = TRACK.evaluate(NO_HEADINGS, DocContext(NO_HEADINGS, Prediction(markdown=markdown)))
    assert result.sub_scores["block_integrity"] == pytest.approx(0.4)
    assert result.sub_scores["heading_tree"] == 1.0
    assert result.score == pytest.approx(0.7)


def annotated():
    parts = [
        para(
            "a",
            "The committee reviewed all financial statements",
            styles=[{"style": "bold", "start": 0, "end": 47}],
        ),
        para("b", "before approving the final budget.", page=1, styles=[]),
    ]
    return truth(*parts, relations=[{"kind": "continues_in", "from": "a", "to": "b"}])


def test_all_five_weights_and_applicability_denominators():
    scores: dict[str, float | None] = {
        "heading_tree": 0.8,
        "block_integrity": 0.8,
        "relations": 0.8,
        "text_styles": 0.8,
        "page_breaks": 0,
    }
    assert TRACK.combine(scores) == pytest.approx(3.2 / 4.5)
    scores["page_breaks"] = None
    assert TRACK.combine(scores) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("prediction", "status", "failed"),
    [
        (Prediction(markdown=""), "parse_failed", True),
        (
            Prediction(
                markdown="",
                layout=[LayoutItem(id="x", cls=ElementClass.PARAGRAPH, page=0, text="layout only")],
            ),
            "scored",
            False,
        ),
    ],
)
def test_missing_text_never_removes_styles_or_page_links(prediction, status, failed):
    result = score_document(annotated(), prediction)
    assert result.failed == failed
    assert result.sub_scores["T3"]["text_styles"] == 0
    assert result.sub_scores["T3"]["page_breaks"] == 0
    assert result.details["T3"]["text_styles"]["status"] == status
    page = result.details["T3"]["page_breaks"]
    assert page["status"] == status
    assert page["continues_in"]["passed"] == 0
    assert page["continues_in"]["total"] == 1
    if failed:
        assert all(not row["evidence"] for row in page["links"])
        assert all(row["reasons"] == ["parse_failed"] for row in page["links"])


def test_inapplicable_results_have_null_scalars_and_explicit_reasons():
    result = score_document(truth(para("p", "Plain text.")), Prediction(markdown="Plain text."))
    assert result.sub_scores["T3"]["text_styles"] is None
    assert result.sub_scores["T3"]["page_breaks"] is None
    assert result.details["T3"]["text_styles"]["status"] == "not_applicable"
    assert result.details["T3"]["page_breaks"]["status"] == "not_applicable"
    assert result.details["T3"]["page_breaks"]["reason"]


def test_block_integrity_and_page_breaks_share_one_evaluation():
    doc = annotated()
    markdown = (
        "**The committee reviewed all financial statements** before approving the final budget."
    )
    ctx = DocContext(doc, Prediction(markdown=markdown))
    with patch(
        "wholedoc_bench.scorer.block_integrity.score_block_integrity", wraps=score_block_integrity
    ) as shared:
        TRACK.evaluate(doc, ctx)
        assert shared.call_count == 1


def test_actual_score_command_serializes_details(tmp_path, capsys):
    doc = annotated()
    corpus = tmp_path / "corpus" / doc.document.id
    corpus.mkdir(parents=True)
    (corpus / "groundtruth.json").write_text(doc.model_dump_json(by_alias=True), encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    markdown = (
        "**The committee reviewed all financial statements** before approving the final budget."
    )
    (predictions / f"{doc.document.id}.json").write_text(
        Prediction(markdown=markdown).model_dump_json(by_alias=True), encoding="utf-8"
    )
    output = tmp_path / "results" / "predictions.jsonl"
    assert (
        main(
            [
                "score",
                "--corpus",
                str(corpus.parent),
                "--predictions",
                str(predictions),
                "--results",
                str(output.parent),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    style = next(row for row in rows if row["sub_metric"] == "text_styles")
    page = next(row for row in rows if row["sub_metric"] == "page_breaks")
    assert style["score"] == 1
    assert page["score"] == 1
    assert page["weight"] == 0.5
    assert page["details"]["continues_in"]["total"] == 1
    assert style["details"]["styles"]["bold"]["expected"] == 42
    assert style["details"]["status"] == "scored"
    common_keys = {
        "document_id",
        "failed",
        "error",
        "track",
        "sub_metric",
        "source",
        "weight",
        "score",
        "status",
    }
    assert style["status"] == page["status"] == "scored"
    assert set(style) == common_keys | {"details"}
    assert set(next(row for row in rows if row["sub_metric"] == "block_integrity")) == common_keys
    assert "T3" in capsys.readouterr().out
