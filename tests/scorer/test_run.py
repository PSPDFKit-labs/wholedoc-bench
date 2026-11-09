import json
from pathlib import Path
from typing import ClassVar

import pytest

from tests.builders import incomplete_truth
from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.run import (
    ParseFailure,
    aggregate,
    score_corpus,
    score_document,
    write_scores_jsonl,
)

TYPO = {"kind": "exact", "printed": "recieve"}


class HasMarkdown(SubMetric):
    name: ClassVar[str] = "has_markdown"
    source: ClassVar[Source] = Source.MARKDOWN

    def score(self, ctx: DocContext) -> float:
        return 1.0 if ctx.pred.markdown else 0.0


class NeverApplies(SubMetric):
    name: ClassVar[str] = "never_applies"
    source: ClassVar[Source] = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        return False

    def score(self, ctx: DocContext) -> float:
        raise AssertionError("must not run on a document it does not apply to")


class NothingForbidden(SubMetric):
    name: ClassVar[str] = "nothing_forbidden"
    source: ClassVar[Source] = Source.MARKDOWN

    def score(self, ctx: DocContext) -> float:
        return 0.0 if "SECRET" in ctx.pred.markdown else 1.0


class Stub(HasMarkdown):
    name: ClassVar[str] = "stub"
    implemented: ClassVar[bool] = False


TRACKS = (
    Track("TA", "a", ((HasMarkdown(), 1.0),)),
    Track("TB", "b", ((NeverApplies(), 1.0),)),
)


def test_failed_parse_scores_zero_and_stays_in_the_mean(demo_groundtruth: Groundtruth) -> None:
    ok = Prediction(markdown="# text")
    result = score_corpus([(demo_groundtruth, ok), (demo_groundtruth, None)], tracks=TRACKS)
    assert result.track_scores["TA"] == pytest.approx(0.5)
    assert result.track_scores["TB"] is None
    assert result.overall == pytest.approx(0.5)
    assert [d.failed for d in result.documents] == [False, True]
    assert result.coverage.value == pytest.approx(0.5)
    assert not result.coverage.passed


def test_absent_check_gives_a_failed_parse_zero(demo_groundtruth: Groundtruth) -> None:
    tracks = (Track("TC", "c", ((NothingForbidden(), 1.0),)),)
    items = [
        (demo_groundtruth, Prediction(markdown="clean text")),
        (demo_groundtruth, Prediction(markdown="leaks the SECRET")),
        (demo_groundtruth, None),
        (demo_groundtruth, Prediction(markdown="   ")),
    ]
    result = score_corpus(items, tracks=tracks)
    assert [d.track_scores["TC"] for d in result.documents] == [1.0, 0.0, 0.0, 0.0]
    assert [d.failed for d in result.documents] == [False, False, True, True]


def test_stub_sub_metrics_are_listed_and_left_out(demo_groundtruth: Groundtruth) -> None:
    tracks = (*TRACKS[:1], Track("TS", "s", ((Stub(), 1.0),)))
    result = score_corpus([(demo_groundtruth, Prediction(markdown="x"))], tracks=tracks)
    assert result.not_implemented == ["TS.stub"]
    assert result.track_scores["TS"] is None


def test_scores_jsonl_has_one_row_per_document_and_sub_metric(
    demo_groundtruth: Groundtruth, tmp_path: Path
) -> None:
    result = score_corpus([(demo_groundtruth, Prediction(markdown="x"))], tracks=TRACKS)
    out = tmp_path / "scores.jsonl"
    write_scores_jsonl(result, out, tracks=TRACKS)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [
        {
            "document_id": "plain_continuation_demo",
            "failed": False,
            "error": None,
            "track": "TA",
            "sub_metric": "has_markdown",
            "source": "markdown",
            "weight": 1.0,
            "score": 1.0,
            "status": "scored",
        },
        {
            "document_id": "plain_continuation_demo",
            "failed": False,
            "error": None,
            "track": "TB",
            "sub_metric": "never_applies",
            "source": "json",
            "weight": 1.0,
            "score": None,
            "status": "not_applicable",
        },
    ]


def test_parse_failure_keeps_its_reason(demo_groundtruth: Groundtruth) -> None:
    result = score_corpus([(demo_groundtruth, ParseFailure("timeout"))], tracks=TRACKS)
    assert result.documents[0].failed
    assert result.documents[0].error == "timeout"
    assert result.track_scores["TA"] == 0.0


@pytest.mark.parametrize(
    "prediction", [ParseFailure("parser crashed"), Prediction(markdown="Budget")]
)
def test_direct_scoring_grades_partial_annotations_even_on_parse_failure(
    prediction: Prediction | ParseFailure | None,
) -> None:
    result = score_document(incomplete_truth(), prediction)
    assert result.sub_scores["T3"]["text_styles"] is not None


def test_aggregate_of_document_results_equals_score_corpus(
    demo_groundtruth: Groundtruth,
) -> None:
    tracks = (*TRACKS, Track("TS", "s", ((Stub(), 1.0),)))
    items = [
        (demo_groundtruth, Prediction(markdown="# text")),
        (demo_groundtruth, ParseFailure("timeout")),
        (demo_groundtruth, Prediction(markdown="   ")),
    ]
    documents, not_implemented = [], set()
    for truth, prediction in items:
        names: set[str] = set()
        documents.append(score_document(truth, prediction, tracks, names))
        not_implemented |= names
    assert aggregate(documents, tracks, not_implemented) == score_corpus(items, tracks=tracks)


def test_aggregate_needs_a_document() -> None:
    with pytest.raises(ValueError, match="no documents"):
        aggregate([], TRACKS)
