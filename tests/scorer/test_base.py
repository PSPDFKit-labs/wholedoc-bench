from typing import ClassVar

import pytest

from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.base import ScoringError, Source, SubMetric, Track
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.run import score_corpus


class Fixed(SubMetric):
    name: ClassVar[str] = "fixed"
    source: ClassVar[Source] = Source.MARKDOWN

    def __init__(self, value: float | None) -> None:
        self.value = value

    def applies(self, truth: Groundtruth) -> bool:
        return self.value is not None

    def score(self, ctx: DocContext) -> float:
        assert self.value is not None
        return self.value


class OtherFixed(Fixed):
    name: ClassVar[str] = "other_fixed"


class Stub(Fixed):
    name: ClassVar[str] = "stub"
    implemented: ClassVar[bool] = False


class Crashes(SubMetric):
    name: ClassVar[str] = "crashes"
    source: ClassVar[Source] = Source.MARKDOWN

    def score(self, ctx: DocContext) -> float:
        raise KeyError("cell")


class OutOfRange(SubMetric):
    name: ClassVar[str] = "out_of_range"
    source: ClassVar[Source] = Source.MARKDOWN

    def score(self, ctx: DocContext) -> float:
        return 1.5


def run(track: Track, truth: Groundtruth, failed: bool = False) -> float | None:
    ctx = None if failed else DocContext(truth, Prediction(markdown="x"))
    return track.evaluate(truth, ctx).score


def test_weighted_mean(demo_groundtruth: Groundtruth) -> None:
    track = Track("TX", "test", ((Fixed(1.0), 1.0), (OtherFixed(0.0), 0.5)))
    assert run(track, demo_groundtruth) == pytest.approx(2 / 3)


def test_sub_metric_that_does_not_apply_is_left_out(demo_groundtruth: Groundtruth) -> None:
    track = Track("TX", "test", ((Fixed(0.8), 1.0), (OtherFixed(None), 0.5)))
    assert run(track, demo_groundtruth) == pytest.approx(0.8)


def test_track_with_nothing_to_score_is_none(demo_groundtruth: Groundtruth) -> None:
    track = Track("TX", "test", ((Fixed(None), 1.0),))
    assert run(track, demo_groundtruth) is None


def test_stub_is_left_out_even_on_a_failed_parse(demo_groundtruth: Groundtruth) -> None:
    track = Track("TX", "test", ((Fixed(0.8), 1.0), (Stub(1.0), 1.0)))
    ctx = DocContext(demo_groundtruth, Prediction(markdown="x"))
    assert track.evaluate(demo_groundtruth, ctx).not_implemented == ("stub",)
    assert run(track, demo_groundtruth) == pytest.approx(0.8)
    assert track.evaluate(demo_groundtruth, None).sub_scores == {"fixed": 0.0, "stub": None}


def test_duplicate_sub_metric_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        Track("TX", "test", ((Fixed(1.0), 1.0), (Fixed(0.0), 0.5)))


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        (Crashes(), "TX.crashes on plain_continuation_demo: KeyError"),
        (OutOfRange(), "TX.out_of_range on plain_continuation_demo: score 1.5 is outside"),
    ],
)
def test_broken_sub_metric_stops_the_run_and_names_itself(
    demo_groundtruth: Groundtruth, metric: SubMetric, message: str
) -> None:
    tracks = (Track("TX", "x", ((metric, 1.0),)),)
    with pytest.raises(ScoringError, match=message):
        score_corpus([(demo_groundtruth, Prediction(markdown="x"))], tracks=tracks)
