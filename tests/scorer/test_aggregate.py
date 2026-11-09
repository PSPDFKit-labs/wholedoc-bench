import pytest

from wholedoc_bench.scorer.aggregate import overall


def test_overall_is_the_unweighted_mean_of_tracks() -> None:
    assert overall({"T1": 0.9, "T2": 0.6, "T6": None}) == pytest.approx(0.75)
