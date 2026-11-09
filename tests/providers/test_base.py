import pytest

from wholedoc_bench.providers import clip_box
from wholedoc_bench.providers.base import grid_place


def test_box_inside_the_page_is_only_scaled() -> None:
    assert clip_box(10, 20, 30, 40, page_width=100, page_height=200) == pytest.approx(
        (0.1, 0.1, 0.3, 0.2)
    )


def test_small_overflow_is_clipped_edge_first() -> None:
    assert clip_box(-0.5, 0, 20, 10, page_width=100, page_height=100) == pytest.approx(
        (0.0, 0.0, 0.195, 0.1)
    )
    assert clip_box(90.5, 0, 10, 10, page_width=100, page_height=100) == pytest.approx(
        (0.905, 0.0, 0.095, 0.1)
    )


@pytest.mark.parametrize(
    "box",
    [
        (-10, 0, 20, 10),
        (90, 0, 20, 10),
        (0, 95, 10, 10),
        (100, 10, 0.5, 10),
        (0, 0, 1e-200, 1e-200),
        (float("nan"), 0, 10, 10),
    ],
)
def test_box_that_does_not_fit_gives_none(box: tuple[float, float, float, float]) -> None:
    assert clip_box(*box, page_width=100, page_height=100) is None


@pytest.mark.parametrize(
    ("numbers", "place"),
    [
        ((2, 0), {"row": 2, "column": 0, "rows": 1, "columns": 1}),
        ((2, 1, 3, 2), {"row": 2, "column": 1, "rows": 3, "columns": 2}),
        ((-1, 0), {}),
        ((0, 0, 0, 1), {}),
        ((0, 0, 1, 0), {}),
        ((None, 0), {}),
        ((True, 0), {}),
        ((1.0, 0), {}),
        (("1", 0), {}),
    ],
)
def test_a_cell_place_needs_whole_numbers_in_the_grid(
    numbers: tuple[object, ...], place: dict[str, int]
) -> None:
    assert grid_place(*numbers) == place
