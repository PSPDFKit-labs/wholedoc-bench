import pytest

from tests.builders import AxisRange, json_only, markdown_only, one_chart, pipe_table
from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t6_charts import TRACK

MILLION, BILLION = 1_000_000, 1_000_000_000


def t6(metric: str, truth: Groundtruth, pred: Prediction) -> float | None:
    return TRACK.evaluate(truth, DocContext(truth, pred)).sub_scores[metric]


def f1(hits: int, truth_count: int, predicted_count: int) -> float:
    return 2 * hits / (truth_count + predicted_count)


def literal(*values: str) -> str:
    rows = [[f"Q{n}", value] for n, value in enumerate(values, 1)]
    return pipe_table(["Quarter", "Revenue"], rows)


def value_score(truth_value: float, cell: str) -> float | None:
    truth = one_chart({"Revenue": [("Q1", truth_value), ("Q2", 20.0)]})
    return t6("data_points", truth, markdown_only(literal(cell, "20")))


READ_AS = {
    18.0: ["$18m", "\\$18m", "**18**", "~18", "≈18"],
    18 * MILLION: ["$18m"],
    1234.5: ["1,234.5"],
    1234567.0: ["1,234,567"],
    1234.0: ["1,234"],
    18.2: ["18.2%", "18,2"],
    12.5: ["12.5 %"],
    -12.0: ["(12)"],
    -5.0: ["−5", "-$5"],
    -3.5: ["-3.5"],
    7.0: ["€ 7", "7 €", "7K"],
    9.0: ["¥9"],
    3.5 * BILLION: ["£3.5bn"],
    -3.5 * BILLION: ["-£3.5bn"],
    42.0: ["USD 42", "42 EUR"],
    1.5: ["GBP 1.5"],
    7000.0: ["7k"],
    2000.0: ["2 thousand"],
    4 * MILLION: ["4mn"],
    1.5 * MILLION: ["1.5 million"],
    4 * BILLION: ["4B"],
    2 * BILLION: ["2 billion"],
}
NOT_READ_AS = {1.234: ["1,234"], 18000.0: ["$18m"], 12.0: ["(12)"], 5.0: ["−5"], 20.0: ["~18"]}
NOT_POINTS = ["", "n/a", "–", "-", "nan", "inf", "$", "$$5", "1.234,5", "1,23,4", "10-20"]
NOT_POINTS += ["(-5)", "~~18", "18~"]


def by_value(cells: dict[float, list[str]]) -> list[tuple[float, str]]:
    return [(value, cell) for value, written in cells.items() for cell in written]


@pytest.mark.parametrize(("truth_value", "cell"), by_value(READ_AS))
def test_number_formats_llms_write_match(truth_value: float, cell: str) -> None:
    assert value_score(truth_value, cell) == 1.0


@pytest.mark.parametrize(("truth_value", "cell"), by_value(NOT_READ_AS))
def test_number_read_the_wrong_way_does_not_match(truth_value: float, cell: str) -> None:
    assert value_score(truth_value, cell) == pytest.approx(f1(1, 2, 2))


@pytest.mark.parametrize("cell", NOT_POINTS)
def test_empty_and_text_cells_are_not_points(cell: str) -> None:
    truth = one_chart({"Revenue": [("Q1", 10.0), ("Q2", 20.0)]})
    pred = markdown_only(literal("10", cell))
    assert t6("data_points", truth, pred) == pytest.approx(f1(1, 2, 1))


WIDE, CENTERED = (0.0, 1000.0), (-500.0, 500.0)
WITHIN = [(200, 202), (-50, -50.5), (0.1, 0.101), (0.1, 0.099), (0, 0), (0.7, 0.707)]
OUTSIDE = [(200, 202.01), (-50, 50), (0, 4e-10)]
AXIS_CASES = [(2, 7, WIDE, 1), (0, -5, CENTERED, 1), (800, 808, WIDE, 1)]
AXIS_CASES += [(2, 7.01, WIDE, 0), (0, 5.01, CENTERED, 0), (800, 808.01, WIDE, 0)]
TOLERANCE = (
    [(v, p, None, 1) for v, p in WITHIN] + [(v, p, None, 0) for v, p in OUTSIDE] + AXIS_CASES
)


@pytest.mark.parametrize(("truth_value", "predicted", "axis", "expected"), TOLERANCE)
def test_json_value_tolerance(
    truth_value: float, predicted: float, axis: AxisRange | None, expected: float
) -> None:
    truth = one_chart({"Revenue": [("Q1", truth_value)]}, value_axis=axis)
    assert t6("data_points", truth, json_only({"Revenue": [("Q1", predicted)]})) == expected
