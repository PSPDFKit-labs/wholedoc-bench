from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

Quality = float | None

CUTOFF_TOLERANCE = 1e-9


def reaches(value: float, cutoff: float) -> bool:
    return value >= cutoff - CUTOFF_TOLERANCE


def exceeds(value: float, limit: float) -> bool:
    return value > limit + CUTOFF_TOLERANCE


def pair_in_order[Row, Column](
    rows: Sequence[Row],
    columns: Sequence[Column],
    quality: Callable[[Row, Column], Quality],
    place: Callable[[Column], tuple[object, ...]],
    priority: Callable[[Row], bool] | None = None,
) -> list[tuple[Row, Column]]:
    table = [[quality(row, column) for column in columns] for row in rows]
    order = sorted(range(len(columns)), key=lambda c: place(columns[c]))
    priority_rows = None if priority is None else [priority(row) for row in rows]
    pairs = best_pairing(table, column_order=order, priority_rows=priority_rows)
    return [(rows[r], columns[c]) for r, c in pairs]


def best_pairing(
    quality: Sequence[Sequence[Quality]],
    *,
    column_order: Sequence[int] | None = None,
    priority_rows: Sequence[bool] | None = None,
) -> list[tuple[int, int]]:
    if len({len(row) for row in quality}) > 1:
        raise ValueError("every row of pair qualities must have the same number of columns")
    if not quality or not quality[0]:
        return []
    columns = len(quality[0])
    order = list(range(columns)) if column_order is None else list(column_order)
    if sorted(order) != list(range(columns)):
        raise ValueError(f"column_order must list each of the {columns} columns once")
    values = np.array([[_checked(row[c]) for c in order] for row in quality])
    allowed = ~np.isnan(values)
    pair_values = _pair_values(len(quality), priority_rows)
    weights = np.where(allowed, pair_values[:, None] + np.nan_to_num(values), 0.0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [(int(r), order[c]) for r, c in zip(rows, cols, strict=True) if allowed[r, c]]


def _pair_values(rows: int, priority_rows: Sequence[bool] | None) -> np.ndarray:
    pair_value = rows + 1
    priority_pair_value = pair_value * (pair_value + 1)
    priority = np.zeros(rows, dtype=bool) if priority_rows is None else np.array(priority_rows)
    if priority.shape != (rows,):
        raise ValueError(f"priority_rows must give one flag for each of the {rows} rows")
    return np.where(priority, priority_pair_value, pair_value)


def _checked(value: Quality) -> float:
    if value is None:
        return np.nan
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"pair quality must be from 0 to 1, got {value}")
    return value
