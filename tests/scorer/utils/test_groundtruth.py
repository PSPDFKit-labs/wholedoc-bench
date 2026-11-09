import pytest

from tests.builders import chain
from wholedoc_bench.schema import Element
from wholedoc_bench.scorer.text_styles import style_variants
from wholedoc_bench.scorer.utils.groundtruth import cell_texts, join_variants, table_grid
from wholedoc_bench.scorer.utils.prose import normalize


@pytest.mark.parametrize(
    ("pieces", "edges", "expected"),
    [
        (
            ["well- ", " known-\t", " thing"],
            [False, False],
            ("well-known-thing", "wellknown-thing", "well-knownthing", "wellknownthing"),
        ),
        (["com- ", " plex", " words"], [True, False], ("complex words",)),
        (["a-", "b-", "c-", "d-", "e"], [False] * 4, ("a-b-c-d-e", "abcde")),
        (["a-", "-", "b"], [False, False], ("a--b", "a-b", "ab")),
        (["left ", " right"], [True], ("leftright",)),
    ],
)
def test_plain_and_mapped_joins_share_spelling_order_and_limit(pieces, edges, expected):
    assert join_variants(pieces, edges) == expected
    _, variants = style_variants(chain(pieces, edges))[0]
    assert tuple(value.text for value in variants) == tuple(normalize(t) for t in expected)


def test_a_merged_cell_is_one_cell_in_the_text_and_fills_its_positions_in_the_grid() -> None:
    table = Element.model_validate(
        {
            "id": "t",
            "class": "table",
            "page": 0,
            "header": ["Region", "Revenue", None],
            "rows": [["North", "100", "120"], [None, "90", "95"]],
            "spans": [{"row": 0, "column": 1, "columns": 2}, {"row": 1, "column": 0, "rows": 2}],
        }
    )
    assert cell_texts(table) == ["Region", "Revenue", "North", "100", "120", "90", "95"]
    assert table_grid(table) == (
        [["Region", "Revenue", "Revenue"]],
        [["North", "100", "120"], ["North", "90", "95"]],
    )
