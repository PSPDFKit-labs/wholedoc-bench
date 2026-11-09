import random
from functools import cache

import numpy as np
import pytest

from wholedoc_bench.scorer.utils.tree_edit import Costs, Node, Tree, tree_edit_distance


def n(label: str, *children: Node) -> Node:
    return Node(label, children)


def label_costs(a: Tree, b: Tree) -> Costs:
    return np.array([[0.0 if x == y else 1.0 for y in b.labels] for x in a.labels])


def distance(first: Node, second: Node) -> float:
    a, b = Tree.of(first), Tree.of(second)
    return tree_edit_distance(a, b, label_costs(a, b))


def test_postorder_and_leftmost() -> None:
    tree = Tree.of(n("f", n("d", n("a"), n("c", n("b"))), n("e")))
    assert tree.labels == ("a", "b", "c", "d", "e", "f")
    assert tree.leftmost == (0, 1, 1, 0, 4, 0)
    assert tree.heights == [0, 0, 1, 2, 0, 3]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (n("a"), n("b"), 1.0),
        (n("a", n("b"), n("c")), n("a", n("b")), 1.0),
        (n("a", n("b", n("c"))), n("a", n("b"), n("c")), 2.0),
        (
            n("f", n("d", n("a"), n("c", n("b"))), n("e")),
            n("f", n("c", n("d", n("a"), n("b"))), n("e")),
            2.0,
        ),
    ],
)
def test_known_distances(first: Node, second: Node, expected: float) -> None:
    assert distance(first, second) == expected
    assert distance(second, first) == expected


def test_rename_costs_must_fit_the_trees() -> None:
    a, b = Tree.of(n("a", n("b"))), Tree.of(n("a"))
    with pytest.raises(ValueError, match="rename costs"):
        tree_edit_distance(a, b, np.zeros((1, 2)))


def _random_tree(rng: random.Random, size: int) -> Node:
    labels = [rng.choice("abc") for _ in range(size)]
    parents = [None, *(rng.randrange(k) for k in range(1, size))]
    children: dict[int, list[int]] = {k: [] for k in range(size)}
    for k, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(k)

    def build(k: int) -> Node:
        return Node(labels[k], tuple(build(c) for c in children[k]))

    return build(0)


def _reference(a: Tree, b: Tree, rename: Costs) -> float:
    a_children = _children(a)
    b_children = _children(b)

    def without_root(forest: tuple[int, ...], children: list[tuple[int, ...]]) -> tuple[int, ...]:
        return (*forest[:-1], *children[forest[-1]])

    @cache
    def forest(f: tuple[int, ...], g: tuple[int, ...]) -> float:
        if not f and not g:
            return 0.0
        if not f:
            return forest(f, without_root(g, b_children)) + 1
        if not g:
            return forest(without_root(f, a_children), g) + 1
        v, w = f[-1], g[-1]
        return min(
            forest(without_root(f, a_children), g) + 1,
            forest(f, without_root(g, b_children)) + 1,
            forest(f[:-1], g[:-1]) + forest(a_children[v], b_children[w]) + rename[v, w],
        )

    return forest((a.size - 1,), (b.size - 1,))


def _children(tree: Tree) -> list[tuple[int, ...]]:
    children: list[tuple[int, ...]] = []
    for node, first in enumerate(tree.leftmost):
        kids: list[int] = []
        child = node - 1
        while child >= first:
            kids.append(child)
            child = tree.leftmost[child] - 1
        children.append(tuple(reversed(kids)))
    return children


def test_matches_the_recursive_definition_on_random_trees() -> None:
    rng = random.Random(158)
    for _ in range(300):
        a = Tree.of(_random_tree(rng, rng.randint(1, 7)))
        b = Tree.of(_random_tree(rng, rng.randint(1, 7)))
        rename = np.array([[rng.choice((0.0, 0.25, 1.0)) for _ in b.labels] for _ in a.labels])
        assert tree_edit_distance(a, b, rename) == pytest.approx(_reference(a, b, rename))
