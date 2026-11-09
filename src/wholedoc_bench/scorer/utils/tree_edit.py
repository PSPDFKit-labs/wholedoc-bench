from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

NODE_EDIT_COST = 1.0

Costs = NDArray[np.float64]
Indexes = NDArray[np.intp]


@dataclass(frozen=True)
class Node:
    label: str
    children: tuple["Node", ...] = ()


@dataclass(frozen=True)
class Tree:
    labels: tuple[str, ...]
    leftmost: tuple[int, ...]

    @classmethod
    def of(cls, root: Node) -> Self:
        labels: list[str] = []
        leftmost: list[int] = []

        def visit(node: Node) -> None:
            first = len(labels)
            for child in node.children:
                visit(child)
            leftmost.append(first)
            labels.append(node.label)

        visit(root)
        return cls(tuple(labels), tuple(leftmost))

    @property
    def size(self) -> int:
        return len(self.leftmost)

    @property
    def keyroots(self) -> list[int]:
        last_with_leftmost = {leaf: node for node, leaf in enumerate(self.leftmost)}
        return sorted(last_with_leftmost.values())

    @property
    def heights(self) -> list[int]:
        heights = [0] * self.size
        for node, first in enumerate(self.leftmost):
            if first < node:
                heights[node] = 1 + max(heights[first:node])
        return heights


def tree_edit_distance(a: Tree, b: Tree, rename: Costs) -> float:
    if rename.shape != (a.size, b.size):
        raise ValueError(f"rename costs are {rename.shape}, the trees need {(a.size, b.size)}")
    if a.size > b.size:
        a, b, rename = b, a, rename.T
    tree_distance = np.zeros((a.size, b.size))
    batches = _keyroot_batches(b)
    for keyroot in a.keyroots:
        for batch in batches:
            _forest_distances(a, keyroot, batch, rename, tree_distance)
    return float(tree_distance[a.size - 1, b.size - 1])


@dataclass(frozen=True)
class _KeyrootBatch:
    nodes: Indexes
    subtree_starts: Indexes
    on_leftmost_path: NDArray[np.bool_]

    @classmethod
    def of(cls, tree: Tree, keyroots: list[int]) -> Self:
        firsts = np.array([tree.leftmost[root] for root in keyroots], dtype=np.intp)
        widths = np.array(keyroots, dtype=np.intp) - firsts + 1
        offsets = np.arange(widths.max(), dtype=np.intp)
        in_forest = offsets < widths[:, None]
        nodes = np.where(in_forest, firsts[:, None] + offsets, 0)
        subtree_starts = np.where(in_forest, np.array(tree.leftmost)[nodes] - firsts[:, None], 0)
        return cls(nodes, subtree_starts, in_forest & (subtree_starts == 0))


def _keyroot_batches(tree: Tree) -> list[_KeyrootBatch]:
    heights = tree.heights
    by_height: dict[int, list[int]] = {}
    for root in tree.keyroots:
        by_height.setdefault(heights[root], []).append(root)
    return [_KeyrootBatch.of(tree, roots) for _, roots in sorted(by_height.items())]


def _forest_distances(
    a: Tree, keyroot: int, batch: _KeyrootBatch, rename: Costs, tree_distance: Costs
) -> None:
    a_first = a.leftmost[keyroot]
    forests, width = batch.nodes.shape
    columns = np.arange(width + 1, dtype=np.float64) * NODE_EDIT_COST
    forest_rows = np.arange(forests)[:, None]
    subtree_starts = [a.leftmost[node] - a_first for node in range(a_first, keyroot + 1)]
    previous = np.broadcast_to(columns, (forests, width + 1))
    kept = {0: previous}
    rows_read_later = {
        start for row_before, start in enumerate(subtree_starts) if start != row_before
    }
    for x, subtree_start in enumerate(subtree_starts, start=1):
        a_node = a_first + x - 1
        best = (
            kept.get(subtree_start, previous)[forest_rows, batch.subtree_starts]
            + tree_distance[a_node][batch.nodes]
        )
        if subtree_start == 0:
            renamed = previous[:, :-1] + rename[a_node][batch.nodes]
            best = np.where(batch.on_leftmost_path, renamed, best)
        best = np.minimum(best, previous[:, 1:] + NODE_EDIT_COST)
        first_column = np.full((forests, 1), x * NODE_EDIT_COST)
        candidates = np.concatenate((first_column, best), axis=1) - columns
        previous = np.minimum.accumulate(candidates, axis=1) + columns
        if x in rows_read_later:
            kept[x] = previous
        if subtree_start == 0:
            path = batch.on_leftmost_path
            tree_distance[a_node][batch.nodes[path]] = previous[:, 1:][path]
