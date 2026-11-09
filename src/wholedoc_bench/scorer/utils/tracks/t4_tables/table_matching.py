from collections.abc import Hashable, Iterable, Sequence

from wholedoc_bench.schema import ElementClass
from wholedoc_bench.scorer.utils.blocks import Grid, MarkdownTable
from wholedoc_bench.scorer.utils.tracks.table_links import (
    TableLink,
    TruthTable,
    cell_tokens,
    token_similarity,
)


def output_links(links: Iterable[TableLink]) -> tuple[TableLink, ...]:
    return tuple(link for link in links if link.kind is ElementClass.TABLE)


def output_tables(links: Iterable[TableLink]) -> tuple[MarkdownTable, ...]:
    return tuple(link.table for link in output_links(links))


def one_to_one[L: Hashable, R: Hashable](ranked: Iterable[tuple[L, R]]) -> list[tuple[L, R]]:
    taken_left: set[L] = set()
    taken_right: set[R] = set()
    pairs: list[tuple[L, R]] = []
    for left, right in ranked:
        if left in taken_left or right in taken_right:
            continue
        taken_left.add(left)
        taken_right.add(right)
        pairs.append((left, right))
    return pairs


def match_tables(
    truth_tables: Sequence[TruthTable], output_tables: Sequence[MarkdownTable]
) -> tuple[MarkdownTable | None, ...]:
    truth_tokens = [cell_tokens(t.grid) for t in truth_tables]
    output_tokens = [cell_tokens(t.rows) for t in output_tables]
    similarity = {
        (t, o): value
        for t, truth in enumerate(truth_tokens)
        for o, output in enumerate(output_tokens)
        if (value := token_similarity(truth, output)) > 0
    }
    matches = dict(one_to_one(sorted(similarity, key=lambda pair: (-similarity[pair], *pair))))
    return tuple(
        output_tables[matches[t]] if t in matches else None for t in range(len(truth_tables))
    )


def header_and_body(table: MarkdownTable) -> tuple[Grid, Grid]:
    if table.header:
        return table.header, table.body
    return table.body[:1], table.body[1:]


def column_labels(table: MarkdownTable) -> tuple[frozenset[str], ...]:
    header, _ = header_and_body(table)
    columns = max((len(row) for row in header), default=0)
    return tuple(
        frozenset(row[column].casefold() for row in header if column < len(row))
        for column in range(columns)
    )


def casefolded_grid(rows: Grid) -> Grid:
    return tuple(tuple(cell.casefold() for cell in row) for row in rows)
