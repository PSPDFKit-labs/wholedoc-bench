from unittest.mock import patch

import pytest

from tests.builders import (
    HEADER,
    ROWS,
    context,
    html_table,
    indexes,
    markdown_of,
    one_table,
    only_table,
    para,
    pipe_table,
    quoted,
    truth,
)
from wholedoc_bench.providers import NutrientProvider, RawResult
from wholedoc_bench.schema import CellSpan, Prediction, StyleName
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.tracks.t1_text import BlockEditSimilarity, SentenceIntegrity, WordBagF1
from wholedoc_bench.scorer.tracks.t2_order import BlockOrderSimilarity, OrderTests, _searchable
from wholedoc_bench.scorer.tracks.t3_structure import BlockIntegrity, HeadingTree, output_headings
from wholedoc_bench.scorer.utils.blocks import (
    MAX_COLSPAN,
    BlockKind,
    Markup,
    markdown_text,
    normalize_text,
    parse_markdown,
    parse_markdown_tables,
    split_blocks,
)
from wholedoc_bench.scorer.utils.prose import normalize, words

H, T, X = BlockKind.HEADING, BlockKind.TABLE, BlockKind.TEXT


def test_nutrient_fixture_blocks(demo_nutrient_raw: RawResult) -> None:
    markdown = NutrientProvider().normalize(demo_nutrient_raw).markdown
    blocks = split_blocks(markdown)
    assert [b.kind for b in blocks] == [H, X, X, T, X, X, X, X]
    assert [b.text for b in blocks if b.kind is X][-2:] == ["2", "3"]


def test_every_part_is_in_exactly_one_block() -> None:
    markdown = (
        "# Title\n\nText one.\n| a | b |\n|---|---|\n| 1 | 2 |\nText two.\n\n"
        "<table><tr><td>x</td></tr></table>"
    )
    blocks = split_blocks(markdown)
    assert [b.kind for b in blocks] == [H, X, T, X, T]
    assert [b.source for b in blocks] == [
        "# Title",
        "Text one.",
        "| a | b |\n|---|---|\n| 1 | 2 |",
        "Text two.",
        "<table><tr><td>x</td></tr></table>",
    ]


def test_block_text_is_what_a_reader_sees() -> None:
    (block,) = split_blocks("See [the **report**](http://x.io) &amp; <b>more</b>")
    assert normalize(block.text) == "See the report & more"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("buy?id=7&region=eu&section=2", "buy?id=7&region=eu&section=2"),
        ("&copy 2026 &times 3", "&copy 2026 &times 3"),
        ("&regx; &notareal;", "&regx; &notareal;"),
        ("&#169 &#xA9", "&#169 &#xA9"),
        ("&copy; &#169; &#xA9; &amp;", "© © © &"),
    ],
)
def test_entity_decodes_only_with_its_semicolon_and_a_known_name(raw: str, expected: str) -> None:
    (block,) = split_blocks(raw)
    assert block.text == expected


def test_html_table_cell_decodes_an_entity_without_its_semicolon() -> None:
    (table,) = parse_markdown_tables("<table><tr><td>&copy 2026</td></tr></table>")
    assert table.rows == (("© 2026",),)


def test_heading_line_is_its_own_block() -> None:
    blocks = split_blocks("# Title\nFirst paragraph line.\nSecond line.")
    assert [(b.kind, b.source) for b in blocks] == [
        (H, "# Title"),
        (X, "First paragraph line.\nSecond line."),
    ]


def test_heading_underline_under_a_pipe_is_not_a_table() -> None:
    assert [b.kind for b in split_blocks("a | b\n---\ntext")] == [BlockKind.HEADING, BlockKind.TEXT]


def test_each_list_item_starts_a_block() -> None:
    blocks = split_blocks("Intro:\n- one\n  still one\n- two\n1. three\n2) four")
    assert [b.source for b in blocks] == [
        "Intro:",
        "- one\n  still one",
        "- two",
        "1. three",
        "2) four",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sign here: __________", "Sign here: __________"),
        ("Rating: ****", "Rating: ****"),
        ("Was: ~~~~~~", "Was: ~~~~~~"),
        ("__bold__ and _it_", "bold and it"),
        ("___both___", "both"),
        ("**bold** and *it*", "bold and it"),
        ("~~gone~~", "gone"),
        ("Sign here: **__________**", "Sign here: __________"),
        ("Sign here: **_ _ _ _**", "Sign here: _ _ _ _"),
        ("Rating: _****_", "Rating: ****"),
    ],
)
def test_markdown_text_removes_emphasis_only_around_text(text: str, expected: str) -> None:
    assert markdown_text(text) == expected


def test_shared_reader_removes_supported_markup() -> None:
    assert markdown_text("## **Bud_get** ~~old~~ `code` \\*star\n> quote\n- item\n1. one") == (
        "Bud_get old code *star quote item 1. one"
    )


def test_a_list_item_starts_a_block_however_many_spaces_follow_its_marker() -> None:
    blocks = split_blocks("-     Bring the signed form.\n-     Bring a photo ID.")
    assert [b.source for b in blocks] == ["-     Bring the signed form.", "-     Bring a photo ID."]


def test_a_year_at_the_start_of_a_line_is_not_a_list_item() -> None:
    assert len(split_blocks("It happened in\n2024. Then it ended.")) == 1


def test_windows_line_endings_separate_blocks() -> None:
    assert len(split_blocks("A one.\r\n\r\nB two.\r\n\r\nC three.")) == 3
    assert len(split_blocks("A one.\r\rB two.")) == 2


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("5 m<sup>2</sup>", ["5", "m2"]),
        ("<u>under</u>lined", ["underlined"]),
        ("un**believ**able", ["unbelievable"]),
        ("<td>a</td><td>b</td>", ["a", "b"]),
        ("snake_case", ["snake", "case"]),
    ],
)
def test_inline_tags_join_and_block_tags_separate(markdown: str, expected: list[str]) -> None:
    (block,) = split_blocks(markdown)
    assert words(block.text) == expected


def test_pipe_table_rows_and_offsets() -> None:
    table_source = pipe_table(HEADER, ROWS)
    markdown = f"Intro text.\n\n{table_source}\n\nAfter text.\n"
    table = only_table(context(one_table(), markdown))
    assert table.index == 0
    assert table.start == markdown.index(table_source)
    assert markdown[table.start : table.end].strip() == table_source
    assert table.header == (("Metric", "Value"),)
    assert table.body == (("Revenue", "10"), ("Cost", "7"))
    assert table.rows == (("Metric", "Value"), ("Revenue", "10"), ("Cost", "7"))


def test_pipe_table_without_outer_pipes() -> None:
    markdown = "Metric | Value\n:--- | ---:\nRevenue | 10\nCost | 7\n"
    table = only_table(context(one_table(), markdown))
    assert table.header == (("Metric", "Value"),)
    assert table.body == (("Revenue", "10"), ("Cost", "7"))


def test_pipe_ragged_rows_are_padded_and_cut_to_the_delimiter() -> None:
    markdown = "| A | B | C |\n|---|---|---|\n| 1 |\n| 2 | 3 | 4 | 5 |\n"
    table = only_table(context(one_table(), markdown))
    assert table.body == (("1", "", ""), ("2", "3", "4"))


def test_pipe_body_ends_at_a_blank_line_or_a_line_without_a_pipe() -> None:
    markdown = (
        "| A | B |\n|---|---|\n| 1 | 2 |\nplain line\n| 3 | 4 |\n\n"
        "| C | D |\n|---|---|\n| 5 | 6 |\n\n| 7 | 8 |\n"
    )
    tables = parse_markdown_tables(markdown)
    assert [t.body for t in tables] == [(("1", "2"),), (("5", "6"),)]


def test_html_thead_rows_are_the_header() -> None:
    markdown = (
        "<table><thead><tr><td>Metric</td><td>Value</td></tr></thead>"
        "<tbody><tr><th>Revenue</th><th>10</th></tr><tr><td>Cost</td><td>7</td></tr></tbody>"
        "</table>"
    )
    table = only_table(context(one_table(), markdown))
    assert table.header == (("Metric", "Value"),)
    assert table.body == (("Revenue", "10"), ("Cost", "7"))


def test_html_leading_all_th_rows_are_the_header_without_thead() -> None:
    markdown = (
        "<table><tr><th>Group</th><th>Group</th></tr><tr><th>Metric</th><th>Value</th></tr>"
        "<tr><td>Revenue</td><td>10</td></tr><tr><th>Cost</th><th>7</th></tr></table>"
    )
    table = only_table(context(one_table(), markdown))
    assert table.header == (("Group", "Group"), ("Metric", "Value"))
    assert table.body == (("Revenue", "10"), ("Cost", "7"))


def test_html_row_mixing_th_and_td_is_a_body_row() -> None:
    markdown = "<table><tr><th>Revenue</th><td>10</td></tr><tr><th>Cost</th><td>7</td></tr></table>"
    table = only_table(context(one_table(), markdown))
    assert table.header == ()
    assert table.body == (("Revenue", "10"), ("Cost", "7"))


def test_html_uppercase_tags_and_attributes_are_read() -> None:
    markdown = (
        '<TABLE class="data" border=1><THEAD><TR><TH style="w">Metric</TH><TH>Value</TH></TR>'
        '</THEAD><TBODY><TR><TD align="left">Revenue</TD><TD>10</TD></TR>'
        '<TR><TD COLSPAN="2">Cost</TD></TR></TBODY></TABLE>'
    )
    table = only_table(context(one_table(), markdown))
    assert table.header == (("Metric", "Value"),)
    assert table.body == (("Revenue", "10"), ("Cost", "Cost"))


@pytest.mark.parametrize("span", ["0", "-1", "x"])
def test_html_span_that_is_not_a_positive_integer_counts_as_one(span: str) -> None:
    markdown = (
        f'<table><tr><td colspan="{span}">a</td><td>b</td></tr>'
        f'<tr><td rowspan="{span}">c</td><td>d</td></tr><tr><td>e</td><td>f</td></tr></table>'
    )
    table = only_table(context(one_table(), markdown))
    assert table.body == (("a", "b"), ("c", "d"), ("e", "f"))


def test_html_rowspan_past_the_last_row_stops_at_the_last_row() -> None:
    markdown = '<table><tr><td rowspan="5">a</td><td>b</td></tr><tr><td>c</td></tr></table>'
    table = only_table(context(one_table(), markdown))
    assert table.body == (("a", "b"), ("a", "c"))


def test_html_spans_are_kept_as_merged_cells_over_the_header_and_body() -> None:
    markdown = (
        '<table><tr><th colspan="2">Revenue</th></tr><tr><td>2025</td><td>2026</td></tr>'
        '<tr><td rowspan="2">North</td><td>10</td></tr><tr><td>20</td></tr></table>'
    )
    table = only_table(context(one_table(), markdown))
    assert table.rows == (
        ("Revenue", "Revenue"),
        ("2025", "2026"),
        ("North", "10"),
        ("North", "20"),
    )
    assert table.merged == (CellSpan(row=0, column=0, columns=2), CellSpan(row=2, column=0, rows=2))


def test_a_rowspan_past_the_last_row_is_merged_only_over_the_rows_there_are() -> None:
    markdown = '<table><tr><td rowspan="5">a</td><td>b</td></tr><tr><td>c</td></tr></table>'
    assert only_table(context(one_table(), markdown)).merged == (CellSpan(row=0, column=0, rows=2),)


def test_a_pipe_table_has_no_merged_cells() -> None:
    assert only_table(context(one_table(), "|a|b|\n|---|---|\n|c|c|\n")).merged == ()


def test_html_colspan_is_clamped_to_the_html_limit() -> None:
    markdown = '<table><tr><td colspan="999999999">a</td></tr></table>'
    table = only_table(context(one_table(), markdown))
    assert table.body == (("a",) * MAX_COLSPAN,)


def test_html_caption_is_not_a_row() -> None:
    markdown = "<table><caption>Table 1</caption><tr><td>a</td><td>b</td></tr></table>"
    table = only_table(context(one_table(), markdown))
    assert table.rows == (("a", "b"),)


def test_html_nested_table_text_joins_the_cell() -> None:
    markdown = (
        "<table><tr><td>outer <table><tr><td>inner</td></tr></table></td><td>c</td></tr></table>"
    )
    table = only_table(context(one_table(), markdown))
    assert table.body == (("outer inner", "c"),)


def test_tables_are_listed_in_markdown_order_across_formats() -> None:
    first = pipe_table(["A", "B"], [["1", "2"]])
    second = html_table(["C", "D"], [["3", "4"]])
    third = pipe_table(["E", "F"], [["5", "6"]])
    markdown = markdown_of("Intro.", first, "Middle.", second, third)
    tables = parse_markdown_tables(markdown)
    assert indexes(tables) == [0, 1, 2]
    assert [t.header for t in tables] == [(("A", "B"),), (("C", "D"),), (("E", "F"),)]
    assert [t.start for t in tables] == sorted(t.start for t in tables)
    assert all(t.start < t.end for t in tables)


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_table_in_a_fenced_code_block_is_not_a_table(fence: str) -> None:
    fenced = f"{fence}\n{pipe_table(HEADER, ROWS)}\n{html_table(HEADER, ROWS)}\n{fence}"
    markdown = markdown_of(fenced, pipe_table(["After", "Fence"], [["x", "y"]]))
    tables = parse_markdown_tables(markdown)
    assert [t.header for t in tables] == [(("After", "Fence"),)]
    assert indexes(tables) == [0]


def indented(markdown: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in markdown.splitlines())


@pytest.mark.parametrize(
    "markdown",
    [
        markdown_of("Intro text.", indented(pipe_table(HEADER, ROWS), "    ")),
        indented(pipe_table(HEADER, ROWS), "\t"),
        markdown_of("- item", indented(pipe_table(HEADER, ROWS), "      ")),
        markdown_of("> Quote.", quoted(indented(pipe_table(HEADER, ROWS), "    "), "> ")),
    ],
    ids=["spaces", "tab", "list_item", "blockquote"],
)
def test_pipe_table_in_an_indented_code_block_is_not_a_table(markdown: str) -> None:
    assert parse_markdown_tables(markdown) == ()


@pytest.mark.parametrize(
    "markdown",
    [
        indented(pipe_table(HEADER, ROWS), "   "),
        markdown_of("- item", indented(pipe_table(HEADER, ROWS), "    ")),
        markdown_of("- item\n  - nested", indented(pipe_table(HEADER, ROWS), "     ")),
    ],
    ids=["three_spaces", "bullet_item", "nested_item"],
)
def test_pipe_table_indented_less_than_code_is_a_table(markdown: str) -> None:
    table = only_table(context(one_table(), markdown))
    assert table.header == (("Metric", "Value"),)
    assert table.body == (("Revenue", "10"), ("Cost", "7"))


def test_table_in_an_inline_code_span_is_not_a_table() -> None:
    markdown = "Write `<table><tr><td>x</td></tr></table>` for an HTML table.\n"
    assert parse_markdown_tables(markdown) == ()


def test_pipe_table_in_a_blockquote_is_read_with_original_offsets() -> None:
    quoted_table = quoted(pipe_table(HEADER, ROWS), ">")
    markdown = markdown_of("Intro text.", quoted_table, "After text.")
    table = only_table(context(one_table(), markdown))
    block_start = markdown.index(quoted_table)
    first_pipe = markdown.index("|", block_start)
    assert block_start <= table.start <= first_pipe
    assert markdown[table.start : table.end].rstrip().endswith("| Cost | 7 |")
    assert table.end <= markdown.index("After text.")
    assert table.header == (("Metric", "Value"),)
    assert table.body == (("Revenue", "10"), ("Cost", "7"))


def test_html_gap_inside_a_span_grid_row_is_an_empty_cell() -> None:
    markdown = (
        '<table><tr><td rowspan="2">a</td><td>b</td><td rowspan="2">c</td></tr><tr></tr></table>'
    )
    table = only_table(context(one_table(), markdown))
    assert table.body == (("a", "b", "c"), ("a", "", "c"))


def test_html_span_grid_row_end_is_not_padded() -> None:
    markdown = '<table><tr><td rowspan="2">a</td><td>b</td><td>c</td></tr><tr></tr></table>'
    table = only_table(context(one_table(), markdown))
    assert table.body == (("a", "b", "c"), ("a",))


def test_last_unclosed_table_tag_runs_to_the_end() -> None:
    markdown = markdown_of("In HTML, <table> starts a table.", "<table><tr><td>a</td><td>b</td>")
    table = only_table(context(one_table(), markdown))
    assert table.body == (("a", "b"),)


@pytest.mark.parametrize(
    "unclosed",
    [
        "<table>\n<tr><td>a</td>\n\n<td>b</td></tr>",
        "<table><tr><td>a</td><td>b",
    ],
)
def test_text_after_an_unclosed_table_stays_text(unclosed: str) -> None:
    markdown = markdown_of("Intro para.", unclosed, "First para after.", "Second para after.")
    blocks = split_blocks(markdown)
    assert [(b.kind, b.source) for b in blocks] == [
        (X, "Intro para."),
        (T, unclosed),
        (X, "First para after."),
        (X, "Second para after."),
    ]
    assert only_table(context(one_table(), markdown)).body == (("a", "b"),)


def test_unclosed_table_tag_with_rows_before_another_table_is_skipped() -> None:
    closed = "<table><tr><td>b</td></tr></table>"
    markdown = f"<table><tr><td>a</td></tr>\n{closed}\n"
    table = only_table(context(one_table(), markdown))
    assert table.start == markdown.index(closed)
    assert table.end == markdown.index(closed) + len(closed)
    assert table.body == (("b",),)


@pytest.mark.parametrize(
    ("markdown", "count"),
    [
        ("```\nunclosed fence", 0),
        ("<table>", 0),
        ("<table><tr><td>Revenue<td>10", 1),
        ("</table><table>", 0),
        ("| Metric | Value |\n|---|", 1),
        ("|---|---|\n| Revenue | 10 |", 0),
        ("|\n|\n|", 0),
        ("> | A |\n> |---|\n> | 1 |\n>", 1),
        ("- | A |\n  |---|\n  | 1 |", 1),
    ],
)
def test_malformed_markdown_gives_tables_inside_the_text(markdown: str, count: int) -> None:
    found = parse_markdown_tables(markdown)
    assert len(found) == count
    for table in found:
        assert 0 <= table.start < table.end <= len(markdown)


@pytest.mark.parametrize("comment", ["<!-- c -->", "<!---->"])
def test_html_comment_in_a_cell_is_removed_with_no_space(comment: str) -> None:
    [table] = parse_markdown_tables(f"<table><tr><td>a{comment}b</td></tr></table>\n")
    assert table.rows[0][0] == "ab"


def test_html_table_inside_a_pipe_cell_is_its_own_table_within_the_pipe_table() -> None:
    markdown = "| a | b |\n|---|---|\n| 1 | <table><tr><td>9</td></tr></table> |\n"
    pipe, html = parse_markdown_tables(markdown)
    assert pipe.rows == (("a", "b"), ("1", ""))
    assert html.rows == (("9",),)
    assert pipe.start <= html.start and html.end <= pipe.end


PIPE_CELL_CASES = [
    ("a \\| b", "a | b"),
    ("AT&amp;T", "AT&T"),
    ("**Total**", "Total"),
    ("_Total_", "Total"),
    ("`Total`", "Total"),
    ("net_income", "net_income"),
    ("H<sub>2</sub>O", "H2O"),
    ('<a href="https://example.com">Total</a>', "Total"),
    ("first<br>second", "first second"),
    ("a b", "a b"),
    ("&amp;lt;", "&lt;"),
]

HTML_CELL_CASES = [
    ("AT&amp;T", "AT&T"),
    ('<span class="n">Total</span>', "Total"),
    ("x<sup>2</sup>", "x2"),
    ("**Total**", "Total"),
    ("\n   Net\n   income\n  ", "Net income"),
    ("first<BR />second", "first second"),
    ("<B>Total</B>", "Total"),
    ("&amp;lt;", "&lt;"),
]

PIPE_ESCAPE_CASES = [
    ("\\$18m", "$18m"),
    ("\\*\\*Total\\*\\*", "**Total**"),
    ("\\*Total*", "*Total*"),
    ("\\`Total\\`", "`Total`"),
    ("a\\\\b", "a\\b"),
    ("C:\\temp", "C:\\temp"),
]

PIPE_CODE_SPAN_CASES = [
    ("`a\\*b`", "a\\*b"),
    ("`a\\\\b`", "a\\\\b"),
    ("`a\\|b`", "a|b"),
]


@pytest.mark.parametrize(("raw", "expected"), PIPE_CELL_CASES)
def test_pipe_cell_text_is_normalized(raw: str, expected: str) -> None:
    markdown = f"| Head |\n|---|\n|  {raw}  |\n"
    assert only_table(context(one_table(), markdown)).body == ((expected,),)


@pytest.mark.parametrize(("raw", "expected"), PIPE_ESCAPE_CASES)
def test_pipe_cell_backslash_escapes_are_removed(raw: str, expected: str) -> None:
    markdown = f"| Head |\n|---|\n| {raw} |\n"
    assert only_table(context(one_table(), markdown)).body == ((expected,),)


@pytest.mark.parametrize(("raw", "expected"), PIPE_CODE_SPAN_CASES)
def test_pipe_code_span_keeps_backslash_escapes_except_the_pipe(raw: str, expected: str) -> None:
    markdown = f"| Head |\n|---|\n| {raw} |\n"
    assert only_table(context(one_table(), markdown)).body == ((expected,),)


def test_html_cell_text_that_looks_like_the_code_marker_stays_as_written() -> None:
    markdown = "<table><tr><td>`c` a\ue0000\ue001b</td></tr></table>"
    assert parse_markdown_tables(markdown)[0].rows == (("c a\ue0000\ue001b",),)


@pytest.mark.parametrize("markup", list(Markup))
def test_every_markup_applies_nfkc_and_collapses_whitespace(markup: Markup) -> None:
    raw = "  ﬁnal \t １２\n\n next   word  "
    assert normalize_text(raw, markup=markup) == "final 12 next word"


@pytest.mark.parametrize("markup", list(Markup))
def test_every_markup_keeps_accents(markup: Markup) -> None:
    assert normalize_text("Café Ñandú", markup=markup) == "Café Ñandú"
    assert normalize_text("Café", markup=markup) == "Café"
    assert normalize_text("Café Ñandú", markup=markup, fold_case=True) == "café ñandú"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Straße", "strasse"),
        ("  Net   INCOME ", "net income"),
    ],
)
def test_fold_case_casefolds_the_normalized_text(raw: str, expected: str) -> None:
    assert normalize_text(raw, fold_case=True) == expected


def test_fold_case_runs_after_the_markup_rules() -> None:
    assert normalize_text("<b>TOTAL</b> AT&amp;T", markup=Markup.HTML, fold_case=True) == (
        "total at&t"
    )
    assert normalize_text("**Net** \\$18M", markup=Markup.MARKDOWN, fold_case=True) == ("net $18m")
    assert normalize_text("**Net** AT&amp;T", fold_case=True) == "**net** at&amp;t"


@pytest.mark.parametrize(("raw", "expected"), HTML_CELL_CASES)
def test_html_markup_matches_html_cells(raw: str, expected: str) -> None:
    assert normalize_text(raw, markup=Markup.HTML) == expected


def test_markdown_markup_keeps_a_backslash_pipe_inside_a_code_span() -> None:
    assert normalize_text("`a\\|b`", markup=Markup.MARKDOWN) == "a\\|b"


def test_html_markup_keeps_backslash_escapes() -> None:
    assert normalize_text("\\$18m", markup=Markup.HTML) == "\\$18m"


@pytest.mark.parametrize("markup", [Markup.HTML, Markup.MARKDOWN])
def test_code_spans_keep_entities_and_the_rest_is_decoded(markup: Markup) -> None:
    assert normalize_text("`a &amp; b` &amp; c", markup=markup) == "a &amp; b & c"


@pytest.mark.parametrize(
    ("printed", "markdown", "score"),
    [
        ("Type **name** here.", r"Type \*\*name\*\* here.", 1),
        ("Type **name** here.", "Type name here.", 15 / 19),
        ("Type name here.", "Type **name** here.", 1),
        ("Type **name** here.", "Type `**name**` here.", 1),
        ("<b>literal</b> &amp; <!-- visible -->", "`<b>literal</b> &amp; <!-- visible -->`", 1),
        (
            "**literal** <b>tag</b> <!-- visible -->",
            "```\n**literal** <b>tag</b> <!-- visible -->\n```",
            1,
        ),
        ("* * *", "`* * *`", 1),
        ("* * *", "* * *", 0),
    ],
)
def test_printed_characters_through_actual_character_metric(printed, markdown, score):
    ctx = DocContext(truth(para("p", printed)), Prediction(markdown=markdown))
    assert BlockEditSimilarity().score(ctx) == pytest.approx(score)


def test_literal_code_all_text_metrics_use_preserved_printed_content():
    text = "**literal** <b>tag</b> <!-- visible -->"
    element = para("p", text)
    element["class"] = "code"
    ctx = DocContext(truth(element), Prediction(markdown="```\n" + text + "\n```"))
    for metric in (BlockEditSimilarity(), WordBagF1(), SentenceIntegrity()):
        assert metric.score(ctx) == 1


@pytest.mark.parametrize("markdown", ["pay<!-- note -->ment", "pay<!--\n\n note -->ment"])
def test_comments_elide_without_inventing_words_through_text_metrics(markdown):
    ctx = DocContext(truth(para("p", "payment")), Prediction(markdown=markdown))
    assert len(ctx.blocks) == 1
    assert ctx.blocks[0].text == "payment"
    assert WordBagF1().score(ctx) == 1
    assert BlockEditSimilarity().score(ctx) == 1


def test_adjacent_comment_sentence_output_is_actual_printed_string():
    ctx = DocContext(truth(para("p", "end.Next")), Prediction(markdown="end.<!-- note -->Next"))
    assert ctx.blocks[0].text == "end.Next"
    assert SentenceIntegrity().score(ctx) == 1


@pytest.mark.parametrize("newline", ["\r\n", "\r"])
def test_headings_code_comments_tables_keep_shared_order_and_raw_offsets(newline):
    table = "| A | B |\n|---|---|\n| x | y |".replace("\n", newline)
    markdown = (
        "Title\n===\n\n```\n# literal\n```\n\n<!-- # hidden -->\n\n".replace("\n", newline)
        + table
        + newline
        + newline
        + "<h2>Section</h2>"
    )
    parsed = parse_markdown(markdown)
    assert [b.kind for b in parsed.blocks] == [
        BlockKind.HEADING,
        BlockKind.TEXT,
        BlockKind.TABLE,
        BlockKind.HEADING,
    ]
    assert parsed.document_text == "Title # literal A B x y Section"
    table_block = parsed.blocks[2]
    assert table_block.table is not None
    assert markdown[table_block.table.start : table_block.table.end] == table


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("Budget\n---", [("Budget", 2)]),
        ("Budget\n\n---", []),
        ("---\n---", []),
        ("- First\n---", []),
        ("# pay<!-- note -->ment", [("payment", 1)]),
        ("<!-- note --># Budget", [("Budget", 1)]),
        ("<!-- note --><h1>Budget</h1>", [("Budget", 1)]),
        ("#<!-- note -->Budget", []),
        ("<h1>`<!-- literal -->`</h1>", [("<!-- literal -->", 1)]),
        ("pay<!-- note -->ment\n===", [("payment", 1)]),
        ("<h1>First\nsecond</h1>", [("First second", 1)]),
        ("# Report\n\n  ### Budget ###", [("Report", 1), ("Budget", 3)]),
    ],
)
def test_heading_interpretation_is_owned_by_reader(markdown, expected):
    ctx = DocContext(truth(para("p", "Budget")), Prediction(markdown=markdown))
    assert [(h.text, h.level) for h in output_headings(ctx)] == expected


def test_document_text_contains_each_merged_source_cell_once():
    text = (
        '<table><tr><th colspan="2">Sales</th></tr>'
        '<tr><td rowspan="2">North</td><td>10</td></tr><tr><td>20</td></tr></table>'
    )
    ctx = DocContext(
        truth(para("p", "Before.")), Prediction(markdown="Before.\n\n" + text + "\n\nAfter.")
    )
    assert ctx.document_text == "Before. Sales North 10 20 After."
    assert ctx.blocks[1].table is not None
    assert ctx.blocks[1].table.rows == (("Sales", "Sales"), ("North", "10"), ("North", "20"))


def test_table_comments_and_inline_literals_are_decoded_once():
    text = "| A | B |\n|---|---|\n| pay<!-- x -->ment | `**literal** &amp;` |"
    parsed = parse_markdown(text)
    assert parsed.document_text == "A B payment **literal** &amp;"


def test_ordered_numbers_stay_written_in_every_shared_path():
    doc = truth(para("a", "1. One"), para("b", "1) Two"))
    ctx = DocContext(doc, Prediction(markdown="1. One\n1) Two"))
    assert _searchable(ctx.document_text).strip() == "1 one 1 two"
    assert ctx.blocks[0].text == "1. One"
    assert ctx.blocks[1].text == "1) Two"
    assert BlockEditSimilarity().score(ctx) == 1
    assert BlockIntegrity().score(ctx) == 1
    assert BlockOrderSimilarity().score(ctx) == 1
    assert OrderTests().score(ctx) == 1


def test_tracks_share_one_cached_parse():
    doc = truth(para("a", "First unique sentence."), para("b", "Second unique sentence."))
    ctx = DocContext(doc, Prediction(markdown="First unique sentence.\n\nSecond unique sentence."))
    with patch("wholedoc_bench.scorer.context.parse_markdown", wraps=parse_markdown) as shared:
        for metric in (
            BlockEditSimilarity(),
            WordBagF1(),
            BlockOrderSimilarity(),
            OrderTests(),
            BlockIntegrity(),
            HeadingTree(),
        ):
            metric.score(ctx)
        assert shared.call_count == 1
    assert ctx.blocks is ctx.parsed.blocks


@pytest.mark.parametrize(
    "markdown",
    [
        "# Title <!-- start\nend --> more",
        "# Title <!-- start\n\n# hidden\nend --> more",
    ],
)
def test_comment_crossing_heading_boundary_is_removed_from_both_blocks(markdown):
    blocks = parse_markdown(markdown).blocks
    assert [(b.kind, b.text.strip()) for b in blocks] == [
        (BlockKind.HEADING, "Title"),
        (BlockKind.TEXT, "more"),
    ]


@pytest.mark.parametrize("markdown", ["> # Title", "> Title\n> ---"])
def test_quoted_heading_uses_same_text_and_outline(markdown):
    element = {"id": "h", "class": "heading", "text": "Title", "page": 0, "level": 1}
    ctx = DocContext(truth(element), Prediction(markdown=markdown))
    assert ctx.blocks[0].kind == BlockKind.HEADING
    assert ctx.blocks[0].text == "Title"
    assert BlockEditSimilarity().score(ctx) == 1
    assert HeadingTree().score(ctx) == 1


def test_setext_lines_have_spaces_before_comparison_normalization():
    assert parse_markdown("Para line\nBudget\n---").blocks[0].text == "Para line Budget"


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("pay\n<!-- note -->ment", "pay\nment"),
        ("<h1>pay<!--\n-->ment</h1>", "payment"),
        ("Text <!-- begin\nend -->\n# Title", "Text  Title"),
        ("# `<!-- literal -->`", "<!-- literal -->"),
        ("foo<!--\n-->\nbar", "foo\nbar"),
        ("foo\n<!--\nx\n-->bar", "foo\nbar"),
    ],
)
def test_comment_newlines_do_not_invent_or_remove_word_separators(markdown, expected):
    assert parse_markdown(markdown).document_text == expected


@pytest.mark.parametrize(
    "markdown",
    [
        "> ```\n> # Title\n> ```",
        "> > ~~~\n> > # Title\n> > ~~~",
    ],
)
def test_quoted_fence_does_not_create_a_heading(markdown):
    blocks = parse_markdown(markdown).blocks
    assert len(blocks) == 1
    assert blocks[0].kind == BlockKind.TEXT
    assert blocks[0].text == "# Title"


def test_quote_marker_inside_plain_fence_cannot_close_it():
    blocks = parse_markdown("```\n> ```\n# Title\n```").blocks
    assert len(blocks) == 1
    assert blocks[0].kind == BlockKind.TEXT
    assert blocks[0].text == "> ```\n# Title"


def test_table_syntax_inside_quoted_fence_remains_literal():
    blocks = parse_markdown("> ```\n> |a|b|\n> |-|-|\n> |1|2|\n> ```").blocks
    assert len(blocks) == 1
    assert blocks[0].kind == BlockKind.TEXT
    assert blocks[0].text == "|a|b|\n|-|-|\n|1|2|"


@pytest.mark.parametrize("gap", ["\n", "\n\n"])
def test_html_heading_with_trailing_content_does_not_swallow_later_blocks(gap):
    blocks = parse_markdown(
        "<h2>Overview</h2> <p>Intro text.</p>" + gap + "Paragraph." + gap + "<h2>Details</h2>"
    ).blocks
    assert [(b.kind, " ".join(b.text.split())) for b in blocks] == [
        (BlockKind.TEXT, "Overview Intro text." + (" Paragraph." if gap == "\n" else "")),
        *([] if gap == "\n" else [(BlockKind.TEXT, "Paragraph.")]),
        (BlockKind.HEADING, "Details"),
    ]


def test_unclosed_html_heading_stops_at_blank_line():
    blocks = parse_markdown("<h2>Unclosed\n\nParagraph.\n\n<h2>Details</h2>").blocks
    assert [(b.kind, b.text.strip()) for b in blocks] == [
        (BlockKind.TEXT, "Unclosed"),
        (BlockKind.TEXT, "Paragraph."),
        (BlockKind.HEADING, "Details"),
    ]


@pytest.mark.parametrize("markdown", ["-\tparent\n\t- child", "- parent\n\t-\tchild"])
def test_tabs_around_supported_bullets_are_removed(markdown):
    assert [b.text for b in parse_markdown(markdown).blocks] == ["parent", "child"]


@pytest.mark.parametrize("marker", ["-", "1."])
def test_wrapped_list_before_divider_is_not_a_setext_heading(marker):
    blocks = parse_markdown(f"{marker} item one\n  continued\n---\nNext").blocks
    assert [b.kind for b in blocks] == [BlockKind.TEXT, BlockKind.TEXT]
    assert blocks[0].text == ("" if marker == "-" else marker + " ") + "item one\n  continued"


@pytest.mark.parametrize(
    ("markdown", "text", "styles"),
    [
        ("__bold _italic___", "bold italic", ["b"] * 5 + ["bi"] * 6),
        ("___Note:__ see below_", "Note: see below", ["bi"] * 5 + ["i"] * 10),
        ("**`*literal*`**", "*literal*", ["b"] * 9),
        (r"**\*literal\***", "*literal*", ["b"] * 9),
        ("**&#42;literal&#42;**", "*literal*", ["b"] * 9),
        ("<u>**a *b***</u>", "a b", ["bu", "bu", "biu"]),
        ("**_* * *_**", "* * *", ["bi"] * 5),
        ("**bold *italic***", "bold italic", ["b"] * 5 + ["bi"] * 6),
        ("***Note:** see below*", "Note: see below", ["bi"] * 5 + ["i"] * 10),
        ("**bold *italic* bold**", "bold italic bold", ["b"] * 5 + ["bi"] * 6 + ["b"] * 5),
        ("***both***", "both", ["bi"] * 4),
    ],
)
def test_nested_emphasis_preserves_literal_and_inherited_styles(markdown, text, styles):
    names = {"b": StyleName.BOLD, "i": StyleName.ITALIC, "u": StyleName.UNDERLINE}
    block = parse_markdown(markdown).blocks[0]
    assert block.text == text
    assert block.styles == tuple(frozenset(names[letter] for letter in style) for style in styles)


@pytest.mark.parametrize(
    ("markdown", "text", "italic"),
    [
        ("**a*", "*a", {1}),
        ("**a* b*", "a b", {0, 1, 2}),
        ("**a* b**c", "a b*c", {0, 1, 2}),
    ],
)
def test_unequal_emphasis_runs_use_remaining_markers(markdown, text, italic):
    block = parse_markdown(markdown).blocks[0]
    assert block.text == text
    assert block.styles == tuple(
        frozenset({StyleName.ITALIC}) if index in italic else frozenset()
        for index in range(len(text))
    )


def test_many_comments_preserve_blank_lines_and_outside_newlines():
    source = "\n\n".join(
        f"<!-- page {index} -->\n" + text
        for index, text in enumerate(
            [
                "pay<!-- a --><!-- b -->ment",
                "first<!--\ninside\n-->\nsecond",
                "before<!--\n\ninside-->after",
            ]
            * 100
        )
    )
    assert [b.text for b in parse_markdown(source).blocks] == [
        "payment",
        "first\nsecond",
        "beforeafter",
    ] * 100


@pytest.mark.parametrize("value", ["**stars**", "&amp;", "<!-- visible -->", "<b>tag</b>"])
def test_html_table_code_is_protected_before_the_html_parser(value):
    markdown = f"<table><tr><td>`{value}`</td></tr></table>"
    parsed = parse_markdown(markdown)
    assert parsed.document_text == value
    assert parsed.blocks[0].table is not None
    assert parsed.blocks[0].table.rows == ((value,),)
