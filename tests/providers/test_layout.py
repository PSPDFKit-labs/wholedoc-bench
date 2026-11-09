from typing import Any

import pytest

from tests.builders import groundtruth, stays_apart, table_element
from wholedoc_bench.providers.layout import (
    QUOTED_CHARACTERS,
    QUOTED_END,
    QUOTED_GAP,
    QUOTED_START,
    previous_page_prompt,
    read_pages,
)
from wholedoc_bench.schema import (
    Channel,
    ChartKind,
    ElementClass,
    Font,
    FontClass,
    Groundtruth,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
)
from wholedoc_bench.scorer.run import score_document
from wholedoc_bench.scorer.utils.blocks import (
    BlockKind,
    markdown_text,
    parse_markdown_tables,
    split_blocks,
)

TABLE = "<table><tr><th>Item</th><th>Qty</th></tr><tr><td>Bolts</td><td>40</td></tr></table>"
CHART = (
    "<table><caption>Sales by region</caption><tr><th>Region</th><th>2025</th></tr>"
    "<tr><td>North</td><td>12</td></tr></table>"
)
ROWS = "<table><tr><td>Nuts</td><td>12</td></tr></table>"
CONTINUES = 'data-continues-previous="true"'
PAGE = f"""\
<div data-bbox="[80, 30, 920, 50]" data-label="Page-header" data-id="e1">Annual Review</div>
<div data-bbox="[100, 80, 900, 120]" data-label="Title" data-id="e2"># Depot notice</div>
<div data-bbox="[100, 130, 900, 160]" data-label="Section-header" data-id="e3">### Deliveries</div>
<div data-bbox="[100, 170, 900, 230]" data-label="Text" data-id="e4">Ship **now**.</div>
<div data-bbox="[100, 240, 900, 300]" data-label="Table" data-id="e5">{TABLE}</div>
<div data-bbox="[100, 305, 900, 320]" data-label="Caption" data-id="e6" \
data-caption-of="e5">Table 1: Stock</div>
<div data-bbox="[100, 325, 900, 340]" data-label="Footnote" data-id="e7" \
data-footnote-of="e5">¹ Counted in May.</div>
<div data-bbox="[100, 350, 900, 380]" data-label="Formula" data-id="e8">$$d = v^2$$</div>
<div data-bbox="[100, 400, 500, 600]" data-label="Chart" data-id="e9" \
data-chart-type="radial bar">{CHART}</div>
<div data-bbox="[600, 400, 700, 500]" data-label="Barcode" data-id="e10">INV-4417</div>
<div data-bbox="[750, 400, 900, 500]" data-label="Picture" data-id="e11"></div>
<div data-bbox="[100, 950, 900, 970]" data-label="Page-footer" data-id="e12">Page 1</div>"""


def normalize(*pages: str) -> Prediction:
    return read_pages(pages)


def items(prediction: Prediction) -> dict[str, LayoutItem]:
    return {item.id: item for item in prediction.layout}


def at(prediction: Prediction, item: LayoutItem) -> str:
    assert item.md_start is not None and item.md_end is not None
    return prediction.markdown[item.md_start : item.md_end]


def div(label: str, content: str, *attributes: str) -> str:
    attributes = (f'data-label="{label}"', *attributes)
    return f'<div data-bbox="[0, 0, 1000, 50]" {" ".join(attributes)}>{content}</div>'


def test_a_page_reads_into_markdown_layout_and_links() -> None:
    prediction = normalize(PAGE)

    assert prediction.markdown == (
        "# Depot notice\n\n### Deliveries\n\nShip **now**.\n\n"
        f"{TABLE}\n\nTable 1: Stock\n\n¹ Counted in May.\n\n$$\nd = v^2\n$$\n\n"
        f"{CHART}\n\nINV-4417\n"
    )
    found = items(prediction)
    assert [(i.id, i.cls) for i in prediction.layout] == [
        ("p1-1", ElementClass.HEADER),
        ("p1-2", ElementClass.HEADING),
        ("p1-3", ElementClass.HEADING),
        ("p1-4", ElementClass.PARAGRAPH),
        ("p1-5", ElementClass.TABLE),
        ("p1-6", ElementClass.CAPTION),
        ("p1-7", ElementClass.FOOTNOTE),
        ("p1-8", ElementClass.FORMULA),
        ("p1-9", ElementClass.CHART),
        ("p1-10", ElementClass.MACHINE_READABLE),
        ("p1-11", ElementClass.FIGURE),
        ("p1-12", ElementClass.FOOTER),
    ]
    assert [i.text for i in prediction.layout] == [
        "Annual Review",
        "Depot notice",
        "Deliveries",
        "Ship now.",
        "Item Qty Bolts 40",
        "Table 1: Stock",
        "¹ Counted in May.",
        "d = v^2",
        "Region 2025 North 12",
        "INV-4417",
        "",
        "Page 1",
    ]
    assert at(prediction, found["p1-5"]) == TABLE
    assert at(prediction, found["p1-8"]) == "$$\nd = v^2\n$$"
    for furniture in ("p1-1", "p1-12"):
        assert found[furniture].channel is Channel.FURNITURE
        assert found[furniture].md_start is None
    assert found["p1-11"].md_start is None
    assert found["p1-4"].channel is Channel.BODY
    assert found["p1-2"].box == pytest.approx((0.1, 0.08, 0.8, 0.04))
    chart = found["p1-9"].chart
    assert chart is not None
    assert (chart.kind, chart.title, chart.series) == (ChartKind.RADIAL_BAR, "Sales by region", [])
    assert prediction.relations == [
        Relation(kind=RelationKind.CAPTION_OF, source="p1-6", target="p1-5"),
        Relation(kind=RelationKind.FOOTNOTE_OF, source="p1-7", target="p1-5"),
    ]
    assert prediction.page_offsets == [0]
    assert prediction.unscored_text == []


@pytest.mark.parametrize(
    ("label", "content", "heading"),
    [
        ("Title", "Depot notice", "# Depot notice"),
        ("Section-header", "Deliveries", "## Deliveries"),
        ("Section-header", "#### Deliveries", "#### Deliveries"),
        ("Title", "## Depot\nnotice", "## Depot notice"),
        ("section_header", "#1 Best Seller", "## #1 Best Seller"),
    ],
)
def test_a_heading_keeps_the_level_the_model_marks_or_takes_its_labels(
    label: str, content: str, heading: str
) -> None:
    page = f'<div data-bbox="[0, 0, 1000, 50]" data-label="{label}">{content}</div>'
    assert normalize(page).markdown == f"{heading}\n"


@pytest.mark.parametrize("latex", ["$$x^2$$", "$x^2$", "\\[x^2\\]", "\\(x^2\\)", "x^2"])
def test_a_formula_is_written_once_as_a_display_formula(latex: str) -> None:
    page = f'<div data-bbox="[0, 0, 1000, 50]" data-label="Formula">{latex}</div>'
    prediction = normalize(page)
    assert prediction.markdown == "$$\nx^2\n$$\n"
    assert prediction.layout[0].text == "x^2"


def test_text_outside_every_wrapper_is_unscored_and_fences_go() -> None:
    page = (
        "```html\nHere is the page:\n"
        '<div data-bbox="[0, 0, 1000, 50]" data-label="Text">Kept.</div>\n'
        "Stray words\n```"
    )
    prediction = normalize(page)
    assert prediction.markdown == "Kept.\n"
    assert prediction.unscored_text == ["Here is the page:", "Stray words"]


def test_only_a_bare_fence_line_is_dropped_from_unscored_text() -> None:
    page = (
        "``` hidden words after a fence\n"
        '<div data-bbox="[0, 0, 1000, 50]" data-label="Text">Kept.</div>\n~~~ python'
    )
    assert normalize(page).unscored_text == ["``` hidden words after a fence"]


@pytest.mark.parametrize(
    ("answer", "markdown"),
    [
        ("Plain text\n\nwithout wrappers", "Plain text\n\nwithout wrappers\n"),
        ("```markdown\n# Notice\n\nText\n```", "# Notice\n\nText\n"),
        ("~~~~\nText\n~~~~", "Text\n"),
        ("```md\nA\n```\nB\n```md\nC\n```", "```md\nA\n```\nB\n```md\nC\n```\n"),
        ("```python\nx = 1\n```", "```python\nx = 1\n```\n"),
    ],
)
def test_an_answer_without_wrappers_is_the_pages_markdown(answer: str, markdown: str) -> None:
    prediction = normalize(answer)
    assert (prediction.markdown, prediction.layout) == (markdown, [])


@pytest.mark.parametrize(
    ("bbox", "box"),
    [
        ("[0, 0, 500, 1000]", (0.0, 0.0, 0.5, 1.0)),
        ("[-5, 0, 1005, 1000]", (0.0, 0.0, 1.0, 1.0)),
        ("[0, 0, 1200, 1000]", None),
        ("[500, 0, 100, 1000]", None),
        ("[0, 0, 500]", None),
        ("[0, 0, true, 1000]", None),
        ("0, 0, 500, 1000", None),
        ("[0, 0, NaN, 1000]", None),
    ],
)
def test_a_box_is_a_page_fraction_or_none(bbox: str, box: Any) -> None:
    page = f'<div data-bbox="{bbox}" data-label="Text">Words.</div>'
    [item] = normalize(page).layout
    assert item.box == (pytest.approx(box) if box else None)
    assert item.text == "Words."


def test_attributes_in_any_order_and_quotes_and_unknown_labels() -> None:
    page = (
        "<div data-label='Key-value' data-bbox='[0, 0, 1000, 50]'>Name: Ada</div>"
        '<DIV DATA-LABEL="List_item" data-bbox="[0, 60, 1000, 90]">- Bring an ID.</DIV>'
    )
    first, second = normalize(page).layout
    assert (first.cls, first.box) == (ElementClass.PARAGRAPH, (0.0, 0.0, 1.0, 0.05))
    assert (second.cls, second.text) == (ElementClass.LIST_ITEM, "Bring an ID.")


def test_nested_and_unclosed_wrappers_each_keep_their_own_text() -> None:
    page = (
        '<div data-bbox="[0, 0, 1000, 1000]" data-label="Text">\n'
        '<div data-bbox="[0, 0, 1000, 50]" data-label="Text">First <div>inner</div> part.</div>\n'
        '<div data-bbox="[0, 60, 1000, 90]" data-label="Text">Second, never closed\n'
        '<div data-bbox="[0, 100, 1000, 130]" data-label="Text">Third.</div>\n'
        "</div>"
    )
    prediction = normalize(page)
    assert [" ".join(i.text.split()) for i in prediction.layout] == [
        "First inner part.",
        "Second, never closed",
        "Third.",
    ]
    assert prediction.unscored_text == []


def test_links_need_a_kept_target_on_the_page_and_the_right_label() -> None:
    page = (
        '<div data-bbox="[0, 0, 1000, 50]" data-label="Caption" data-id="c1" '
        'data-caption-of="missing">Figure 1</div>'
        '<div data-bbox="[0, 60, 1000, 90]" data-label="Text" data-id="t1" '
        'data-caption-of="f1">Not a caption.</div>'
        '<div data-bbox="[0, 100, 1000, 130]" data-label="Caption" data-id="c2" '
        'data-caption-of="c2">Itself</div>'
        '<div data-bbox="[0, 140, 1000, 300]" data-label="Picture" data-id="f1"></div>'
        '<div data-bbox="[0, 310, 1000, 330]" data-label="Text" data-id="empty"></div>'
        '<div data-bbox="[0, 340, 1000, 360]" data-label="Caption" data-caption-of="empty">'
        "Figure 2</div>"
    )
    prediction = normalize(page)
    assert prediction.relations == []
    assert [i.id for i in prediction.layout] == ["p1-1", "p1-2", "p1-3", "p1-4", "p1-6"]


def test_ids_and_links_stay_on_their_page() -> None:
    first = (
        '<div data-bbox="[0, 0, 1000, 500]" data-label="Picture" data-id="e1">Map</div>'
        '<div data-bbox="[0, 510, 1000, 530]" data-label="Caption" data-id="e2" '
        'data-caption-of="e1">Figure 1</div>'
    )
    second = '<div data-bbox="[0, 0, 1000, 20]" data-label="Caption" data-caption-of="e1">X</div>'
    prediction = normalize(first, second)
    assert [(i.id, i.page) for i in prediction.layout] == [("p1-1", 0), ("p1-2", 0), ("p2-1", 1)]
    assert prediction.relations == [
        Relation(kind=RelationKind.CAPTION_OF, source="p1-2", target="p1-1")
    ]


def test_an_unknown_chart_type_is_left_out() -> None:
    page = (
        f'<div data-bbox="[0, 0, 1000, 500]" data-label="Chart" data-chart-type="3d">{TABLE}</div>'
    )
    [item] = normalize(page).layout
    assert item.chart is not None
    assert (item.chart.kind, item.chart.title) == (None, None)


def test_pages_without_markdown_start_where_the_next_page_starts() -> None:
    furniture = '<div data-bbox="[0, 950, 1000, 980]" data-label="Page-footer">2</div>'
    text = '<div data-bbox="[0, 0, 1000, 50]" data-label="Text">Body.</div>'
    prediction = normalize(text, furniture, "", text, furniture)
    assert prediction.markdown == "Body.\n\nBody.\n"
    assert prediction.page_offsets == [0, 7, 7, 7, 13]


def test_no_pages_is_an_empty_prediction() -> None:
    prediction = normalize()
    assert (prediction.markdown, prediction.layout, prediction.page_offsets) == ("", [], [])


def test_a_wrapper_keeps_its_text_around_the_wrappers_nested_in_it() -> None:
    page = (
        '<div data-bbox="[0, 0, 1000, 500]" data-label="Table" data-id="t1">'
        '<div data-bbox="[0, 0, 1000, 20]" data-label="Caption" data-caption-of="t1">Table 1</div>'
        f"{TABLE}</div>"
        '<div data-bbox="[0, 510, 1000, 900]" data-label="Picture" data-id="f1">Map key'
        '<div data-bbox="[0, 880, 1000, 900]" data-label="Caption" data-caption-of="f1">'
        "Figure 1</div>North</div>"
    )
    prediction = normalize(page)
    caption, table, picture, figure_caption = prediction.layout
    assert (caption.cls, at(prediction, caption)) == (ElementClass.CAPTION, "Table 1")
    assert (table.cls, at(prediction, table)) == (ElementClass.TABLE, TABLE)
    assert (picture.cls, at(prediction, picture)) == (ElementClass.FIGURE, "Map key\nNorth")
    assert at(prediction, figure_caption) == "Figure 1"
    assert prediction.relations == [
        Relation(kind=RelationKind.CAPTION_OF, source=caption.id, target=table.id),
        Relation(kind=RelationKind.CAPTION_OF, source=figure_caption.id, target=picture.id),
    ]
    assert prediction.unscored_text == []


def test_an_unclosed_wrapper_ends_where_the_next_wrapper_starts() -> None:
    page = (
        '<div data-bbox="[0, 0, 1000, 20]" data-label="Page-header">Annual Review\n'
        '<div data-bbox="[0, 30, 1000, 50]" data-label="Text">B</div>\n'
        '<div data-bbox="[0, 60, 1000, 80]" data-label="Text">C</div>\n\n'
        "Let me know if you need anything else."
    )
    prediction = normalize(page)
    assert [(i.cls, i.text) for i in prediction.layout] == [
        (ElementClass.HEADER, "Annual Review"),
        (ElementClass.PARAGRAPH, "B"),
        (ElementClass.PARAGRAPH, "C"),
    ]
    assert prediction.markdown == "B\n\nC\n"
    assert prediction.unscored_text == ["Let me know if you need anything else."]


def test_an_unclosed_wrapper_in_a_fenced_answer_leaves_the_next_page_alone() -> None:
    first = (
        "Here you go:\n```html\n"
        '<div data-bbox="[0, 0, 1000, 50]" data-label="Text">Page one text\n```'
    )
    second = '<div data-bbox="[0, 0, 1000, 50]" data-label="Section-header">## Page two</div>'
    prediction = normalize(first, second)
    assert prediction.markdown == "Page one text\n\n## Page two\n"
    assert parse_markdown_tables(prediction.markdown) == ()
    assert prediction.unscored_text == ["Here you go:"]


@pytest.mark.parametrize(
    "sample",
    [
        '```html\n<div class="card">\n```',
        "```html\n</div>\n```",
        '```html\n<div class="card">',
        '```html\n<div class="card">```',
    ],
)
def test_div_tags_in_a_code_sample_are_code(sample: str) -> None:
    page = (
        f'<div data-bbox="[0, 0, 1000, 300]" data-label="Code">{sample}</div>'
        '<div data-bbox="[0, 310, 1000, 330]" data-label="Text">After.</div>'
    )
    prediction = normalize(
        page, '<div data-bbox="[0, 0, 1000, 50]" data-label="Text">Next page.</div>'
    )
    code, after, following = prediction.layout
    assert code.cls is ElementClass.CODE
    assert at(prediction, code).startswith("```html\n")
    assert at(prediction, code).endswith("\n```")
    assert (after.text, following.text) == ("After.", "Next page.")
    assert prediction.markdown.endswith("\n\nAfter.\n\nNext page.\n")


@pytest.mark.parametrize(
    "printed",
    ["Close the block with ```", "End of chapter ~~~", "Run ```make all```"],
)
def test_a_fence_like_run_before_a_closing_tag_stays_as_printed(printed: str) -> None:
    page = (
        f'<div data-bbox="[0, 0, 1000, 50]" data-label="Text">{printed}</div>'
        '<div data-bbox="[0, 60, 1000, 90]" data-label="Text">After.</div>'
    )
    prediction = normalize(page)
    assert [i.text for i in prediction.layout] == [markdown_text(printed), "After."]
    assert prediction.markdown == f"{printed}\n\nAfter.\n"


def test_an_unclosed_wrapper_keeps_the_children_written_before_its_text() -> None:
    page = (
        '<div data-bbox="[0, 0, 1000, 500]" data-label="Table" data-id="t1">'
        '<div data-bbox="[0, 0, 1000, 20]" data-label="Caption" data-caption-of="t1">'
        f"Table 1: Sales</div>{TABLE}\n"
        '<div data-bbox="[0, 510, 1000, 530]" data-label="Text">Next paragraph.</div>'
    )
    prediction = normalize(page)
    caption, table, text = prediction.layout
    assert (caption.text, at(prediction, table), text.text) == (
        "Table 1: Sales",
        TABLE,
        "Next paragraph.",
    )
    assert prediction.relations == [
        Relation(kind=RelationKind.CAPTION_OF, source=caption.id, target=table.id)
    ]
    assert prediction.unscored_text == []


def test_a_self_closing_wrapper_holds_nothing() -> None:
    page = (
        '<div data-bbox="[0, 0, 1000, 500]" data-label="Picture" />'
        '<div data-bbox="[0, 510, 1000, 530]" data-label="Text">Para.</div>stray words'
    )
    prediction = normalize(page)
    assert [(i.cls, i.text) for i in prediction.layout] == [
        (ElementClass.FIGURE, ""),
        (ElementClass.PARAGRAPH, "Para."),
    ]
    assert (prediction.markdown, prediction.unscored_text) == ("Para.\n", ["stray words"])


def test_a_fence_left_open_in_one_element_does_not_reach_the_next() -> None:
    page = (
        '<div data-bbox="[0, 0, 1000, 300]" data-label="Code">```python\nx = 1</div>\n'
        '<div data-bbox="[0, 310, 1000, 330]" data-label="Text">Close the block with ```</div>'
    )
    code, text = normalize(page).layout
    assert code.text == "x = 1"
    assert text.text == "Close the block with ```"


def test_windows_line_endings_read_as_newlines() -> None:
    answer = (
        '```markdown\r\n<div data-bbox="[0, 0, 1000, 50]" data-label="Text">One\r\ntwo</div>\r\n```'
    )
    prediction = normalize(answer)
    assert prediction.markdown == "One\ntwo\n"


@pytest.mark.parametrize(
    ("content", "markdown", "text"),
    [
        ("\\[ x = \\$5 \\]", "$$\nx = \\$5\n$$", "x = \\$5"),
        ("$a$ and $b$", "$a$ and $b$", "$a$ and $b$"),
        ("\\(a+b\\) = \\(c\\)", "\\(a+b\\) = \\(c\\)", "(a+b) = (c)"),
        ("\\[ a = b \\\\[2pt] c = d \\]", "$$\na = b \\\\[2pt] c = d\n$$", "a = b \\\\[2pt] c = d"),
    ],
)
def test_a_formula_with_dollar_signs(content: str, markdown: str, text: str) -> None:
    page = f'<div data-bbox="[0, 0, 1000, 50]" data-label="Formula">{content}</div>'
    prediction = normalize(page)
    assert (prediction.markdown, prediction.layout[0].text) == (f"{markdown}\n", text)


def test_an_empty_heading_is_left_out() -> None:
    page = '<div data-bbox="[0, 0, 1000, 50]" data-label="Title"> </div>'
    assert normalize(page).layout == []


def test_an_element_names_its_font_with_the_fields_the_model_gives() -> None:
    answer = (
        '<div data-bbox="[100,100,900,150]" data-label="Title" data-font-family="Arial" '
        'data-font-class="Sans" data-font-size="18pt"># Annual report</div>\n'
        '<div data-bbox="[100,200,900,300]" data-label="Text" data-font-class="serif" '
        'data-font-size="-3">Revenue grew in every region.</div>\n'
        '<div data-bbox="[100,400,900,500]" data-label="Text">No font given.</div>'
    )
    title, body, plain = read_pages([answer]).layout
    assert title.font == Font(family="Arial", font_class=FontClass.SANS, size=18.0)
    assert body.font == Font(font_class=FontClass.SERIF)
    assert plain.font is None


def test_claude_and_gpt_both_ask_for_fonts() -> None:
    from wholedoc_bench.providers import anthropic, openai

    for module in (anthropic, openai):
        assert "data-font-family" in module.SYSTEM_PROMPT
        assert "data-font-size" in module.SYSTEM_PROMPT


def test_a_paragraph_that_continues_the_previous_page_goes_on_in_its_block() -> None:
    first = div("Text", "The depot opens at") + div("Page-footer", "1")
    second = (
        div("Page-header", "Annual Review")
        + div("Text", "nine on weekdays.", CONTINUES)
        + div("Text", "Next paragraph.")
    )
    prediction = normalize(first, second)

    assert prediction.markdown == "The depot opens at nine on weekdays.\n\nNext paragraph.\n"
    found = items(prediction)
    assert at(prediction, found["p1-1"]) == "The depot opens at"
    assert at(prediction, found["p2-2"]) == "nine on weekdays."
    assert at(prediction, found["p2-3"]) == "Next paragraph."
    assert (found["p2-2"].page, found["p2-2"].text) == (1, "nine on weekdays.")
    assert prediction.page_offsets == [0, 19]
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-2")
    ]


@pytest.mark.parametrize(
    ("ending", "start", "joined"),
    [
        ("a deliberately comp-", "lex example.", "a deliberately complex example."),
        ("a well-", "known example.", "a wellknown example."),
    ],
)
def test_a_hyphen_after_a_letter_goes_and_the_word_joins(
    ending: str, start: str, joined: str
) -> None:
    prediction = normalize(div("Text", ending), div("Text", start, CONTINUES))
    assert prediction.markdown == f"{joined}\n"
    first, second = prediction.layout
    assert (first.text, at(prediction, first)) == (ending, ending.removesuffix("-"))
    assert at(prediction, second) == start
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-1", hyphenated=True)
    ]


@pytest.mark.parametrize(
    ("ending", "start", "joined"),
    [
        ("See pages 12-", "15 for details.", "See pages 12-15 for details."),
        ("from 1990-", "and goes on.", "from 1990-and goes on."),
        ("a dash -", "and goes on.", "a dash -and goes on."),
        ("ends with a word", "and goes on.", "ends with a word and goes on."),
        ("a dash—", "and goes on.", "a dash— and goes on."),
    ],
)
def test_any_other_ending_keeps_its_hyphen_or_takes_a_space(
    ending: str, start: str, joined: str
) -> None:
    prediction = normalize(div("Text", ending), div("Text", start, CONTINUES))
    assert prediction.markdown == f"{joined}\n"
    first, second = prediction.layout
    assert (at(prediction, first), at(prediction, second)) == (ending, start)
    [relation] = prediction.relations
    assert not relation.hyphenated


@pytest.mark.parametrize(
    ("ending", "start", "joined", "text"),
    [
        ("- Bring an ID and", "- a copy.", "- Bring an ID and a copy.", "a copy."),
        ("2. Bring an ID and", "2. a copy.", "2. Bring an ID and a copy.", "a copy."),
        ("- Bring an ID and", "a copy.", "- Bring an ID and a copy.", "a copy."),
        (
            "- The plant opened in",
            "2019. It now employs 40 people.",
            "- The plant opened in 2019. It now employs 40 people.",
            "2019. It now employs 40 people.",
        ),
        ("2. Bring an ID and", "3. a copy.", "2. Bring an ID and 3. a copy.", "3. a copy."),
    ],
)
def test_a_list_item_goes_on_without_a_marker_that_repeats_its_own(
    ending: str, start: str, joined: str, text: str
) -> None:
    prediction = normalize(div("List-item", ending), div("List-item", start, CONTINUES))
    assert prediction.markdown == f"{joined}\n"
    piece = prediction.layout[1]
    assert (piece.text, at(prediction, piece)) == (text, text)
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-1")
    ]


def test_a_list_items_piece_labelled_text_goes_on_in_the_item() -> None:
    prediction = normalize(div("List-item", "- Bring an ID and"), div("Text", "a copy.", CONTINUES))
    assert prediction.markdown == "- Bring an ID and a copy.\n"
    item, piece = prediction.layout
    assert (item.cls, piece.cls) == (ElementClass.LIST_ITEM, ElementClass.PARAGRAPH)
    assert at(prediction, piece) == "a copy."
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-1")
    ]


@pytest.mark.parametrize(
    ("ending", "start"),
    [
        ("Run this:", "```sh\nmake all\n```\nthen wait."),
        ("Run:\n```sh\nmake all\n```", "and wait."),
        ("Intro\n---", "more text"),
        ("| Item | Qty |\n|---|---|\n| Bolts | 40 |", "and more"),
        ("The total is", "# 5 in stock."),
        ("Bring an ID and", "- a copy."),
        ("<!-- a\nb -->", "more"),
        ("text", "<!-- a\nb -->"),
    ],
    ids=[
        "fence opens",
        "fence closes",
        "setext heading",
        "pipe table",
        "heading mark",
        "bullet",
        "comment ending",
        "comment piece",
    ],
)
def test_a_piece_that_would_change_how_the_markdown_reads_is_a_block_of_its_own(
    ending: str, start: str
) -> None:
    after = div("Section-header", "## Next") + div("Table", TABLE)
    prediction = normalize(div("Text", ending), div("Text", start, CONTINUES) + after)
    unmarked = normalize(div("Text", ending), div("Text", start) + after)
    assert (
        prediction.markdown == unmarked.markdown == f"{ending}\n\n{start}\n\n## Next\n\n{TABLE}\n"
    )
    assert [(b.kind, b.text) for b in split_blocks(prediction.markdown)] == [
        (b.kind, b.text) for b in split_blocks(ending) + split_blocks(start)
    ] + [(BlockKind.HEADING, "Next"), (BlockKind.TABLE, "Item Qty Bolts 40")]
    tables = parse_markdown_tables(prediction.markdown)
    assert prediction.markdown[tables[-1].start : tables[-1].end] == TABLE
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-1")
    ]


def test_a_continuation_goes_on_before_the_footnotes_of_the_previous_page() -> None:
    first = div("Text", "Stock is counted") + div("Footnote", "¹ In May.") + div("Page-footer", "1")
    second = div("Text", "every month.", CONTINUES) + div("Text", "After.")
    prediction = normalize(first, second)

    assert prediction.markdown == "Stock is counted every month.\n\n¹ In May.\n\nAfter.\n"
    found = items(prediction)
    assert at(prediction, found["p1-1"]) == "Stock is counted"
    assert at(prediction, found["p1-2"]) == "¹ In May."
    assert at(prediction, found["p2-1"]) == "every month."
    assert at(prediction, found["p2-2"]) == "After."
    assert prediction.page_offsets == [0, 17]
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-1")
    ]


def test_a_paragraph_can_go_on_over_several_pages_past_their_footnotes() -> None:
    pages = [
        div("Text", "A deliberately comp-") + div("Footnote", "¹ One.") + div("Page-footer", "1"),
        div("Text", "lex and long sen-", CONTINUES) + div("Footnote", "² Two."),
        div("Page-header", "Report")
        + div("Text", "tence.", CONTINUES)
        + div("Footnote", "³ Three."),
    ]
    prediction = normalize(*pages)

    joined = "A deliberately complex and long sentence."
    assert prediction.markdown == f"{joined}\n\n¹ One.\n\n² Two.\n\n³ Three.\n"
    found = items(prediction)
    assert [at(prediction, found[i]) for i in ("p1-1", "p2-1", "p3-2")] == [
        "A deliberately comp",
        "lex and long sen",
        "tence.",
    ]
    assert [found[i].text for i in ("p1-1", "p2-1", "p3-2")] == [
        "A deliberately comp-",
        "lex and long sen-",
        "tence.",
    ]
    assert [at(prediction, found[i]) for i in ("p1-2", "p2-2", "p3-3")] == [
        "¹ One.",
        "² Two.",
        "³ Three.",
    ]
    assert prediction.page_offsets == [0, 19, 35]
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-1", hyphenated=True),
        Relation(kind=RelationKind.CONTINUES_IN, source="p2-1", target="p3-2", hyphenated=True),
    ]


def test_a_table_that_continues_the_previous_page_is_linked_and_stays_a_table() -> None:
    first = div("Table", TABLE) + div("Page-footer", "1")
    second = div("Table", ROWS, 'data-continues-previous="True"') + div("Text", "After.")
    prediction = normalize(first, second)

    assert prediction.markdown == f"{TABLE}\n\n{ROWS}\n\nAfter.\n"
    assert len(parse_markdown_tables(prediction.markdown)) == 2
    assert prediction.relations == [
        Relation(kind=RelationKind.CONTINUES_IN, source="p1-1", target="p2-1")
    ]


@pytest.mark.parametrize(
    ("pages", "markdown"),
    [
        ([div("Text", "Alone.", CONTINUES)], "Alone."),
        (
            [div("Text", "First."), div("Text", "Second.") + div("Text", "Third.", CONTINUES)],
            "First.\n\nSecond.\n\nThird.",
        ),
        ([div("Table", TABLE), div("Text", "Text.", CONTINUES)], f"{TABLE}\n\nText."),
        ([div("Text", "Text."), div("Table", ROWS, CONTINUES)], f"Text.\n\n{ROWS}"),
        (
            [div("Text", "Text.") + div("Picture", "Map"), div("Text", "More.", CONTINUES)],
            "Text.\n\nMap\n\nMore.",
        ),
        (
            [div("Section-header", "## Costs"), div("Text", "More.", CONTINUES)],
            "## Costs\n\nMore.",
        ),
        ([div("Page-footer", "1"), div("Text", "More.", CONTINUES)], "More."),
        ([div("Text", "Text."), "", div("Text", "More.", CONTINUES)], "Text.\n\nMore."),
        (
            [div("Text", "Text."), div("Text", "More.", 'data-continues-previous="no"')],
            "Text.\n\nMore.",
        ),
        ([div("Text", "Text."), div("List-item", "- More.", CONTINUES)], "Text.\n\n- More."),
    ],
    ids=[
        "first page",
        "not the first element",
        "paragraph after a table",
        "table after a paragraph",
        "after a picture",
        "after a heading",
        "no ending",
        "not the page before",
        "not true",
        "list item after a paragraph",
    ],
)
def test_the_mark_is_ignored_without_an_ending_of_its_class_on_the_page_before(
    pages: list[str], markdown: str
) -> None:
    prediction = normalize(*pages)
    assert prediction.markdown == f"{markdown}\n"
    assert prediction.relations == []


def test_the_previous_page_prompt_quotes_the_element_that_ends_the_page() -> None:
    answer = (
        div("Text", "Bring these:")
        + div("List-item", "- An ID.")
        + div("Footnote", "¹ Or a passport.")
        + div("Text", "")
        + div("Page-footer", "Page 1")
    )
    assert previous_page_prompt(answer) == (
        'The previous page ended with this element:\n\n<div data-label="List-item">\n'
        "- An ID.\n</div>"
    )
    assert previous_page_prompt(f"```html\n{div('table', TABLE)}\n```") == (
        f'The previous page ended with this element:\n\n<div data-label="table">\n{TABLE}\n</div>'
    )
    quote = previous_page_prompt("<div data-label='Key \"value\"'>Name: Ada</div>")
    assert quote is not None and '<div data-label="Key &quot;value&quot;">' in quote


def test_a_long_ending_is_quoted_by_its_start_and_its_end() -> None:
    start, middle, end = "a" * QUOTED_START, "b" * 500, "c" * QUOTED_END
    quote = previous_page_prompt(div("Text", start + middle + end))
    assert quote is not None
    assert quote.endswith(f">\n{start}{QUOTED_GAP}{end}\n</div>")
    whole = "d" * QUOTED_CHARACTERS
    assert f">\n{whole}\n</div>" in (previous_page_prompt(div("Text", whole)) or "")


@pytest.mark.parametrize(
    "answer",
    [
        div("Text", "Text.") + div("Picture", ""),
        div("Text", "Text.") + div("Formula", "$$x$$"),
        div("Title", "# Notice"),
        div("Page-header", "Annual Review") + div("Footnote", "¹ Note."),
        "Plain text without wrappers",
        "",
    ],
    ids=["picture", "formula", "heading", "furniture and footnotes", "no wrappers", "empty"],
)
def test_no_previous_page_prompt_without_a_paragraph_list_item_or_table_at_the_end(
    answer: str,
) -> None:
    assert previous_page_prompt(answer) is None


def test_marked_continuations_score_as_one_paragraph_and_one_table(
    demo_groundtruth: Groundtruth,
) -> None:
    by_id = {e.id: e for e in demo_groundtruth.elements}

    def table(id_: str) -> str:
        element = by_id[id_]
        rows = [*([element.header[0]] if element.header else []), *(element.rows or [])]
        cells = "".join(f"<tr>{''.join(f'<td>{c}</td>' for c in row)}</tr>" for row in rows)
        return f"<table>{cells}</table>"

    def scores(continues: str) -> dict[str, float | None]:
        pages = [
            div("Title", f"# {by_id['e1'].text}")
            + div("Text", by_id["e2"].text)
            + div("Caption", by_id["e3"].text, 'data-caption-of="t1"')
            + div("Table", table("e4"), 'data-id="t1"')
            + div("Page-footer", "1"),
            div("Table", table("e6"), continues)
            + div("Text", by_id["e7"].text)
            + div("Page-footer", "2"),
            div("Text", by_id["e9"].text, continues) + div("Page-footer", "3"),
        ]
        result = score_document(demo_groundtruth, read_pages(pages))
        return {
            "page_breaks": result.sub_scores["T3"]["page_breaks"],
            "block_integrity": result.sub_scores["T3"]["block_integrity"],
            "table_integrity": result.sub_scores["T4"]["table_integrity"],
        }

    assert scores(CONTINUES) == {"page_breaks": 1.0, "block_integrity": 1.0, "table_integrity": 1.0}
    split = scores("")
    assert (split["page_breaks"], split["table_integrity"]) == (0.0, 0.0)


def test_marking_a_pair_that_must_stay_apart_loses() -> None:
    first = "The depot opens at nine on weekdays and closes early on public holidays."
    second = "Deliveries arrive each morning and are checked against the order list."
    regions = (
        "<table><tr><th>Region</th><th>Sales</th></tr><tr><td>North</td><td>12</td></tr></table>"
    )
    truth = groundtruth(
        [
            {"id": "a", "class": "paragraph", "page": 0, "text": first},
            {"id": "b", "class": "paragraph", "page": 1, "text": second},
            table_element("t1", 1, ["Item", "Qty"], [["Bolts", "40"]]),
            table_element("t2", 2, ["Region", "Sales"], [["North", "12"]]),
        ],
        [stays_apart("a", "b"), stays_apart("t1", "t2")],
    )

    def scores(mark: str) -> dict[str, float | None]:
        pages = [
            div("Text", first),
            div("Text", second, mark) + div("Table", TABLE),
            div("Table", regions, mark),
        ]
        result = score_document(truth, read_pages(pages))
        return {
            "page_breaks": result.sub_scores["T3"]["page_breaks"],
            "block_integrity": result.sub_scores["T3"]["block_integrity"],
            "table_integrity": result.sub_scores["T4"]["table_integrity"],
        }

    assert scores("") == {"page_breaks": 1.0, "block_integrity": 1.0, "table_integrity": 1.0}
    assert scores(CONTINUES) == {
        "page_breaks": 0.0,
        "block_integrity": 0.0,
        "table_integrity": 0.0,
    }


@pytest.mark.parametrize(("verdict", "valid"), [("valid", True), ("invalid", False), ("", None)])
def test_an_mrz_is_a_zone_with_the_models_check_digit_verdict(
    verdict: str, valid: bool | None
) -> None:
    lines = (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"
    )
    answer = (
        f'<div data-bbox="[50, 800, 950, 900]" data-label="MRZ" data-check-digits="{verdict}">\n'
        f"```\n{lines}\n```\n</div>"
    )
    (item,) = read_pages([answer]).layout
    assert (item.cls, item.text) == (ElementClass.MACHINE_READABLE, lines)
    assert (item.format, item.check_digits_valid) == ("Mrz", valid)
