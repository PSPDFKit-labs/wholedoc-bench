import hashlib
import html
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import accumulate
from math import isfinite
from typing import Any

from wholedoc_bench.providers.base import clip_box
from wholedoc_bench.schema import (
    Box,
    Channel,
    Chart,
    ChartKind,
    ElementClass,
    Font,
    FontClass,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
)
from wholedoc_bench.scorer.utils.blocks import BlockKind, markdown_text, split_blocks
from wholedoc_bench.scorer.utils.groundtruth import join_boundary, join_pieces

GRID = 1000.0

SYSTEM_PROMPT = f"""\
You are a document parser. Your task is to convert document pages to clean, well-structured \
markdown.

Guidelines:
- Preserve the document structure (headings, paragraphs, lists, tables)
- Convert tables to HTML format (<table>, <tr>, <th>, <td>)
- For existing tables in the document: use colspan and rowspan attributes to preserve merged \
cells and hierarchical headers
- For charts/graphs being converted to tables: use flat combined column headers (e.g., \
"Primary 2015" not separate rows) so each data cell's row contains all its labels
- For pictures, transcribe only the text printed in them; do not describe them
- Preserve any code blocks with appropriate syntax highlighting
- Maintain reading order (left-to-right, top-to-bottom for Western documents)
- Do not add commentary or explanations - only output the parsed content
- Transcribe text exactly as printed, with its spelling, punctuation, diacritics and numbers; \
do not correct or translate it
- Keep inline formatting as printed: **bold**, *italic*, <u>underline</u>, ~~strikethrough~~, \
<sup>superscript</sup> and <sub>subscript</sub>
- Write formulas in LaTeX
- Leave out text that a reader cannot see on the page, such as text under a redaction box or \
text that is only in the PDF's text layer

Additionally, wrap each layout element in a <div> tag with:
- data-bbox="[x1, y1, x2, y2]" — bounding box in normalized 0-1000 coordinates where x is \
horizontal (left edge = 0, right edge = 1000) and y is vertical (top = 0, bottom = 1000). \
x1,y1 is the top-left corner and x2,y2 is the bottom-right corner.
- data-label="<category>" — one of: Caption, Footnote, Formula, List-item, Page-footer, \
Page-header, Picture, Section-header, Table, Text, Title, Chart, Code, Barcode, MRZ
- data-id="<id>" — a short id, unique on the page, such as e1

Start a Title with # and a Section-header with ## to ###### by its level. Each List-item \
holds one item of a list. A page number is a Page-header or a Page-footer.
On a Caption, data-caption-of="<id>" names the Picture, Table or Chart it describes. On a \
Footnote that belongs to a table, data-footnote-of="<id>" names the table.
A Chart holds its data as an HTML table, with the chart's title as the table's <caption>, \
and data-chart-type="<type>" names its type, one of: {", ".join(ChartKind)}.
A Barcode holds the decoded content of the barcode or QR code, or nothing when you cannot \
decode it.
An MRZ is the machine readable zone of a passport, identity card or visa: write its lines in \
a code block, one per line, exactly as printed with every <, and set data-check-digits to \
"valid" when every check digit of the zone is right for what is printed, "invalid" otherwise.

On each Text, Title, Section-header, List-item, Caption, Footnote and Code element, name the \
font most of its text is printed in:
- data-font-family="<family>" — the typeface family, such as Arial, Times New Roman or Calibri, \
without weight or slope (Arial, not Arial Bold). Leave it out when you cannot tell the family.
- data-font-class="<class>" — one of: {", ".join(FontClass)}
- data-font-size="<points>" — the size in points at the page's printed size

When the user message quotes the element that ended the previous page, check whether this \
page's first element, after any Page-header or Page-footer, continues it: the same paragraph, \
list item or table carried over the page break. If it does, give that element the quoted \
element's data-label, add data-continues-previous="true" and transcribe only the part printed \
on this page. Leave the attribute out when the first element starts a new one, even when it \
looks like the quoted one.

Place elements in reading order. Every piece of content must be inside exactly one <div> \
wrapper."""
USER_PROMPT = (
    "Parse this document page and output its content as clean markdown, with each layout "
    'element wrapped in a <div data-bbox="[x1,y1,x2,y2]" data-label="Category"> tag. '
    "Use HTML tables for any tabular data. For charts/graphs, use flat combined column "
    "headers. Output ONLY the parsed content with div wrappers, no explanations."
)
PREVIOUS_PAGE_PROMPT = "The previous page ended with this element:\n\n{element}"
QUOTED_ELEMENT = '<div data-label="{label}">\n{content}\n</div>'
QUOTED_CHARACTERS = 2000
QUOTED_START = 500
QUOTED_END = 1500
QUOTED_GAP = " … "
PROMPT_DIGEST = hashlib.sha256(
    "\n".join(
        str(part)
        for part in (
            SYSTEM_PROMPT,
            USER_PROMPT,
            PREVIOUS_PAGE_PROMPT,
            QUOTED_ELEMENT,
            QUOTED_CHARACTERS,
            QUOTED_START,
            QUOTED_END,
            QUOTED_GAP,
        )
    ).encode()
).hexdigest()[:12]

_CLASSES = {
    "title": ElementClass.HEADING,
    "section-header": ElementClass.HEADING,
    "text": ElementClass.PARAGRAPH,
    "list-item": ElementClass.LIST_ITEM,
    "caption": ElementClass.CAPTION,
    "footnote": ElementClass.FOOTNOTE,
    "table": ElementClass.TABLE,
    "chart": ElementClass.CHART,
    "picture": ElementClass.FIGURE,
    "figure": ElementClass.FIGURE,
    "formula": ElementClass.FORMULA,
    "code": ElementClass.CODE,
    "barcode": ElementClass.MACHINE_READABLE,
    "mrz": ElementClass.MACHINE_READABLE,
    "page-header": ElementClass.HEADER,
    "page-footer": ElementClass.FOOTER,
}
_FURNITURE = frozenset({ElementClass.HEADER, ElementClass.FOOTER})
_OUTSIDE_FLOW = _FURNITURE | {ElementClass.FOOTNOTE}
_CONTINUING = frozenset({ElementClass.PARAGRAPH, ElementClass.LIST_ITEM, ElementClass.TABLE})
_CONTINUATIONS = frozenset(
    {(cls, cls) for cls in _CONTINUING} | {(ElementClass.LIST_ITEM, ElementClass.PARAGRAPH)}
)
_MAY_BE_EMPTY = frozenset({ElementClass.FIGURE, ElementClass.MACHINE_READABLE})
_LINKS = {
    ElementClass.CAPTION: ("data-caption-of", RelationKind.CAPTION_OF),
    ElementClass.FOOTNOTE: ("data-footnote-of", RelationKind.FOOTNOTE_OF),
}
_DIV = re.compile(r"""<(/?)div\b((?:[^>"']|"[^"]*"|'[^']*')*)>""", re.IGNORECASE)
_ATTRIBUTE = re.compile(r"""([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_FENCE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[ \t]*[\w+#.-]*[ \t]*$", re.MULTILINE)
_FENCE_LINE = re.compile(r"[ \t]{0,3}(`{3,}|~{3,})(.*)")
_FENCE_AFTER_TAG = re.compile(r"""(<div\b(?:[^>"']|"[^"]*"|'[^']*')*>)[ \t]*(?=`{3}|~{3})""", re.I)
_FENCE_BEFORE_TAG = re.compile(r"(`{3,}|~{3,})[ \t]*(?=</div>)", re.I)
_DELIMITER = re.compile(r"(?<!\\)(?:\\\\)*(?:\$|\\[()\[\]])")
_OPENING_FENCE = re.compile(r"[ \t]*(`{3,}|~{3,})[ \t]*(?:markdown|md|html)?[ \t]*", re.IGNORECASE)
_HEADING_MARK = re.compile(r"\A(#{1,6})[ \t]+")
_CAPTION = re.compile(r"<caption[^>]*>(.*?)</caption>", re.IGNORECASE | re.DOTALL)
_SPLIT_WORD = re.compile(r"[^\W\d_]-\Z")
_LIST_MARKER = re.compile(r"\A[ \t]*([-+*]|\d{1,3}[.)])[ \t]+(?=\S)")


def read_pages(answers: Iterable[str]) -> Prediction:
    out = _Output()
    for page, answer in enumerate(answers):
        out.read_page(page, answer)
    return out.prediction()


def previous_page_prompt(answer: str) -> str | None:
    elements, _ = _elements(_prepared(answer))
    ending = _ending(_kept(elements))
    if ending is None:
        return None
    content = ending.markdown
    if len(content) > QUOTED_CHARACTERS:
        content = content[:QUOTED_START] + QUOTED_GAP + content[-QUOTED_END:]
    label = html.escape(ending.attributes.get("data-label", "").strip(), quote=True)
    return PREVIOUS_PAGE_PROMPT.format(element=QUOTED_ELEMENT.format(label=label, content=content))


@dataclass(frozen=True)
class _Element:
    attributes: dict[str, str]
    content: str


@dataclass(frozen=True)
class _Kept:
    index: int
    attributes: dict[str, str]
    content: str
    cls: ElementClass
    markdown: str
    text: str


@dataclass(frozen=True)
class _Ending:
    id: str
    cls: ElementClass
    item: int
    span: tuple[int, int, int]


def _prepared(answer: str) -> str:
    answer = _unwrapped(answer.replace("\r\n", "\n").replace("\r", "\n"))
    return _fences_on_their_lines(answer)


def _kept(elements: list[_Element]) -> list[_Kept]:
    kept: list[_Kept] = []
    for index, element in enumerate(elements):
        label = _label(element.attributes)
        cls = _CLASSES.get(label, ElementClass.PARAGRAPH)
        content = element.content
        if cls is ElementClass.HEADING:
            markdown = _heading(label, content)
            text = markdown_text(markdown)
        elif cls is ElementClass.FORMULA and (latex := _formula(content)) is not None:
            markdown, text = f"$$\n{latex}\n$$", latex
        else:
            markdown = _closed(content)
            text = markdown_text(markdown)
        if markdown or cls in _MAY_BE_EMPTY:
            kept.append(_Kept(index, element.attributes, content, cls, markdown, text))
    return kept


def _ending(kept: list[_Kept]) -> _Kept | None:
    for element in reversed(kept):
        if element.cls not in _OUTSIDE_FLOW:
            return element if element.cls in _CONTINUING else None
    return None


def _unmarked(piece: str, item: str) -> str:
    repeated, own = _LIST_MARKER.match(piece), _LIST_MARKER.match(item)
    if repeated is None or own is None or repeated[1] != own[1]:
        return piece
    return piece[repeated.end() :]


def _plain(line: str) -> bool:
    blocks = split_blocks(line)
    return len(blocks) == 1 and blocks[0].kind is BlockKind.TEXT and bool(blocks[0].text.strip())


def _reads_joined(block: str, piece: str, joined: str, *, hyphenated: bool) -> bool:
    if _closed(joined) != joined:
        return False
    if not (_plain(block.rpartition("\n")[2]) and _plain(piece.partition("\n")[0])):
        return False
    before, after = split_blocks(block), split_blocks(piece)
    if not before or not after:
        return False
    if before[-1].kind is not BlockKind.TEXT or after[0].kind is not BlockKind.TEXT:
        return False
    text = join_pieces(before[-1].text, after[0].text, hyphenated=hyphenated)
    expected = [
        *((b.kind, b.text, b.level) for b in before[:-1]),
        (BlockKind.TEXT, text, None),
        *((b.kind, b.text, b.level) for b in after[1:]),
    ]
    return [(b.kind, b.text, b.level) for b in split_blocks(joined)] == expected


def _elements(answer: str) -> tuple[list[_Element], list[str]]:
    tags = list(_DIV.finditer(answer))
    is_wrapper = [not tag[1] and "data-label" in _attributes(tag) for tag in tags]
    wrapper_starts = [tag.start() for tag, wrapper in zip(tags, is_wrapper, strict=True) if wrapper]
    code = [
        (start, end)
        for start, end, _ in _fenced_blocks(answer)
        if not any(start <= at < end for at in wrapper_starts)
    ]
    wrappers: list[list[Any]] = []
    stack: list[int | None] = []
    for tag, wrapper in zip(tags, is_wrapper, strict=True):
        if any(start <= tag.start() < end for start, end in code):
            continue
        if not tag[1] and tag[2].rstrip().endswith("/"):
            if wrapper:
                wrappers.append([tag, tag.end(), tag.end()])
        elif not tag[1]:
            stack.append(len(wrappers) if wrapper else None)
            if wrapper:
                wrappers.append([tag, None, None])
        elif stack and (index := stack.pop()) is not None:
            wrappers[index][1:] = tag.start(), tag.end()
    for index in reversed(range(len(wrappers))):
        opening, content_end, _ = wrappers[index]
        if content_end is not None:
            continue
        position = opening.end()
        stop = len(answer)
        for inner, _, inner_end in wrappers[index + 1 :]:
            if inner.start() < position:
                continue
            if _DIV.sub("", answer[position : inner.start()]).strip():
                stop = inner.start()
                break
            position = inner_end
        wrappers[index][1:] = [stop, stop]
    placed: list[tuple[int, _Element]] = []
    for opening, content_end, _ in wrappers:
        pieces: list[tuple[int, str]] = []
        position = opening.end()
        for inner, _, inner_end in wrappers:
            if position <= inner.start() and inner_end <= content_end:
                pieces.append((position, answer[position : inner.start()]))
                position = inner_end
        pieces.append((position, answer[position:content_end]))
        own = [(at + len(piece) - len(piece.lstrip()), piece) for at, piece in pieces]
        own = [(at, piece.strip()) for at, piece in own if piece.strip()]
        content = "\n".join(piece for _, piece in own)
        placed.append(
            (own[0][0] if own else opening.start(), _Element(_attributes(opening), content))
        )
    elements = [element for _, element in sorted(placed, key=lambda pair: pair[0])]
    outside: list[str] = []
    position = 0
    for opening, _, end in [*wrappers, [None, 0, len(answer)]]:
        start = len(answer) if opening is None else opening.start()
        if start >= position:
            text = _FENCE.sub("", _DIV.sub("", answer[position:start])).strip()
            if text:
                outside.append(text)
        position = max(position, end)
    return elements, outside


def _fences_on_their_lines(answer: str) -> str:
    lines = _FENCE_AFTER_TAG.sub("\\1\n", answer).split("\n")
    opening: str | None = None
    for index, line in enumerate(lines):
        starts = [
            t.start() for t in _DIV.finditer(line) if not t[1] and "data-label" in _attributes(t)
        ]
        if opening is not None and (closing := _FENCE_BEFORE_TAG.search(line)) is not None:
            same = closing[1][0] == opening[0] and len(closing[1]) >= len(opening)
            if same and not any(at < closing.start() for at in starts):
                before, after = line[: closing.start()].rstrip(), line[closing.end() :]
                lines[index] = "\n".join(part for part in (before, closing[1], after) if part)
                opening = None
                continue
        if starts:
            opening = None
        if (fence := _FENCE_LINE.fullmatch(line)) is not None:
            mark, rest = fence[1], fence[2]
            if opening is None:
                if not (mark[0] == "`" and "`" in rest):
                    opening = mark
            elif mark[0] == opening[0] and len(mark) >= len(opening) and not rest.strip():
                opening = None
    return "\n".join(lines)


def _fenced_blocks(text: str) -> list[tuple[int, int, str | None]]:
    blocks: list[tuple[int, int, str | None]] = []
    opening: tuple[int, str] | None = None
    position = 0
    for line in text.splitlines(keepends=True):
        if (fence := _FENCE_LINE.fullmatch(line.rstrip("\n"))) is not None:
            mark, rest = fence[1], fence[2]
            if opening is None:
                if not (mark[0] == "`" and "`" in rest):
                    opening = (position, mark)
            elif mark[0] == opening[1][0] and len(mark) >= len(opening[1]) and not rest.strip():
                blocks.append((opening[0], position + len(line), None))
                opening = None
        position += len(line)
    if opening is not None:
        blocks.append((opening[0], len(text), opening[1]))
    return blocks


def _closed(markdown: str) -> str:
    blocks = _fenced_blocks(markdown)
    if not blocks or (fence := blocks[-1][2]) is None:
        return markdown
    start = blocks[-1][0]
    if "\n" not in markdown[start:].strip():
        return markdown[:start].rstrip()
    return f"{markdown.rstrip()}\n{fence}"


def _attributes(tag: re.Match[str]) -> dict[str, str]:
    return {
        name.lower(): double if double or not single else single
        for name, double, single in _ATTRIBUTE.findall(tag[2])
    }


def _mrz(attributes: dict[str, str]) -> dict[str, Any]:
    if _label(attributes) != "mrz":
        return {}
    verdict = attributes.get("data-check-digits", "").strip().casefold()
    valid = {"valid": True, "invalid": False}.get(verdict)
    return {"format": "Mrz", "check_digits_valid": valid}


def _label(attributes: dict[str, str]) -> str:
    return re.sub(r"[\s_]+", "-", attributes.get("data-label", "").strip().lower())


def _font(attributes: dict[str, str]) -> Font | None:
    family = attributes.get("data-font-family", "").strip() or None
    name = attributes.get("data-font-class", "").strip().casefold()
    font_class = FontClass(name) if name in FontClass._value2member_map_ else None
    try:
        size: float | None = float(attributes.get("data-font-size", "").strip().removesuffix("pt"))
    except ValueError:
        size = None
    if size is not None and not (isfinite(size) and size > 0):
        size = None
    if family is None and font_class is None and size is None:
        return None
    return Font(family=family, font_class=font_class, size=size)


def _box(attributes: dict[str, str]) -> Box | None:
    try:
        values = json.loads(attributes.get("data-bbox", ""))
    except ValueError:
        return None
    if not isinstance(values, list) or len(values) != 4:
        return None
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return None
    x1, y1, x2, y2 = (float(v) for v in values)
    if not all(isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    return clip_box(x1, y1, x2 - x1, y2 - y1, page_width=GRID, page_height=GRID)


def _heading(label: str, content: str) -> str:
    line = " ".join(content.split())
    if not line:
        return ""
    mark = _HEADING_MARK.match(line)
    level = len(mark[1]) if mark else 1 if label == "title" else 2
    text = line[mark.end() :] if mark else line
    return f"{'#' * level} {text or line}"


def _formula(content: str) -> str | None:
    latex = content.strip()
    for opening, closing in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if latex.startswith(opening) and latex.endswith(closing):
            if len(latex) >= len(opening) + len(closing):
                latex = latex[len(opening) : len(latex) - len(closing)].strip()
                break
    if not latex or _DELIMITER.search(latex):
        return None
    return latex


def _chart(attributes: dict[str, str], content: str) -> Chart:
    name = re.sub(r"[\s-]+", "_", attributes.get("data-chart-type", "").strip().lower())
    kind = ChartKind(name) if name in ChartKind else None
    caption = _CAPTION.search(content)
    title = markdown_text(caption[1]).strip() if caption else ""
    return Chart(kind=kind, title=title or None)


def _unwrapped(answer: str) -> str:
    first, *lines = answer.strip().split("\n")
    if (opening := _OPENING_FENCE.fullmatch(first)) is None:
        return answer
    fence = opening[1]
    for index, line in enumerate(lines):
        mark = line.strip()
        if len(mark) >= len(fence) and set(mark) == {fence[0]}:
            return "\n".join(lines[:index]) if index == len(lines) - 1 else answer
    return answer


class _Output:
    def __init__(self) -> None:
        self.blocks: list[str] = []
        self.layout: list[LayoutItem] = []
        self.spans: list[tuple[int, int, int] | None] = []
        self.relations: list[Relation] = []
        self.unscored: list[str] = []
        self.page_starts: list[tuple[int, int] | None] = []
        self.ending: _Ending | None = None

    def prediction(self) -> Prediction:
        markdown = "\n\n".join(self.blocks) + ("\n" if self.blocks else "")
        starts = list(accumulate((len(block) + 2 for block in self.blocks), initial=0))
        layout = list(self.layout)
        for index, span in enumerate(self.spans):
            if span is not None:
                block, start, end = span
                layout[index] = layout[index].model_copy(
                    update={"md_start": starts[block] + start, "md_end": starts[block] + end}
                )
        offsets: list[int] = []
        following = len(markdown)
        for page_start in reversed(self.page_starts):
            if page_start is not None:
                following = starts[page_start[0]] + page_start[1]
            offsets.append(following)
        return Prediction(
            markdown=markdown,
            layout=layout,
            relations=self.relations,
            page_offsets=offsets[::-1],
            unscored_text=self.unscored,
        )

    def write(self, text: str) -> tuple[int, int, int]:
        self.blocks.append(text)
        span = (len(self.blocks) - 1, 0, len(text))
        if self.page_starts[-1] is None:
            self.page_starts[-1] = span[:2]
        return span

    def go_on(self, ending: _Ending, text: str) -> tuple[tuple[int, int, int], bool] | None:
        index, start, _ = ending.span
        block = self.blocks[index]
        hyphenated = _SPLIT_WORD.search(block) is not None
        boundary = join_boundary(block, text, hyphenated=hyphenated)
        head = block[: boundary.first_end] + boundary.separator
        joined = head + text[boundary.second_start :]
        if not _reads_joined(block, text, joined, hyphenated=hyphenated):
            return None
        self.blocks[index] = joined
        self.spans[ending.item] = (index, start, boundary.first_end)
        span = (index, len(head), len(joined))
        if self.page_starts[-1] is None:
            self.page_starts[-1] = span[:2]
        return span, hyphenated

    def read_page(self, page: int, answer: str) -> None:
        self.page_starts.append(None)
        previous, self.ending = self.ending, None
        answer = _prepared(answer)
        elements, outside = _elements(answer)
        if not elements:
            if markdown := _closed(answer.strip()):
                self.write(markdown)
            return
        self.unscored.extend(outside)
        kept = _kept(elements)
        opening = next((e for e in kept if e.cls not in _FURNITURE), None)
        continued = (
            previous
            if previous is not None
            and opening is not None
            and (previous.cls, opening.cls) in _CONTINUATIONS
            and opening.attributes.get("data-continues-previous", "").strip().casefold() == "true"
            else None
        )
        ending = _ending(kept)
        ids: dict[str, str] = {}
        links: list[tuple[RelationKind, str, str]] = []
        for element in kept:
            attributes, cls = element.attributes, element.cls
            item_id = f"p{page + 1}-{element.index + 1}"
            if local := attributes.get("data-id", "").strip():
                ids.setdefault(local, item_id)
            span, hyphenated, text = None, False, element.text
            if element is opening and continued is not None:
                if cls is not ElementClass.TABLE:
                    piece = element.markdown
                    if continued.cls is ElementClass.LIST_ITEM:
                        piece = _unmarked(piece, self.blocks[continued.span[0]])
                    if (written := self.go_on(continued, piece)) is not None:
                        span, hyphenated = written
                        text = markdown_text(piece)
                self.relations.append(
                    Relation(
                        kind=RelationKind.CONTINUES_IN,
                        source=continued.id,
                        target=item_id,
                        hyphenated=hyphenated,
                    )
                )
            if span is None and element.markdown and cls not in _FURNITURE:
                span = self.write(element.markdown)
            self.layout.append(
                LayoutItem(
                    id=item_id,
                    cls=cls,
                    page=page,
                    text=text,
                    box=_box(attributes),
                    channel=Channel.FURNITURE if cls in _FURNITURE else Channel.BODY,
                    font=_font(attributes),
                    chart=_chart(attributes, element.content)
                    if cls is ElementClass.CHART
                    else None,
                    **_mrz(attributes),
                )
            )
            self.spans.append(span)
            if element is ending and span is not None:
                self.ending = _Ending(item_id, cls, len(self.layout) - 1, span)
            if cls in _LINKS:
                attribute, kind = _LINKS[cls]
                if target := attributes.get(attribute, "").strip():
                    links.append((kind, item_id, target))
        for kind, source, target in links:
            if (target_id := ids.get(target)) is not None and target_id != source:
                self.relations.append(Relation(kind=kind, source=source, target=target_id))
