import html
import re
import string
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from html.entities import html5
from html.parser import HTMLParser

from wholedoc_bench.schema import CellSpan, StyleName
from wholedoc_bench.scorer.utils.prose import normalize

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_INLINE_HTML_TAGS = (
    "a abbr b big cite code del dfn em font i ins kbd mark q s samp small span"
    " strike strong sub sup tt u var"
).split()
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_LIST_ITEM = re.compile(r" *(?:[-+*]|\d{1,3}[.)]) {1,4}(?=\S)")
_LIST_ITEM_START = re.compile(r"[ \t]*(?:[-+*]|\d{1,3}[.)])[ \t]+(?=\S)")
_CODE_SPAN_SOURCE = r"(?<!`)(?P<ticks>`+)(?!`)(?P<code>(?:(?!\n[ \t]*\n).)+?)(?<!`)(?P=ticks)(?!`)"
_BACKSLASH_ESCAPE_SOURCE = rf"\\(?P<escaped>[{re.escape(string.punctuation)}])"
_CODE_SPAN = re.compile(_CODE_SPAN_SOURCE, re.DOTALL)
_MARKDOWN_LITERAL = re.compile(f"{_BACKSLASH_ESCAPE_SOURCE}|{_CODE_SPAN_SOURCE}", re.DOTALL)

Cells = tuple[str, ...]
Grid = tuple[Cells, ...]

MAX_COLSPAN = 1000
MAX_ROWSPAN = 65534

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}(?=[^`]*$)|~{3,})")
_FENCE_CLOSE = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*$")
_HTML_TABLE_TAG = re.compile(r"<(/?)table\b[^>]*>", re.IGNORECASE)
_HTML_ROW_OR_CELL_TAG = re.compile(r"</?(?:tr|td|th)\b[^>]*>", re.IGNORECASE)
_QUOTE_MARKERS = re.compile(r"(?: {0,3}> ?)*")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")
_ESCAPED_PIPE = "\\|"
_UNNUMBERED = -1
_DELIMITER_CELL = re.compile(r":?-+:?")
_DIGITS = re.compile(r"\d+", re.ASCII)
_CARRIAGE_RETURN = "\r"
_CODE_INDENT = 4
_TAB_STOP = 4


class BlockKind(StrEnum):
    TABLE = "table"
    HEADING = "heading"
    TEXT = "text"


@dataclass(frozen=True)
class MarkdownTable:
    index: int
    start: int
    end: int
    header: Grid
    body: Grid
    source_cells: tuple[str, ...] = ()
    merged: tuple[CellSpan, ...] = ()

    @property
    def rows(self) -> Grid:
        return self.header + self.body


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    source: str
    text: str
    table: MarkdownTable | None = None
    level: int | None = None
    styles: tuple[frozenset[StyleName], ...] = ()


def comparable_text(text: str) -> str:
    return normalize(text)


@dataclass(frozen=True)
class ParsedDocument:
    blocks: list[Block]
    document_text: str
    comment_blocks: list[str]


def parse_markdown(markdown: str) -> ParsedDocument:
    blocks, comment_blocks = _read_blocks(markdown)
    return ParsedDocument(blocks, " ".join(block.text for block in blocks), comment_blocks)


def split_blocks(markdown: str) -> list[Block]:
    return _read_blocks(markdown)[0]


def _read_blocks(markdown: str) -> tuple[list[Block], list[str]]:
    raw_positions = _raw_positions(markdown)
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[Block] = []
    comment_blocks: list[str] = []

    def read_text(section: str) -> None:
        reader = _BlockReader(section)
        blocks.extend(reader.read())
        comment_blocks.extend(reader.comment_blocks)

    position = 0
    for table in parse_markdown_tables(markdown):
        start, end = table.start, table.end
        read_text(markdown[position:start])
        blocks.append(_block(BlockKind.TABLE, markdown[start:end], _in_raw(table, raw_positions)))
        position = end
    read_text(markdown[position:])
    return blocks, comment_blocks


_TAG_STYLES = {
    "b": StyleName.BOLD,
    "strong": StyleName.BOLD,
    "i": StyleName.ITALIC,
    "em": StyleName.ITALIC,
    "s": StyleName.STRIKETHROUGH,
    "del": StyleName.STRIKETHROUGH,
    "strike": StyleName.STRIKETHROUGH,
    "u": StyleName.UNDERLINE,
    "ins": StyleName.UNDERLINE,
    "sup": StyleName.SUPERSCRIPT,
    "sub": StyleName.SUBSCRIPT,
}
_SUPERSCRIPT = frozenset("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUBSCRIPT = frozenset("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜᵢⱼᵣᵤᵥ")


@dataclass
class _Delimiter:
    marker: str
    start: int
    remaining: int
    can_open: bool
    can_close: bool
    length: int


def _delimiter(source: str, start: int, end: int, output: int) -> _Delimiter:
    before = source[start - 1] if start else " "
    after = source[end] if end < len(source) else " "
    before_punct = unicodedata.category(before)[0] in "PS"
    after_punct = unicodedata.category(after)[0] in "PS"
    left = not after.isspace() and (not after_punct or before.isspace() or before_punct)
    right = not before.isspace() and (not before_punct or after.isspace() or after_punct)
    if source[start] == "_":
        left, right = left and (not right or before_punct), right and (not left or after_punct)
    return _Delimiter(source[start], output, end - start, left, right, end - start)


def _resolve_emphasis(
    chars: list[str],
    memberships: list[frozenset[StyleName]],
    delimiters: list[_Delimiter],
) -> tuple[str, tuple[frozenset[StyleName], ...]]:
    removed: set[int] = set()
    content = [0]
    for char in chars:
        content.append(content[-1] + (not char.isspace() and char not in "*_~"))
    openers: list[_Delimiter] = []
    for closer in delimiters:
        while closer.can_close and closer.remaining:
            opener_index = next(
                (
                    i
                    for i in range(len(openers) - 1, -1, -1)
                    if openers[i].marker == closer.marker
                    and (
                        openers[i].remaining == closer.remaining
                        or content[closer.start] > content[openers[i].start + openers[i].remaining]
                    )
                    and not (
                        (openers[i].can_close or closer.can_open)
                        and (openers[i].length + closer.length) % 3 == 0
                        and (openers[i].length % 3 or closer.length % 3)
                    )
                ),
                None,
            )
            if opener_index is None:
                break
            opener = openers[opener_index]
            count = 2 if min(opener.remaining, closer.remaining) >= 2 else 1
            if closer.marker == "~" and count != 2:
                break
            style = (
                StyleName.STRIKETHROUGH
                if closer.marker == "~"
                else StyleName.BOLD
                if count == 2
                else StyleName.ITALIC
            )
            opening_end = opener.start + opener.remaining
            removed.update(range(opening_end - count, opening_end))
            removed.update(range(closer.start, closer.start + count))
            for index in range(opening_end, closer.start):
                memberships[index] = memberships[index] | {style}
            opener.remaining -= count
            closer.remaining -= count
            closer.start += count
            del openers[opener_index + bool(opener.remaining) :]
        if closer.can_open and closer.remaining:
            openers.append(closer)
    return (
        "".join(char for index, char in enumerate(chars) if index not in removed),
        tuple(styles for index, styles in enumerate(memberships) if index not in removed),
    )


_ENTITY = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]*);")
_HTML_ENTITY = re.compile(r"&(?:#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[a-zA-Z][a-zA-Z0-9]+;?)")
_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)\b[^<>]*>")
_PLAIN_INLINE = re.compile(r"[^`\\<!\[*_~&]+")


def _decode_inline(
    source: str,
    inherited: frozenset[StyleName] = frozenset(),
    *,
    html_mode: bool = False,
) -> tuple[str, tuple[frozenset[StyleName], ...]]:
    chars: list[str] = []
    memberships: list[frozenset[StyleName]] = []
    active: list[str] = []
    delimiters: list[_Delimiter] = []
    styles = inherited

    def append(text: str, styles: frozenset[StyleName], *, literal: bool = False) -> None:
        for char in text:
            extra = (
                (
                    frozenset((StyleName.SUPERSCRIPT,))
                    if char in _SUPERSCRIPT
                    else frozenset((StyleName.SUBSCRIPT,))
                    if char in _SUBSCRIPT
                    else frozenset()
                )
                if not literal
                else frozenset()
            )
            chars.append(char)
            memberships.append(styles | extra)

    i = 0
    while i < len(source):
        if match := (_CODE_SPAN if html_mode else _MARKDOWN_LITERAL).match(source, i):
            append(_literal_text(match), styles, literal=True)
            i = match.end()
            continue
        if match := _HTML_COMMENT.match(source, i):
            i = match.end()
            continue
        if match := _IMAGE.match(source, i):
            i = match.end()
            continue
        if match := _LINK.match(source, i):
            text, nested = _decode_inline(match[1], styles, html_mode=html_mode)
            chars.extend(text)
            memberships.extend(nested)
            i = match.end()
            continue
        if match := _TAG.match(source, i):
            tag = match[2].lower()
            if match[1]:
                if tag in active:
                    active.reverse()
                    active.remove(tag)
                    active.reverse()
            elif tag in _TAG_STYLES:
                active.append(tag)
            if tag not in _INLINE_HTML_TAGS:
                append(" ", styles)
            styles = inherited | frozenset(_TAG_STYLES[t] for t in active)
            i = match.end()
            continue
        if source[i] in "*_~":
            end = i + 1
            while end < len(source) and source[end] == source[i]:
                end += 1
            run = end - i
            if run <= 3 and (source[i] != "~" or run == 2):
                delimiters.append(_delimiter(source, i, end, len(chars)))
            append(source[i:end], styles)
            i = end
        elif match := _entity_at(source, i, html_mode=html_mode):
            append(html.unescape(match[0]), styles)
            i = match.end()
        else:
            match = _PLAIN_INLINE.match(source, i)
            end = match.end() if match else i + 1
            append(source[i:end], styles)
            i = end
    return _resolve_emphasis(chars, memberships, delimiters)


def _entity_at(source: str, i: int, *, html_mode: bool) -> re.Match[str] | None:
    if html_mode:
        return _HTML_ENTITY.match(source, i)
    match = _ENTITY.match(source, i)
    return match if match and (match[0][1] == "#" or match[0][1:] in html5) else None


def _inline_text(source: str) -> str:
    return _decode_inline(source)[0]


@dataclass(frozen=True)
class WrittenSource:
    text: str
    unseen: tuple[str, ...]


def written_source(source: str) -> WrittenSource:
    if _FENCE_OPEN.match(source):
        return WrittenSource(source, ())
    unseen: list[str] = []

    def cut_comment(match: re.Match[str]) -> str:
        unseen.append(match[0][4:-3])
        return ""

    def cut_target(match: re.Match[str]) -> str:
        unseen.append(match[2])
        return match[1]

    def cut_tag(match: re.Match[str]) -> str:
        unseen.append(match[0])
        return "" if match[2].lower() in _INLINE_HTML_TAGS else " "

    def written(segment: str) -> str:
        text = _HTML_COMMENT.sub(cut_comment, segment)
        return _TAG.sub(cut_tag, _LINK.sub(cut_target, _IMAGE.sub(cut_target, text)))

    pieces: list[str] = []
    start = 0
    for code in _CODE_SPAN.finditer(source):
        pieces += [written(source[start : code.start()]), code[0]]
        start = code.end()
    pieces.append(written(source[start:]))
    return WrittenSource("".join(pieces), tuple(unseen))


def markdown_text(markdown: str) -> str:
    return " ".join(b.text for b in split_blocks(markdown))


def _block(
    kind: BlockKind,
    source: str,
    table: MarkdownTable | None = None,
    *,
    content: str | None = None,
    level: int | None = None,
    literal: bool = False,
) -> Block:
    if table is not None:
        text = " ".join(table.source_cells or tuple(c for row in table.rows for c in row))
        styles = tuple(frozenset() for _ in text)
    else:
        content = source if content is None else content
        if literal:
            text, styles = content, tuple(frozenset() for _ in content)
        else:
            text, styles = _decode_inline(content)
    return Block(kind, source, text, table, level, styles)


def _raw_positions(markdown: str) -> list[int]:
    return [i for i in range(len(markdown)) if not markdown.startswith("\r\n", i)]


def _in_raw(table: MarkdownTable, raw_positions: Sequence[int]) -> MarkdownTable:
    return replace(table, start=raw_positions[table.start], end=raw_positions[table.end - 1] + 1)


_THEMATIC_BREAK = re.compile(r"^ {0,3}(?:(?:\* *){3,}|(?:_ *){3,}|(?:- *){3,})$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_HTML_HEADING = re.compile(r"^ {0,3}<h([1-6])\b[^>]*>(.*?)</h\1>[ \t]*$", re.I | re.S)
_ATX_CONTENT = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
_CLOSING_HASHES = re.compile(r"[ \t]+#+[ \t]*$")
_BULLET = re.compile(r"^[ \t]*(?:[-+*])[ \t]+(?=\S)")


class _BlockReader:
    def __init__(self, markdown: str) -> None:
        self.lines = markdown.split("\n")
        comments = [
            m.span()
            for m in _HTML_COMMENT.finditer(
                _blank(
                    markdown,
                    _fenced_ranges(markdown) + [m.span() for m in _CODE_SPAN.finditer(markdown)],
                )
            )
        ]
        visible: list[str] = []
        previous = 0
        for start, end in comments:
            visible.extend((markdown[previous:start], "\n" * markdown[start:end].count("\n")))
            previous = end
        visible.append(markdown[previous:])
        self.visible = "".join(visible).split("\n")
        spans = iter(comments)
        current = next(spans, None)

        def inside(position: int) -> bool:
            nonlocal current
            while current is not None and current[1] <= position:
                current = next(spans, None)
            return current is not None and current[0] <= position < current[1]

        self.blank: list[bool] = []
        self.separators = [0]
        offset = 0
        for line in self.lines:
            starts_in_comment = inside(offset)
            self.blank.append(not line.strip() and not starts_in_comment)
            newline = offset + len(line)
            outside = not inside(newline)
            self.separators.append(self.separators[-1] + outside)
            offset = newline + 1
        self.blocks: list[Block] = []
        self.comment_blocks: list[str] = []
        self.comment_lines: list[str] = []
        self.pending: list[str] = []
        self.raw: list[str] = []
        self.last_pending = 0
        self.pending_list = False

    def separator(self, start: int, end: int) -> str:
        return "\n" if self.separators[end] > self.separators[start] else ""

    def append(self, content: str, index: int) -> None:
        if self.pending:
            self.pending.append(self.separator(self.last_pending, index))
        else:
            self.raw.extend(self.comment_lines)
            self.comment_lines.clear()
        self.pending.append(content)
        self.raw.append(self.lines[index])
        self.last_pending = index

    def comment_line(self, index: int) -> None:
        if self.pending:
            self.raw.append(self.lines[index])
        else:
            self.comment_lines.append(self.lines[index])

    def flush(self) -> None:
        if self.pending:
            self.blocks.append(
                _block(BlockKind.TEXT, "\n".join(self.raw), content="".join(self.pending))
            )
        if self.comment_lines:
            self.comment_blocks.append("\n".join(self.comment_lines))
            self.comment_lines.clear()
        self.clear()

    def clear(self) -> None:
        self.pending.clear()
        self.raw.clear()
        self.pending_list = False

    def fence(self, index: int, opening: str, depth: int) -> int:
        self.flush()
        end = index + 1
        while end < len(self.lines):
            closing = _FENCE_CLOSE.match(_without_quote_depth(self.lines[end], depth))
            if closing and _closes(opening, closing[1]):
                break
            end += 1
        self.blocks.append(
            _block(
                BlockKind.TEXT,
                "\n".join(self.lines[index : min(end + 1, len(self.lines))]),
                content="\n".join(
                    _without_quote_depth(value, depth) for value in self.lines[index + 1 : end]
                ),
                literal=True,
            )
        )
        return end + 1

    def html_heading(self, index: int, level: str) -> int | None:
        closing = re.compile(rf"</h{level}>", re.I)
        parts: list[str] = []
        for end in range(index, len(self.lines)):
            line = _QUOTE_MARKERS.sub("", self.visible[end], count=1)
            if end > index and (
                self.blank[end]
                or _ATX_CONTENT.match(line)
                or re.match(r"^ {0,3}<h[1-6]\b", line, re.I)
            ):
                return None
            if end > index:
                parts.append(self.separator(end - 1, end))
            parts.append(line)
            if close := closing.search(line):
                if line[close.end() :].strip():
                    return None
                heading = _HTML_HEADING.fullmatch("".join(parts))
                if heading is None:
                    return None
                self.flush()
                self.blocks.append(
                    _block(
                        BlockKind.HEADING,
                        "\n".join(self.lines[index : end + 1]),
                        content=heading[2].replace("\n", " "),
                        level=int(level),
                    )
                )
                return end + 1
        return None

    def read(self) -> list[Block]:
        index = 0
        while index < len(self.lines):
            source, line = self.lines[index], self.visible[index]
            fence_line = _unquoted(_Line(0, line))
            if opening := _FENCE_OPEN.match(fence_line.text):
                index = self.fence(index, opening[1], fence_line.quote_depth)
                continue
            if not line.strip():
                if self.blank[index]:
                    self.flush()
                else:
                    self.comment_line(index)
                index += 1
                continue
            line = _QUOTE_MARKERS.sub("", line, count=1)
            if heading := _ATX_CONTENT.match(line):
                self.flush()
                self.blocks.append(
                    _block(
                        BlockKind.HEADING,
                        source,
                        content=_CLOSING_HASHES.sub("", heading[2]),
                        level=len(heading[1]),
                    )
                )
            elif opening := re.match(r"^ {0,3}<h([1-6])\b", line, re.I):
                following = self.html_heading(index, opening[1])
                if following is not None:
                    index = following
                    continue
                self.append(line, index)
            elif (
                index + 1 < len(self.lines)
                and not self.pending_list
                and not _LIST_ITEM_START.match(line)
                and not _THEMATIC_BREAK.fullmatch(line)
                and (
                    underline := _SETEXT.match(
                        _QUOTE_MARKERS.sub("", self.visible[index + 1], count=1)
                    )
                )
            ):
                self.append(line, index)
                self.raw.append(self.lines[index + 1])
                self.blocks.append(
                    _block(
                        BlockKind.HEADING,
                        "\n".join(self.raw),
                        content="".join(self.pending).replace("\n", " "),
                        level=1 if underline[1][0] == "=" else 2,
                    )
                )
                self.clear()
                index += 1
            elif _THEMATIC_BREAK.fullmatch(line):
                self.flush()
                if source != self.visible[index]:
                    self.comment_line(index)
            else:
                if _LIST_ITEM_START.match(line):
                    self.flush()
                    self.pending_list = True
                self.append(_BULLET.sub("", line), index)
            index += 1
        self.flush()
        return self.blocks


class Markup(StrEnum):
    NONE = "none"
    HTML = "html"
    MARKDOWN = "markdown"


def normalize_text(text: str, *, markup: Markup = Markup.NONE, fold_case: bool = False) -> str:
    if markup is Markup.MARKDOWN:
        text = _inline_text(text)
    elif markup is Markup.HTML:
        text = _decode_inline(text, html_mode=True)[0]
    normalized = normalize(text)
    return normalized.casefold() if fold_case else normalized


def _literal_text(match: re.Match[str]) -> str:
    groups = match.groupdict()
    code = groups["code"]
    return groups["escaped"] if code is None else code


@dataclass(frozen=True)
class _Line:
    offset: int
    text: str
    quote_depth: int = 0


def parse_markdown_tables(markdown: str) -> tuple[MarkdownTable, ...]:
    unfenced = _blank(markdown, _fenced_ranges(markdown))
    unfenced = _blank(unfenced, [m.span() for m in _HTML_COMMENT.finditer(unfenced)])
    html_search = _blank(unfenced, [m.span() for m in _CODE_SPAN.finditer(unfenced)])
    html_tables = [
        table
        for start, end in _html_ranges(html_search)
        if (table := _parse_html_table(markdown, start, end)).rows
    ]
    pipe_source = _blank(unfenced, [(t.start, t.end) for t in html_tables])
    parsed = sorted(
        [
            *html_tables,
            *_parse_pipe_tables(
                pipe_source, _blank(markdown, [(t.start, t.end) for t in html_tables])
            ),
        ],
        key=lambda t: t.start,
    )
    return tuple(replace(t, index=i) for i, t in enumerate(t for t in parsed if t.rows))


def _lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    for line in text.split("\n"):
        lines.append(_Line(offset, line.removesuffix(_CARRIAGE_RETURN)))
        offset += len(line) + 1
    return lines


def without_quote_markers(text: str) -> str:
    return _QUOTE_MARKERS.sub("", text, count=1)


def _unquoted(line: _Line) -> _Line:
    text = _QUOTE_MARKERS.sub("", line.text, count=1)
    markers = line.text[: len(line.text) - len(text)]
    return _Line(line.offset + len(markers), text, markers.count(">"))


def _without_quote_depth(text: str, depth: int) -> str:
    for _ in range(depth):
        match = re.match(r" {0,3}> ?", text)
        if match is None:
            break
        text = text[match.end() :]
    return text


def _fenced_ranges(markdown: str) -> list[tuple[int, int]]:
    fence_lines = _fence_lines(markdown)
    openings = fence_lines[::2]
    ends = [line.offset + len(line.text) for line in fence_lines[1::2]] + [len(markdown)]
    return [(opening.offset, end) for opening, end in zip(openings, ends, strict=False)]


def _fence_lines(markdown: str) -> list[_Line]:
    fence_lines: list[_Line] = []
    fence: str | None = None
    depth = 0
    for line in _lines(markdown):
        if fence is None:
            unquoted = _unquoted(line)
            if opening := _FENCE_OPEN.match(unquoted.text):
                fence = opening.group(1)
                depth = unquoted.quote_depth
                fence_lines.append(line)
        elif (closing := _FENCE_CLOSE.match(_without_quote_depth(line.text, depth))) and _closes(
            fence, closing.group(1)
        ):
            fence_lines.append(line)
            fence = None
    return fence_lines


def _closes(opening: str, closing: str) -> bool:
    return closing[0] == opening[0] and len(closing) >= len(opening)


def _blank(text: str, ranges: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in ranges:
        for i in range(start, end):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _html_ranges(text: str) -> list[tuple[int, int]]:
    tags = list(_HTML_TABLE_TAG.finditer(text))
    opening = [i for i, tag in enumerate(tags) if not tag.group(1)]
    ranges: list[tuple[int, int]] = []
    position = 0
    for later, open_index in enumerate(opening, start=1):
        if open_index < position:
            continue
        close_index = _matching_close(tags, open_index)
        if close_index is not None:
            ranges.append((tags[open_index].start(), tags[close_index].end()))
            position = close_index + 1
        elif later == len(opening):
            start = tags[open_index].start()
            ranges.append((start, _unclosed_table_end(text, start)))
    return ranges


def _unclosed_table_end(text: str, start: int) -> int:
    tags = _HTML_ROW_OR_CELL_TAG.finditer(text, start)
    last_tag_end = max((tag.end() for tag in tags), default=start)
    blank_line = _BLANK_LINE.search(text, last_tag_end)
    return len(text) if blank_line is None else blank_line.start()


def _matching_close(tags: Sequence[re.Match[str]], open_index: int) -> int | None:
    depth = 0
    for i in range(open_index, len(tags)):
        depth += -1 if tags[i].group(1) else 1
        if depth == 0:
            return i
    return None


def _parse_pipe_tables(text: str, raw: str | None = None) -> list[MarkdownTable]:
    lines = [_unquoted(line) for line in _lines(text)]
    raw_lines = [_unquoted(line) for line in _lines(raw if raw is not None else text)]
    tables: list[MarkdownTable] = []
    i = 0
    while i + 1 < len(lines):
        header_line, delimiter_line = lines[i], lines[i + 1]
        depth = header_line.quote_depth
        if not _starts_pipe_table(lines, i):
            i += 1
            continue
        columns = len(_split_pipe_row(delimiter_line.text))
        end_line = i + 2
        while end_line < len(lines) and _is_pipe_row(lines[end_line], depth):
            end_line += 1
        header = _fit_row(_split_pipe_row(raw_lines[i].text), columns)
        body = tuple(
            _fit_row(_split_pipe_row(line.text), columns) for line in raw_lines[i + 2 : end_line]
        )
        last_line = lines[end_line - 1]
        tables.append(
            MarkdownTable(
                index=_UNNUMBERED,
                start=header_line.offset,
                end=last_line.offset + len(last_line.text),
                header=(header,) if any(header) else (),
                body=body,
                source_cells=tuple(c for row in ((header,) + body) for c in row),
            )
        )
        i = end_line
    return tables


def _starts_pipe_table(lines: Sequence[_Line], index: int) -> bool:
    header_line, delimiter_line = lines[index], lines[index + 1]
    depth = header_line.quote_depth
    return (
        _is_pipe_row(header_line, depth)
        and _is_delimiter_row(delimiter_line, depth)
        and not _in_indented_code(lines, index)
    )


def _in_indented_code(lines: Sequence[_Line], index: int) -> bool:
    while _is_code_indented(lines, index):
        if index == 0 or _is_blank(lines[index - 1]):
            return True
        index -= 1
    return False


def _is_code_indented(lines: Sequence[_Line], index: int) -> bool:
    line = lines[index]
    return not _is_blank(line) and _indent(line) - _content_column(lines, index) >= _CODE_INDENT


def _content_column(lines: Sequence[_Line], index: int) -> int:
    indent = _indent(lines[index])
    for line in reversed(lines[:index]):
        if _is_blank(line):
            continue
        item = _LIST_ITEM.match(line.text.expandtabs(_TAB_STOP))
        if item is not None and item.end() <= indent:
            return item.end()
        if _indent(line) == 0:
            return 0
    return 0


def _indent(line: _Line) -> int:
    text = line.text.expandtabs(_TAB_STOP)
    return len(text) - len(text.lstrip(" "))


def _is_blank(line: _Line) -> bool:
    return not line.text.strip()


def _is_pipe_row(line: _Line, quote_depth: int) -> bool:
    return (
        line.quote_depth == quote_depth
        and bool(line.text.strip())
        and _UNESCAPED_PIPE.search(line.text) is not None
    )


def _is_delimiter_row(line: _Line, quote_depth: int) -> bool:
    if line.quote_depth != quote_depth or "|" not in line.text:
        return False
    return all(_DELIMITER_CELL.fullmatch(cell) for cell in _split_pipe_row(line.text))


def _split_pipe_row(line: str) -> list[str]:
    content = line.strip().removeprefix("|")
    if content.endswith("|") and not content.endswith(_ESCAPED_PIPE):
        content = content[:-1]
    return [cell.strip() for cell in _UNESCAPED_PIPE.split(content)]


def _fit_row(cells: Sequence[str], columns: int) -> Cells:
    fitted = list(cells[:columns]) + [""] * (columns - len(cells))
    return tuple(
        normalize_text(cell.replace(_ESCAPED_PIPE, "|"), markup=Markup.MARKDOWN) for cell in fitted
    )


@dataclass
class _HtmlCell:
    is_header: bool
    colspan: int
    rowspan: int
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return normalize_text("".join(self.parts), markup=Markup.HTML)


@dataclass
class _HtmlRow:
    in_thead: bool
    cells: list[_HtmlCell] = field(default_factory=list)

    @property
    def is_all_header(self) -> bool:
        return bool(self.cells) and all(cell.is_header for cell in self.cells)


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[_HtmlRow] = []
        self._depth = 0
        self._in_thead = False
        self._in_caption = False
        self._row: _HtmlRow | None = None
        self._cell: _HtmlCell | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._depth += 1
            self._add_text(" ")
        elif self._depth != 1:
            self._add_separator(tag)
        elif tag == "thead":
            self._in_thead = True
        elif tag in ("tbody", "tfoot"):
            self._in_thead = False
        elif tag == "caption":
            self._in_caption = True
        elif tag == "tr":
            self._start_row()
        elif tag in ("td", "th"):
            self._start_cell(tag == "th", dict(attrs))
        else:
            self._add_separator(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            if self._depth == 1:
                self._end_row()
            self._depth = max(0, self._depth - 1)
            self._add_text(" ")
        elif self._depth != 1:
            self._add_separator(tag)
        elif tag == "thead":
            self._in_thead = False
        elif tag == "caption":
            self._in_caption = False
        elif tag == "tr":
            self._end_row()
        elif tag in ("td", "th"):
            self._cell = None
        else:
            self._add_separator(tag)

    def handle_data(self, data: str) -> None:
        self._add_text(data)

    def _add_separator(self, tag: str) -> None:
        self._add_text(_tag_separator(tag))

    def _add_text(self, text: str) -> None:
        if self._cell is not None and not self._in_caption:
            self._cell.parts.append(text)

    def _start_row(self) -> _HtmlRow:
        self._end_row()
        self._row = _HtmlRow(self._in_thead)
        self.rows.append(self._row)
        return self._row

    def _end_row(self) -> None:
        self._cell = None
        self._row = None

    def _start_cell(self, is_header: bool, attrs: dict[str, str | None]) -> None:
        row = self._row or self._start_row()
        self._cell = _HtmlCell(
            is_header=is_header,
            colspan=_span(attrs.get("colspan"), MAX_COLSPAN),
            rowspan=_span(attrs.get("rowspan"), MAX_ROWSPAN),
        )
        row.cells.append(self._cell)


def _tag_separator(tag: str) -> str:
    return "" if tag in _INLINE_HTML_TAGS else " "


def _span(value: str | None, limit: int) -> int:
    if value is None or not _DIGITS.fullmatch(value.strip()):
        return 1
    return min(max(int(value), 1), limit)


def _keep_entities_raw(source: str) -> str:
    return source.replace("&", "&amp;")


def _parse_html_table(text: str, start: int, end: int) -> MarkdownTable:
    parser = _HtmlTableParser()
    source = text[start:end]
    marker = "\ue000"
    while marker in source:
        marker += "\ue000"
    literals: list[str] = []

    def protect(match: re.Match[str]) -> str:
        literals.append(match[0])
        return f"{marker}{len(literals) - 1}\ue001"

    protected = _CODE_SPAN.sub(protect, source)
    parser.feed(_keep_entities_raw(protected))
    parser.close()
    rows = parser.rows
    for row in rows:
        for cell in row.cells:
            cell.parts = [
                re.sub(re.escape(marker) + r"(\d+)\ue001", lambda m: literals[int(m[1])], part)
                for part in cell.parts
            ]
    header_flags = _html_header_flags(rows)
    grid, anchors = _expand_spans(rows)
    flagged = list(zip(grid, header_flags, strict=True))
    order = [i for i, (row, flag) in enumerate(flagged) if row and flag]
    order += [i for i, (row, flag) in enumerate(flagged) if row and not flag]
    return MarkdownTable(
        index=_UNNUMBERED,
        start=start,
        end=end,
        header=tuple(grid[i] for i in order if header_flags[i]),
        body=tuple(grid[i] for i in order if not header_flags[i]),
        source_cells=tuple(cell.text for row in rows for cell in row.cells),
        merged=_merged_cells(anchors, order),
    )


def _merged_cells(anchors: Sequence[CellSpan], order: Sequence[int]) -> tuple[CellSpan, ...]:
    position = {source: final for final, source in enumerate(order)}
    merged: list[CellSpan] = []
    for span in anchors:
        if span.row not in position:
            continue
        top = position[span.row]
        there = 1
        while there < span.rows and position.get(span.row + there) == top + there:
            there += 1
        if there * span.columns > 1:
            merged.append(span.model_copy(update={"row": top, "rows": there}))
    return tuple(sorted(merged, key=lambda span: (span.row, span.column)))


def _html_header_flags(rows: Sequence[_HtmlRow]) -> list[bool]:
    if any(row.in_thead for row in rows):
        return [row.in_thead for row in rows]
    flags: list[bool] = []
    leading = True
    for row in rows:
        leading = leading and row.is_all_header
        flags.append(leading)
    return flags


def _expand_spans(rows: Sequence[_HtmlRow]) -> tuple[list[Cells], list[CellSpan]]:
    carried: dict[int, tuple[str, int]] = {}
    grid: list[Cells] = []
    anchors: list[CellSpan] = []
    for index, row in enumerate(rows):
        slots = {column: text for column, (text, _) in carried.items()}
        carried = {c: (text, left - 1) for c, (text, left) in carried.items() if left > 1}
        column = 0
        for cell in row.cells:
            while column in slots:
                column += 1
            if cell.rowspan > 1 or cell.colspan > 1:
                anchors.append(
                    CellSpan(row=index, column=column, rows=cell.rowspan, columns=cell.colspan)
                )
            text = cell.text
            for covered in range(column, column + cell.colspan):
                slots[covered] = text
                if cell.rowspan > 1:
                    carried[covered] = (text, cell.rowspan - 1)
            column += cell.colspan
        grid.append(tuple(slots.get(c, "") for c in range(max(slots, default=-1) + 1)))
    return grid, anchors
