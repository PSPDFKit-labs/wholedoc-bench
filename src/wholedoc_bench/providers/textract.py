import html
import re
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from wholedoc_bench.providers.base import Provider, RawResult, clip_box, grid_place
from wholedoc_bench.providers.credentials import CredentialError
from wholedoc_bench.providers.markdown import escape_markdown
from wholedoc_bench.providers.pdf import pdf_pages, require_pdf_reader
from wholedoc_bench.schema import (
    Box,
    Channel,
    ElementClass,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
    TableCell,
)

FEATURES = ("LAYOUT", "TABLES")
MAX_ATTEMPTS = 5
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 120

_HEADINGS = {"LAYOUT_TITLE": "#", "LAYOUT_SECTION_HEADER": "##"}
_FURNITURE = {"LAYOUT_HEADER": ElementClass.HEADER, "LAYOUT_FOOTER": ElementClass.FOOTER}
_FURNITURE_TYPES = frozenset({*_FURNITURE, "LAYOUT_PAGE_NUMBER"})
_KEY_VALUE = "LAYOUT_KEY_VALUE"
_LAYOUT_TYPES = frozenset(
    {
        *_HEADINGS,
        *_FURNITURE,
        _KEY_VALUE,
        "LAYOUT_TEXT",
        "LAYOUT_LIST",
        "LAYOUT_TABLE",
        "LAYOUT_FIGURE",
        "LAYOUT_PAGE_NUMBER",
    }
)
_TABLE_NOTES = {
    "TABLE_TITLE": (ElementClass.CAPTION, RelationKind.CAPTION_OF),
    "TABLE_FOOTER": (ElementClass.FOOTNOTE, RelationKind.FOOTNOTE_OF),
}

_CELL_SYNTAX = re.compile(r"[\\`*_\[\]~|]")
_BULLET = re.compile(r"^[•◦▪▫●○■□‣⁃∙·–—*-]\s+")
_ORDERED = re.compile(r"^(\d{1,3}[.)])\s+(.*)$", re.DOTALL)


class TextractProvider(Provider):
    name: ClassVar[str] = "textract"
    sends_documents: ClassVar[bool] = True
    formats: ClassVar[frozenset[str]] = frozenset(
        {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    meters: ClassVar[frozenset[str]] = frozenset({"pages"})

    def __init__(self, config: dict[str, Any] | None = None, *, client: Any = None) -> None:
        super().__init__(config)
        self._client = client

    def prepare(self) -> None:
        require_pdf_reader(self.name)
        self._textract()

    def fetch(self, document_path: Path) -> RawResult:
        client = self._textract()
        pages: list[dict[str, Any]] = []
        for page in _pages(document_path):
            try:
                response = client.analyze_document(
                    Document={"Bytes": page}, FeatureTypes=list(FEATURES)
                )
            except Exception as e:
                raise RuntimeError(_failure(e)) from None
            pages.append({k: v for k, v in response.items() if k != "ResponseMetadata"})
        models = sorted({str(p.get("AnalyzeDocumentModelVersion", "unknown")) for p in pages})
        return RawResult(
            provider=self.name,
            document_id=document_path.parent.name,
            payload={"pages": pages},
            metadata={"version": _version(models), "config": dict(self.config)},
        )

    def _textract(self) -> Any:
        if self._client is None:
            self._client = _client()
        return self._client

    def normalize(self, raw: RawResult) -> Prediction:
        out = _Output()
        for index, response in enumerate(raw.payload["pages"]):
            out.read_page(index, _Blocks(response["Blocks"]))
        return out.prediction()

    def usage(self, raw: RawResult) -> dict[str, float]:
        return {"pages": sum(r["DocumentMetadata"]["Pages"] for r in raw.payload["pages"])}


def _version(models: list[str]) -> str:
    return f"textract AnalyzeDocument {'+'.join(FEATURES)}, model {', '.join(models) or 'unknown'}"


def _client() -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ValueError("textract needs boto3: run `uv sync --extra textract`") from None
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is not None:
            credentials.get_frozen_credentials()
    except Exception as e:
        raise CredentialError(
            f"AWS credentials could not be loaded ({type(e).__name__}); check the AWS "
            "environment variables or profile. Do not put credentials in --config."
        ) from None
    if credentials is None:
        raise CredentialError(
            "No AWS credentials found; set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "(see .env.example), AWS_PROFILE, or run with an IAM role. "
            "Do not put credentials in --config."
        )
    if not session.region_name:
        raise CredentialError("No AWS region; set AWS_DEFAULT_REGION or a profile region.")
    config = Config(
        retries={"mode": "standard", "total_max_attempts": MAX_ATTEMPTS},
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        read_timeout=READ_TIMEOUT_SECONDS,
    )
    try:
        check = Config(
            retries={"mode": "standard", "total_max_attempts": 2},
            connect_timeout=5,
            read_timeout=10,
        )
        session.client("sts", config=check).get_caller_identity()
    except Exception as e:
        raise CredentialError(
            f"AWS did not accept the credentials or could not be reached ({_reason(e)}); "
            "check that the keys are current. Do not put credentials in --config."
        ) from None
    return session.client("textract", config=config)


def _pages(document_path: Path) -> Iterator[bytes]:
    if document_path.suffix.lower() != ".pdf":
        yield document_path.read_bytes()
        return
    yield from pdf_pages(document_path)


def _reason(error: Exception) -> str:
    response = getattr(error, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return code if isinstance(code, str) and code.isidentifier() else type(error).__name__


def _failure(error: Exception) -> str:
    return (
        f"Textract AnalyzeDocument failed ({_reason(error)}); check the AWS credentials, the "
        "textract:AnalyzeDocument permission, the region and the document limits."
    )


class _Blocks:
    def __init__(self, blocks: list[dict[str, Any]]) -> None:
        self.all = blocks
        self.by_id = {b["Id"]: b for b in blocks}

    def related(self, block: dict[str, Any], kind: str = "CHILD") -> list[dict[str, Any]]:
        return [
            self.by_id[i]
            for r in block.get("Relationships") or []
            if r["Type"] == kind
            for i in r["Ids"]
            if i in self.by_id
        ]

    def words(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        if block["BlockType"] == "WORD":
            return [block]
        return [w for child in self.related(block) for w in self.words(child)]

    def word_ids(self, block: dict[str, Any]) -> set[str]:
        return {w["Id"] for w in self.words(block)}

    def lines(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        if block["BlockType"] == "LINE":
            return [block]
        return [line for child in self.related(block) for line in self.lines(child)]

    def text(self, block: dict[str, Any], skip: Collection[str] = frozenset()) -> str:
        if block["BlockType"] == "LINE":
            return self.line_text(block, skip)
        lines = [c for c in self.related(block) if c["BlockType"] == "LINE"]
        if lines:
            return _join_lines([self.line_text(line, skip) for line in lines])
        return " ".join(
            t for w in self.words(block) if w["Id"] not in skip and (t := w.get("Text"))
        )

    def line_text(self, line: dict[str, Any], skip: Collection[str] = frozenset()) -> str:
        words = self.words(line)
        if not any(w["Id"] in skip for w in words):
            return line.get("Text", "")
        return " ".join(t for w in words if w["Id"] not in skip and (t := w.get("Text")))


def _join_lines(lines: list[str]) -> str:
    text = ""
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if text and not (text.endswith("-") and text[-2:-1].isalpha()):
            text += " "
        text += line
    return text


class _Output:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.layout: list[LayoutItem] = []
        self.relations: list[Relation] = []
        self.unscored: list[str] = []
        self.page_starts: list[int | None] = []
        self.table_words: set[str] = set()

    def prediction(self) -> Prediction:
        markdown = "".join(self.parts) + ("\n" if self.parts else "")
        offsets: list[int] = []
        following = len(markdown)
        for start in reversed(self.page_starts):
            following = start if start is not None else following
            offsets.append(following)
        return Prediction(
            markdown=markdown,
            layout=self.layout,
            relations=self.relations,
            page_offsets=offsets[::-1],
            unscored_text=self.unscored,
        )

    def write(self, text: str, *, separator: str = "\n\n") -> tuple[int, int]:
        if self.parts:
            self.parts.append(separator)
            self.length += len(separator)
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        if self.page_starts[-1] is None:
            self.page_starts[-1] = start
        return start, self.length

    def in_tables(self, blocks: _Blocks, line: dict[str, Any]) -> bool:
        return bool(words := blocks.word_ids(line)) and words <= self.table_words

    def new_text(self, blocks: _Blocks, block: dict[str, Any]) -> str:
        return blocks.text(block, self.table_words)

    def write_text(
        self, text: str, prefix: str = "", separator: str = "\n\n"
    ) -> tuple[int, int] | None:
        return self.write(prefix + escape_markdown(text), separator=separator) if text else None

    def write_lines(self, blocks: _Blocks, lines: list[dict[str, Any]]) -> tuple[int, int] | None:
        skip = self.table_words
        return self.write_text(_join_lines([blocks.line_text(line, skip) for line in lines]))

    def text_item(
        self,
        blocks: _Blocks,
        block: dict[str, Any],
        cls: ElementClass,
        page: int,
        span: tuple[int, int] | None,
    ) -> bool:
        lines = blocks.lines(block)
        own = [line for line in lines if not self.in_tables(blocks, line)]
        if lines and not own:
            return False
        source = block if len(own) == len(lines) else _region(block["Id"], own)
        self.item(source, cls, page, self.new_text(blocks, block), span)
        return True

    def item(
        self,
        block: dict[str, Any],
        cls: ElementClass,
        page: int,
        text: str,
        span: tuple[int, int] | None = None,
        *,
        channel: Channel = Channel.BODY,
        cells: list[TableCell] | None = None,
    ) -> None:
        self.layout.append(
            LayoutItem(
                id=block["Id"],
                cls=cls,
                page=page,
                text=text,
                box=_box(block),
                channel=channel,
                md_start=span[0] if span else None,
                md_end=span[1] if span else None,
                cells=cells,
            )
        )

    def read_page(self, page: int, blocks: _Blocks) -> None:
        self.page_starts.append(None)
        layout = [b for b in blocks.all if b["BlockType"].startswith("LAYOUT_")]
        nested = {c["Id"] for b in layout for c in blocks.related(b)}
        tables = _match_tables(blocks, [b for b in layout if b["BlockType"] == "LAYOUT_TABLE"])
        grids = [b for b in blocks.all if b["BlockType"] == "TABLE"]
        matched = {t["Id"] for t in tables.values()}
        loose = [t for t in grids if t["Id"] not in matched]
        anchors = _anchor_tables(blocks, [b for b in layout if b["Id"] not in nested], loose)
        self.table_words = {w for table in grids for w in blocks.word_ids(table)}
        owned = [
            *((layout_id, t) for layout_id, t in tables.items()),
            *((t["Id"], t) for held in anchors.values() for t in held),
        ]
        notes = _table_notes(blocks, owned)
        covered: set[str] = set()
        for block in layout:
            covered |= blocks.word_ids(block)
            if block["Id"] in nested:
                continue
            kind = block["BlockType"]
            if kind not in _LAYOUT_TYPES:
                raise ValueError(f"block {block['Id']}: no class mapping for Textract {kind!r}")
            held = anchors.get(block["Id"], [])
            if kind == "LAYOUT_TABLE":
                self.read_table(page, blocks, block, tables.get(block["Id"]), held)
                continue
            if held and kind in ("LAYOUT_TEXT", _KEY_VALUE):
                self.read_text_around(page, blocks, block, held, notes)
                continue
            first = self.grids_first(blocks, block, held)
            self.write_grids(page, blocks, held if first else [])
            self.read_block(page, blocks, block, notes)
            self.write_grids(page, blocks, [] if first else held)
        self.write_grids(page, blocks, anchors.get(None, []))
        for line in blocks.all:
            if line["BlockType"] == "LINE" and not blocks.word_ids(line) <= covered:
                if orphan := line.get("Text", "").strip():
                    self.unscored.append(orphan)

    def read_block(
        self,
        page: int,
        blocks: _Blocks,
        block: dict[str, Any],
        notes: dict[str, tuple[ElementClass, RelationKind, str]],
    ) -> None:
        kind = block["BlockType"]
        text = blocks.text(block)
        if kind == _KEY_VALUE and not text.strip():
            return
        if kind in _HEADINGS:
            span = self.write_text(self.new_text(blocks, block), f"{_HEADINGS[kind]} ")
            self.text_item(blocks, block, ElementClass.HEADING, page, span)
        elif kind in ("LAYOUT_TEXT", _KEY_VALUE):
            cls, relation, table = notes.get(block["Id"], (ElementClass.PARAGRAPH, None, None))
            span = self.write_text(self.new_text(blocks, block))
            if self.text_item(blocks, block, cls, page, span) and relation and table:
                self.relations.append(Relation(kind=relation, source=block["Id"], target=table))
        elif kind == "LAYOUT_LIST":
            self.read_list(page, blocks, block)
        elif kind == "LAYOUT_FIGURE":
            self.item(block, ElementClass.FIGURE, page, text)
        else:
            cls = _FURNITURE.get(kind) or _page_number_class(block)
            self.item(block, cls, page, text, channel=Channel.FURNITURE)

    def read_text_around(
        self,
        page: int,
        blocks: _Blocks,
        block: dict[str, Any],
        grids: list[dict[str, Any]],
        notes: dict[str, tuple[ElementClass, RelationKind, str]],
    ) -> None:
        spans = self.write_around(page, blocks, blocks.lines(block), [(t, t) for t in grids])
        cls, relation, table = notes.get(block["Id"], (ElementClass.PARAGRAPH, None, None))
        span = spans[0] if spans else None
        if self.text_item(blocks, block, cls, page, span) and relation and table:
            self.relations.append(Relation(kind=relation, source=block["Id"], target=table))

    def grids_first(
        self, blocks: _Blocks, block: dict[str, Any], grids: list[dict[str, Any]]
    ) -> bool:
        if not grids:
            return True
        rest = [line for line in blocks.lines(block) if not self.in_tables(blocks, line)]
        return not (rest and _is_above(rest[0], _top(grids[0])))

    def write_around(
        self,
        page: int,
        blocks: _Blocks,
        lines: list[dict[str, Any]],
        grids: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[tuple[int, int]]:
        held = {w for line in lines for w in blocks.word_ids(line)}
        pieces: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for table, owner in grids:
            cells = blocks.word_ids(table)
            found = {
                kind: sorted(
                    (
                        n
                        for n in blocks.related(table, kind)
                        if (words := blocks.word_ids(n)) <= held and not words & cells
                    ),
                    key=_order,
                )
                for kind in _TABLE_NOTES
            }
            group = sorted([*found["TABLE_TITLE"], table, *found["TABLE_FOOTER"]], key=_order)
            pieces += [(piece, owner) for piece in group]
        self.table_words |= {
            w
            for piece, _ in pieces
            if piece["BlockType"] != "TABLE"
            for w in blocks.word_ids(piece)
        }
        spans = []
        rest = list(lines)
        for piece, owner in pieces:
            top = _top(piece)
            above = [line for line in rest if _is_above(line, top)]
            rest = [line for line in rest if line not in above]
            if span := self.write_lines(blocks, above):
                spans.append(span)
            if piece["BlockType"] == "TABLE":
                self.write_grid(page, blocks, piece, owner)
            else:
                self.line_note(page, blocks, piece, owner)
        if span := self.write_lines(blocks, rest):
            spans.append(span)
        return spans

    def line_note(
        self, page: int, blocks: _Blocks, note: dict[str, Any], owner: dict[str, Any]
    ) -> None:
        if text := blocks.text(note):
            cls, relation = _TABLE_NOTES[note["BlockType"]]
            self.item(note, cls, page, text, self.write(escape_markdown(text)))
            self.relations.append(Relation(kind=relation, source=note["Id"], target=owner["Id"]))

    def write_grids(self, page: int, blocks: _Blocks, grids: list[dict[str, Any]]) -> None:
        for table in grids:
            self.write_grid(page, blocks, table, table)

    def read_list(self, page: int, blocks: _Blocks, block: dict[str, Any]) -> None:
        separator = "\n\n"
        for child in blocks.related(block):
            marker, body = "-", _BULLET.sub("", self.new_text(blocks, child))
            if ordered := _ORDERED.match(body):
                marker, body = ordered.groups()
            span = self.write_text(body, f"{marker} ", separator)
            self.text_item(blocks, child, ElementClass.LIST_ITEM, page, span)
            separator = "\n" if span else separator

    def read_table(
        self,
        page: int,
        blocks: _Blocks,
        block: dict[str, Any],
        table: dict[str, Any] | None,
        extra: list[dict[str, Any]],
    ) -> None:
        lines = [c for c in blocks.related(block) if c["BlockType"] == "LINE"]
        grids = [*([(table, block)] if table else []), *((t, t) for t in extra)]
        spans = self.write_around(page, blocks, lines, sorted(grids, key=lambda g: _order(g[0])))
        if table is None:
            self.text_item(blocks, block, ElementClass.TABLE, page, spans[0] if spans else None)

    def write_grid(
        self, page: int, blocks: _Blocks, table: dict[str, Any], owner: dict[str, Any]
    ) -> None:
        grid = _Grid(blocks, table)
        self.table_note(page, owner, grid, grid.title_rows, "title")
        entries = [
            (e, _place(index, e.block))
            for index, row in enumerate(grid.rows)
            for e in grid.entries(row)
            if e.block is not None
        ]
        self.item(
            owner,
            ElementClass.TABLE,
            page,
            " ".join(e.text for e, _ in entries if e.text),
            self.write(grid.html()),
            cells=[TableCell(text=e.text, box=e.box, **place) for e, place in entries],
        )
        self.table_note(page, owner, grid, grid.footer_rows, "footer")

    def table_note(
        self, page: int, block: dict[str, Any], grid: "_Grid", rows: list[int], kind: str
    ) -> None:
        if not (text := grid.text(rows)):
            return
        cls, relation = _TABLE_NOTES[f"TABLE_{kind.upper()}"]
        note = _region(f"{block['Id']}-{kind}", grid.cells(rows))
        self.item(note, cls, page, text, self.write(escape_markdown(text)))
        self.relations.append(Relation(kind=relation, source=note["Id"], target=block["Id"]))


def _match_tables(
    blocks: _Blocks, layout_tables: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    tables = [b for b in blocks.all if b["BlockType"] == "TABLE"]
    matched: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for layout in layout_tables:
        named = [c for c in blocks.related(layout) if c["BlockType"] == "TABLE"]
        words = blocks.word_ids(layout)
        candidates = [t for t in named or tables if t["Id"] not in used]
        overlaps = [(len(words & blocks.word_ids(t)), t) for t in candidates]
        best = max(overlaps, key=lambda o: o[0], default=(0, None))
        if best[1] is not None and (named or best[0] > 0):
            matched[layout["Id"]] = best[1]
            used.add(best[1]["Id"])
    return matched


def _anchor_tables(
    blocks: _Blocks, layout: list[dict[str, Any]], tables: list[dict[str, Any]]
) -> dict[str | None, list[dict[str, Any]]]:
    anchors: dict[str | None, list[dict[str, Any]]] = {}
    for table in tables:
        if not (words := blocks.word_ids(table)):
            continue
        holders = [(len(words & blocks.word_ids(b)), b) for b in layout]
        holders = [(shared, b) for shared, b in holders if shared]
        body = [(shared, b) for shared, b in holders if b["BlockType"] not in _FURNITURE_TYPES]
        if holders and not body:
            continue
        anchor = max(body, key=lambda h: h[0])[1]["Id"] if body else None
        anchors.setdefault(anchor, []).append(table)
    for held in anchors.values():
        held.sort(key=_order)
    return anchors


def _table_notes(
    blocks: _Blocks, tables: list[tuple[str, dict[str, Any]]]
) -> dict[str, tuple[ElementClass, RelationKind, str]]:
    notes: dict[str, tuple[ElementClass, RelationKind, str]] = {}
    texts = [b for b in blocks.all if b["BlockType"] == "LAYOUT_TEXT"]
    for item_id, table in tables:
        cells = blocks.word_ids(table)
        for kind, (cls, relation) in _TABLE_NOTES.items():
            for note in blocks.related(table, kind):
                words = blocks.word_ids(note)
                for text in texts:
                    if (ids := blocks.word_ids(text)) and ids <= words and not ids <= cells:
                        notes.setdefault(text["Id"], (cls, relation, item_id))
    return notes


class _Entry(NamedTuple):
    block: dict[str, Any] | None
    text: str
    header: bool
    box: Box | None


class _Grid:
    def __init__(self, blocks: _Blocks, table: dict[str, Any]) -> None:
        self.blocks = blocks
        cells = [c for c in blocks.related(table) if c["BlockType"] == "CELL"]
        self.at = {(c["RowIndex"], c["ColumnIndex"]): c for c in cells}
        self.columns = max((column for _, column in self.at), default=0)
        self.merged = {
            (m["RowIndex"], m["ColumnIndex"]): m
            for kind in ("MERGED_CELL", "CHILD")
            for m in blocks.related(table, kind)
            if m["BlockType"] == "MERGED_CELL"
        }
        self.covered: set[tuple[int, int]] = set()
        for (row, column), m in self.merged.items():
            self.covered |= {
                (row + r, column + c)
                for r in range(m.get("RowSpan", 1))
                for c in range(m.get("ColumnSpan", 1))
            } - {(row, column)}
        rows = sorted({row for row, _ in self.at})
        self.title_rows: list[int] = []
        self.footer_rows: list[int] = []
        while rows and self.marked(rows[0], "TABLE_TITLE"):
            self.title_rows.append(rows.pop(0))
        while rows and self.marked(rows[-1], "TABLE_FOOTER"):
            self.footer_rows.insert(0, rows.pop())
        while self.title_rows and rows and self.joined(self.title_rows[-1], rows[0]):
            rows.insert(0, self.title_rows.pop())
        while self.footer_rows and rows and self.joined(rows[-1], self.footer_rows[0]):
            rows.append(self.footer_rows.pop(0))
        if not rows:
            rows, self.title_rows, self.footer_rows = self.title_rows + self.footer_rows, [], []
        self.rows = rows

    def cells(self, rows: list[int]) -> list[dict[str, Any]]:
        return [
            self.at[row, column]
            for row in rows
            for column in range(1, self.columns + 1)
            if (row, column) in self.at
        ]

    def joined(self, upper: int, lower: int) -> bool:
        return any(
            row <= upper and lower < row + m.get("RowSpan", 1)
            for (row, _), m in self.merged.items()
        )

    def marked(self, row: int, entity: str) -> bool:
        kinds = [c.get("EntityTypes") or [] for c in self.cells([row])]
        return bool(kinds) and all(entity in k for k in kinds)

    def text(self, rows: list[int]) -> str:
        return " ".join(t for c in self.cells(rows) if (t := self.blocks.text(c)))

    def html(self) -> str:
        head = 0
        while head < len(self.rows) and any(_is_header(c) for c in self.cells([self.rows[head]])):
            head += 1
        rows = [self.row(row) for row in self.rows]
        thead = ["<thead>", *rows[:head], "</thead>"] if head else []
        return "\n".join(["<table>", *thead, *rows[head:], "</table>"])

    def entries(self, row: int) -> list[_Entry]:
        out = []
        for column in range(1, self.columns + 1):
            if (row, column) in self.covered:
                continue
            if (m := self.merged.get((row, column))) is not None:
                parts = [p for p in self.blocks.related(m) if p["BlockType"] == "CELL"]
                text = " ".join(t for p in parts if (t := self.blocks.text(p)))
                box = _box(m if _bounds(m) else _region(m["Id"], parts))
                out.append(_Entry(m, text, any(_is_header(p) for p in parts), box))
            elif (cell := self.at.get((row, column))) is not None:
                out.append(_Entry(cell, self.blocks.text(cell), _is_header(cell), _box(cell)))
            else:
                out.append(_Entry(None, "", False, None))
        return out

    def row(self, row: int) -> str:
        out = []
        for entry in self.entries(row):
            attributes = ""
            if entry.block is not None and entry.block["BlockType"] == "MERGED_CELL":
                for name, key in (("rowspan", "RowSpan"), ("colspan", "ColumnSpan")):
                    if (span := entry.block.get(key, 1)) > 1:
                        attributes += f' {name}="{span}"'
            tag = "th" if entry.header else "td"
            out.append(f"<{tag}{attributes}>{_escape_cell(entry.text)}</{tag}>")
        return f"<tr>{''.join(out)}</tr>"


def _place(row: int, block: dict[str, Any]) -> dict[str, int]:
    column = block.get("ColumnIndex")
    return grid_place(
        row,
        column - 1 if type(column) is int else None,
        block.get("RowSpan", 1),
        block.get("ColumnSpan", 1),
    )


def _is_header(cell: dict[str, Any]) -> bool:
    return "COLUMN_HEADER" in (cell.get("EntityTypes") or [])


def _escape_cell(text: str) -> str:
    return _CELL_SYNTAX.sub(lambda m: f"&#{ord(m[0])};", html.escape(text, quote=False))


def _page_number_class(block: dict[str, Any]) -> ElementClass:
    bounds = _bounds(block)
    middle = bounds["Top"] + bounds["Height"] / 2 if bounds else 1.0
    return ElementClass.HEADER if middle < 0.5 else ElementClass.FOOTER


def _bounds(block: dict[str, Any]) -> dict[str, float] | None:
    return (block.get("Geometry") or {}).get("BoundingBox")


def _top(block: dict[str, Any]) -> float | None:
    bounds = _bounds(block)
    return bounds["Top"] if bounds else None


def _order(block: dict[str, Any]) -> tuple[bool, float]:
    top = _top(block)
    return (top is None, 0.0 if top is None else top)


def _is_above(block: dict[str, Any], top: float | None) -> bool:
    return top is not None and (own := _top(block)) is not None and own < top


def _region(id: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    boxes = [b for block in blocks if (b := _bounds(block))]
    if not boxes:
        return {"Id": id}
    left, top = min(b["Left"] for b in boxes), min(b["Top"] for b in boxes)
    right = max(b["Left"] + b["Width"] for b in boxes)
    bottom = max(b["Top"] + b["Height"] for b in boxes)
    box = {"Left": left, "Top": top, "Width": right - left, "Height": bottom - top}
    return {"Id": id, "Geometry": {"BoundingBox": box}}


def _box(block: dict[str, Any]) -> Box | None:
    bounds = _bounds(block)
    if bounds is None:
        return None
    return clip_box(bounds["Left"], bounds["Top"], bounds["Width"], bounds["Height"], 1.0, 1.0)
