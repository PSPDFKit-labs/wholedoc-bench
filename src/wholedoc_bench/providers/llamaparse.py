import math
import re
import time
from collections.abc import Callable, Iterator
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

import httpx

from wholedoc_bench.providers.base import Provider, RawResult, clip_box
from wholedoc_bench.providers.credentials import require_env
from wholedoc_bench.schema import (
    Box,
    Channel,
    ElementClass,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
)
from wholedoc_bench.scorer.utils.blocks import markdown_text

API_URL = "https://api.cloud.llamaindex.ai"
API_KEY_VARIABLE = "LLAMA_CLOUD_API_KEY"
ENGINE_VERSION = "2026-09-28"
FAST_VERSION = "2026-06-15"
TIMEOUT_SECONDS = 600.0
POLL_SECONDS = 2.0
RETRIES = 3
TIERS = ("fast", "cost_effective", "agentic", "agentic_plus")
CHART_MODES = ("efficient", "agentic", "agentic_plus")
_TYPES = {
    "text": ElementClass.PARAGRAPH,
    "heading": ElementClass.HEADING,
    "table": ElementClass.TABLE,
    "image": ElementClass.FIGURE,
    "code": ElementClass.CODE,
    "link": ElementClass.PARAGRAPH,
    "header": ElementClass.HEADER,
    "footer": ElementClass.FOOTER,
}
_TABLE_TAG = re.compile(r"""<(/?)table\b(?:[^>"']|"[^"]*"|'[^']*')*>""", re.IGNORECASE)


class LlamaParseProvider(Provider):
    name: ClassVar[str] = "llamaparse"
    formats: ClassVar[frozenset[str]] = frozenset(
        {".pdf", ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    meters: ClassVar[frozenset[str]] = frozenset({"pages"})

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config)
        self._transport = transport
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("provider config must be a JSON object")
        if set(config) - {"tier", "version", "specialized_chart_parsing"}:
            raise ValueError(
                "llamaparse: unsupported config keys; "
                "allowed: tier, version, specialized_chart_parsing"
            )
        tier = config.get("tier", "agentic")
        if tier not in TIERS:
            raise ValueError(f"llamaparse tier must be one of {', '.join(TIERS)}")
        version = config.get("version", FAST_VERSION if tier == "fast" else ENGINE_VERSION)
        if version != "latest":
            try:
                if not isinstance(version, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", version):
                    raise ValueError
                date.fromisoformat(version)
            except (ValueError, TypeError):
                raise ValueError("llamaparse version must be latest or YYYY-MM-DD") from None
            if tier == "fast" and version != FAST_VERSION:
                raise ValueError(f"llamaparse fast requires version {FAST_VERSION} or latest")
        validated = {"tier": tier, "version": version}
        if "specialized_chart_parsing" in config:
            mode = config["specialized_chart_parsing"]
            if mode not in CHART_MODES:
                raise ValueError(
                    "llamaparse specialized_chart_parsing must be one of " + ", ".join(CHART_MODES)
                )
            if tier == "fast":
                raise ValueError("llamaparse fast does not support specialized_chart_parsing")
            validated["specialized_chart_parsing"] = mode
        return validated

    def prepare(self) -> None:
        require_env(API_KEY_VARIABLE)

    def fetch(self, document_path: Path) -> RawResult:
        key = require_env(API_KEY_VARIABLE).strip()
        deadline = self._clock() + TIMEOUT_SECONDS
        options = dict(self.config)
        chart_mode = options.pop("specialized_chart_parsing", None)
        if chart_mode is not None:
            options["processing_options"] = {"specialized_chart_parsing": chart_mode}
        try:
            with httpx.Client(
                transport=self._transport,
                headers={"Authorization": f"Bearer {key}"},
                follow_redirects=False,
            ) as client:
                with document_path.open("rb") as file:
                    upload = self._request(
                        client,
                        "POST",
                        "/api/v1/beta/files",
                        deadline,
                        data={"purpose": "parse"},
                        files={"file": (document_path.name, file)},
                    )
                file_id = _identifier(upload)
                created = self._request(
                    client,
                    "POST",
                    "/api/v2/parse",
                    deadline,
                    json={
                        "file_id": file_id,
                        **options,
                        "disable_cache": True,
                        "output_options": {
                            "markdown": {
                                "tables": {
                                    "output_tables_as_markdown": False,
                                    "merge_continued_tables": True,
                                }
                            }
                        },
                        "processing_control": {
                            "job_failure_conditions": {
                                "allowed_page_failure_ratio": 0.01,
                            }
                        },
                    },
                )
                job_id = _identifier(created)
                while True:
                    payload = self._request(
                        client,
                        "GET",
                        f"/api/v2/parse/{job_id}",
                        deadline,
                        params=[("expand", "markdown"), ("expand", "items")],
                    )
                    status = (payload.get("job") or {}).get("status")
                    if status == "COMPLETED":
                        _pages(payload, "markdown", required=True)
                        _pages(payload, "items")
                        break
                    if status in ("FAILED", "CANCELLED"):
                        raise RuntimeError(f"LlamaParse job {status}; check the service dashboard.")
                    if status not in ("PENDING", "RUNNING"):
                        raise RuntimeError("LlamaParse returned an invalid job status.")
                    self._wait(POLL_SECONDS, deadline)
        except httpx.RequestError:
            raise RuntimeError("LlamaParse request failed; check network connectivity.") from None
        return RawResult(
            provider=self.name,
            document_id=document_path.parent.name,
            payload=payload,
            metadata={
                "version": (
                    f"llamaparse v2 {self.config['tier']} {self.config['version']} merged-tables"
                    + (f" charts={chart_mode}" if chart_mode is not None else "")
                ),
                "config": dict(self.config),
            },
        )

    def _wait(self, delay: float, deadline: float) -> None:
        if self._clock() + delay >= deadline:
            raise RuntimeError("LlamaParse job timed out.")
        self._sleep(delay)

    def _request(
        self, client: httpx.Client, method: str, path: str, deadline: float, **kwargs: Any
    ) -> dict[str, Any]:
        for attempt in range(RETRIES + 1):
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise RuntimeError("LlamaParse job timed out.")
            response = client.request(
                method, API_URL + path, timeout=min(60.0, remaining), **kwargs
            )
            if method == "GET" and response.status_code in (429, 500, 502, 503, 504):
                if attempt < RETRIES:
                    try:
                        delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                    except ValueError:
                        delay = 2 ** (attempt + 1)
                    if not math.isfinite(delay) or delay < 0:
                        delay = 2 ** (attempt + 1)
                    self._wait(min(delay, 60.0), deadline)
                    continue
            if not response.is_success:
                raise RuntimeError(
                    f"LlamaParse API HTTP {response.status_code}; "
                    "check credentials, account limits, and document requirements."
                )
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError("LlamaParse returned invalid JSON.") from None
            if not isinstance(payload, dict):
                raise RuntimeError("LlamaParse returned an invalid response.")
            return payload
        raise AssertionError("unreachable")

    def usage(self, raw: RawResult) -> dict[str, float]:
        return {"pages": len(_pages(raw.payload, "markdown", required=True))}

    def normalize(self, raw: RawResult) -> Prediction:
        pages = _pages(raw.payload, "markdown", required=True)
        item_pages = {p["page_number"]: p for p in _pages(raw.payload, "items")}
        markdown = ""
        offsets: list[int] = []
        layout: list[LayoutItem] = []
        unscored: list[str] = []
        blank_pages: list[int] = []
        relations: list[Relation] = []
        merged: list[tuple[int, list[int], list[LayoutItem]]] = []
        for page in pages:
            page_number = page["page_number"]
            source = page.get("markdown")
            if not isinstance(source, str):
                raise ValueError("LlamaParse page has no Markdown.")
            if source:
                if markdown:
                    markdown += "\n\n"
                for index in blank_pages:
                    offsets[index] = len(markdown)
                blank_pages.clear()
            else:
                blank_pages.append(len(offsets))
            offsets.append(len(markdown))
            markdown += source
            item_page = item_pages.get(page_number, {})
            cursor = 0
            page_layout_start = len(layout)
            coloured: list[tuple[int, str]] = []
            for item, inherited in _walk(item_page.get("items") or []):
                kind = inherited or _TYPES.get(item.get("type", ""))
                md = item.get("md") or ""
                text = item.get("value") or item.get("text") or markdown_text(md)
                if kind is None:
                    if text:
                        unscored.append(text)
                    continue
                furniture = kind in (ElementClass.HEADER, ElementClass.FOOTER)
                start = source.find(md, cursor) if md and not furniture else -1
                end = start + len(md) if start >= 0 else None
                if start < 0 and kind is ElementClass.TABLE and (html := item.get("html")):
                    start = source.find(html, cursor)
                    end = start + len(html) if start >= 0 else None
                    if start < 0 and (plain := _without_colour(html)) is not None:
                        coloured.append((len(layout), plain))
                if end is not None:
                    cursor = end
                layout.append(
                    LayoutItem(
                        id=f"p{page_number}-{len(layout)}",
                        cls=kind,
                        page=page_number - 1,
                        text=text,
                        box=_box(item.get("bbox"), item_page),
                        channel=Channel.FURNITURE if furniture else Channel.BODY,
                        md_start=offsets[-1] + start if start >= 0 else None,
                        md_end=offsets[-1] + end if end is not None else None,
                    )
                )
                if kind is ElementClass.TABLE:
                    _link_merged_piece(item, layout[-1], page_number, merged, relations)
            _place_coloured_tables(source, offsets[-1], layout, page_layout_start, coloured)
            for field, kind in (("header", ElementClass.HEADER), ("footer", ElementClass.FOOTER)):
                furniture_md = page.get(field)
                if (
                    isinstance(furniture_md, str)
                    and furniture_md.strip()
                    and not any(item.cls == kind for item in layout[page_layout_start:])
                ):
                    layout.append(
                        LayoutItem(
                            id=f"p{page_number}-{len(layout)}",
                            cls=kind,
                            page=page_number - 1,
                            text=markdown_text(furniture_md),
                            channel=Channel.FURNITURE,
                        )
                    )
        return Prediction(
            markdown=markdown,
            layout=layout,
            relations=relations,
            page_offsets=offsets,
            unscored_text=unscored,
        )


class _Uncoloured(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.removed = False
        self._open: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        colour = (tag == "font" and any(name == "color" for name, _ in attrs)) or (
            tag == "span" and any(n == "style" and "color" in (v or "").lower() for n, v in attrs)
        )
        if tag in ("span", "font"):
            self._open.append((tag, colour))
        if colour:
            self.removed = True
        else:
            self.parts.append(self.get_starttag_text() or "")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        opened = [i for i, (name, _) in enumerate(self._open) if name == tag]
        if opened and self._open.pop(opened[-1])[1]:
            return
        self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.parts.append(f"<!--{data}-->")


def _without_colour(html: str) -> str | None:
    reader = _Uncoloured()
    reader.feed(html)
    reader.close()
    return "".join(reader.parts) if reader.removed else None


def _whole_tables(source: str) -> list[tuple[int, int]]:
    tables: list[tuple[int, int]] = []
    depth, start = 0, 0
    for tag in _TABLE_TAG.finditer(source):
        if not tag[1]:
            if depth == 0:
                start = tag.start()
            depth += 1
        elif depth:
            depth -= 1
            if depth == 0:
                tables.append((start, tag.end()))
    return tables


def _place_coloured_tables(
    source: str, offset: int, layout: list[LayoutItem], first: int, coloured: list[tuple[int, str]]
) -> None:
    if not coloured:
        return
    tables = _whole_tables(source)
    for index, plain in coloured:
        taken = {(i.md_start, i.md_end) for i in layout[first:] if i.md_start is not None}
        before = max(
            (i.md_end for i in layout[first:index] if i.md_end is not None), default=offset
        )
        after = min(
            (i.md_start for i in layout[index + 1 :] if i.md_start is not None),
            default=offset + len(source),
        )
        found = [
            (offset + s, offset + e)
            for s, e in tables
            if source[s:e] == plain
            and (offset + s, offset + e) not in taken
            and before <= offset + s
            and offset + e <= after
        ]
        if len(found) == 1:
            start, end = found[0]
            layout[index] = layout[index].model_copy(update={"md_start": start, "md_end": end})


def _link_merged_piece(
    item: dict[str, Any],
    piece: LayoutItem,
    page_number: int,
    merged: list[tuple[int, list[int], list[LayoutItem]]],
    relations: list[Relation],
) -> None:
    pages = item.get("merged_from_pages")
    if isinstance(pages, list) and all(type(p) is int for p in pages) and len(pages) > 1:
        merged.append((page_number, pages, [piece]))
        return
    into = item.get("merged_into_page")
    if type(into) is not int:
        return
    holders = [m for m in merged if m[0] == into and page_number in m[1]]
    if len(holders) != 1:
        return
    last = holders[0][2]
    if last[-1].page != page_number - 2:
        return
    relations.append(Relation(kind=RelationKind.CONTINUES_IN, source=last[-1].id, target=piece.id))
    last.append(piece)


def _identifier(payload: dict[str, Any]) -> str:
    value = payload.get("id")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise RuntimeError("LlamaParse returned an invalid file or job identifier.")
    return value


def _pages(payload: dict[str, Any], field: str, *, required: bool = False) -> list[dict]:
    container = payload.get(field)
    pages = container.get("pages") if isinstance(container, dict) else None
    if pages is None and not required:
        return []
    if not isinstance(pages, list) or (required and not pages):
        raise ValueError(f"LlamaParse returned no {field} pages.")
    numbers = []
    for page in pages:
        if not isinstance(page, dict) or page.get("success") is False:
            raise ValueError("LlamaParse returned a failed or invalid page.")
        number = page.get("page_number")
        if type(number) is not int or number < 1:
            raise ValueError("LlamaParse returned an invalid page number.")
        numbers.append(number)
    if len(set(numbers)) != len(numbers):
        raise ValueError("LlamaParse returned duplicate page numbers.")
    pages = sorted(pages, key=lambda page: page["page_number"])
    if required and [p["page_number"] for p in pages] != list(range(1, len(pages) + 1)):
        raise ValueError("LlamaParse returned incomplete pages.")
    return pages


def _walk(
    items: list[dict], inherited: ElementClass | None = None
) -> Iterator[tuple[dict, ElementClass | None]]:
    for item in items:
        kind = item.get("type")
        if kind in ("list", "header", "footer") and item.get("items"):
            child_kind = inherited or (ElementClass.LIST_ITEM if kind == "list" else _TYPES[kind])
            yield from _walk(item["items"], child_kind)
        else:
            yield item, inherited


def _box(boxes: Any, page: dict) -> Box | None:
    if not isinstance(boxes, list) or not boxes:
        return None
    try:
        width, height = float(page["page_width"]), float(page["page_height"])
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            return None
        corners = []
        for box in boxes:
            x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
            angle = math.radians(float(box.get("r") or 0))
            if not all(math.isfinite(v) for v in (x, y, w, h, angle)) or w <= 0 or h <= 0:
                return None
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
                corners.append(
                    (
                        x + w / 2 + dx * math.cos(angle) - dy * math.sin(angle),
                        y + h / 2 + dx * math.sin(angle) + dy * math.cos(angle),
                    )
                )
        left, top = min(x for x, _ in corners), min(y for _, y in corners)
        right, bottom = max(x for x, _ in corners), max(y for _, y in corners)
        return clip_box(left, top, right - left, bottom - top, width, height)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
