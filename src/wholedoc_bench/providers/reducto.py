import hashlib
import json
import math
import mimetypes
import os
import re
import time
from collections.abc import Callable
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from wholedoc_bench.providers.base import Provider, RawResult, clip_box, grid_place
from wholedoc_bench.providers.credentials import CredentialError
from wholedoc_bench.schema import (
    Channel,
    Chart,
    ChartKind,
    ChartPoint,
    ChartSeries,
    ElementClass,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
    TableCell,
    ValueAxis,
)
from wholedoc_bench.scorer.utils.blocks import markdown_text

API_URL = "https://platform.reducto.ai"
TIMEOUT_SECONDS = 900.0
CONFIG = {
    "retrieval": {"chunking": {"chunk_mode": "disabled"}},
    "formatting": {"table_output_format": "html", "merge_tables": True},
    "settings": {"persist_results": False, "model": "r-1"},
    "enhance": {"agentic": [{"scope": "figure", "advanced_chart_agent": True}]},
}
MODES = ("r-1", "legacy", "legacy-agentic")
_CLASSES = {
    "Title": ElementClass.HEADING,
    "Section Header": ElementClass.HEADING,
    "Header": ElementClass.HEADER,
    "Footer": ElementClass.FOOTER,
    "Page Number": ElementClass.FOOTER,
    "Text": ElementClass.PARAGRAPH,
    "Key Value": ElementClass.PARAGRAPH,
    "List Item": ElementClass.LIST_ITEM,
    "Table": ElementClass.TABLE,
    "Figure": ElementClass.FIGURE,
}


class ReductoProvider(Provider):
    name = "reducto"
    formats = frozenset(
        {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".webp",
            ".gif",
            ".docx",
            ".pptx",
            ".xlsx",
        }
    )
    meters = frozenset({"pages", "charts"})

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config)
        self._transport, self._sleep, self._clock = transport, sleep, clock
        self._cancelled = lambda: False

    def set_cancelled(self, callback):
        self._cancelled = callback

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("provider config must be a JSON object")
        if unknown := config.keys() - {"mode"}:
            raise ValueError(f"{cls.name}: unsupported config keys: {', '.join(sorted(unknown))}")
        mode = config.get("mode", "r-1")
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError(f"{cls.name}: mode must be one of {', '.join(MODES)}")
        return {"mode": mode}

    def _request_config(self) -> dict[str, Any]:
        config = deepcopy(CONFIG)
        if self.config["mode"] != "r-1":
            config["settings"]["model"] = "legacy"
            del config["enhance"]
        if self.config["mode"] == "legacy-agentic":
            config["enhance"] = {
                "agentic": [{"scope": scope} for scope in ("text", "table", "figure")],
                "summarize_figures": True,
            }
        return config

    def prepare(self) -> None:
        if error := self.configuration_error():
            raise CredentialError(error)

    def raw_cache_key(self) -> str:
        digest = hashlib.sha256(
            json.dumps(self._request_config(), sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"reducto REST {self.config['mode']} {digest}"

    def version(self) -> str:
        return f"{self.raw_cache_key()} adapter 4"

    def configuration_error(self) -> str | None:
        key = os.environ.get("REDUCTO_API_KEY", "").strip()
        return "REDUCTO_API_KEY is not configured" if not key or key.startswith("op://") else None

    def _check(self, deadline):
        if self._cancelled():
            raise RuntimeError("Reducto processing cancelled locally")
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise RuntimeError("Reducto parsing timed out after 900 seconds")
        return remaining

    def _request(self, client, method, path, deadline, **kwargs):
        remaining = self._check(deadline)
        try:
            response = client.request(method, path, timeout=min(60.0, remaining), **kwargs)
        except httpx.HTTPError:
            raise RuntimeError("Reducto request failed or timed out") from None
        if not response.is_success:
            raise RuntimeError(f"Reducto request failed (HTTP {response.status_code})")
        try:
            value = response.json()
        except ValueError:
            raise RuntimeError("Reducto returned invalid JSON") from None
        if not isinstance(value, dict):
            raise RuntimeError("Reducto returned an invalid response")
        return value

    def fetch(self, document_path: Path) -> RawResult:
        if error := self.configuration_error():
            raise RuntimeError(error)
        if document_path.suffix.lower() not in self.formats:
            raise RuntimeError("Reducto does not support this file format")
        deadline = self._clock() + TIMEOUT_SECONDS
        headers = {"Authorization": f"Bearer {os.environ['REDUCTO_API_KEY'].strip()}"}
        with httpx.Client(
            base_url=API_URL, headers=headers, transport=self._transport, follow_redirects=False
        ) as client:
            with document_path.open("rb") as stream:
                uploaded = self._request(
                    client,
                    "POST",
                    "/upload",
                    deadline,
                    files={
                        "file": (
                            document_path.name,
                            stream,
                            mimetypes.guess_type(document_path.name)[0]
                            or "application/octet-stream",
                        )
                    },
                )
            file_id = uploaded.get("file_id")
            if not isinstance(file_id, str) or not file_id.startswith("reducto://"):
                raise RuntimeError("Reducto returned an invalid upload identifier")
            submitted = self._request(
                client,
                "POST",
                "/parse_async",
                deadline,
                json={"input": file_id, **self._request_config()},
            )
            job_id = submitted.get("job_id")
            if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
                raise RuntimeError("Reducto returned an invalid job identifier")
            try:
                while True:
                    job = self._request(client, "GET", f"/job/{job_id}", deadline)
                    status = job.get("status")
                    if status == "Completed":
                        break
                    if status not in {"Idle", "Pending", "InProgress", "Completing"}:
                        raise RuntimeError("Reducto parsing failed; check the job in Reducto")
                    self._sleep(min(2.0, self._check(deadline)))
                payload = job.get("result")
                if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
                    raise RuntimeError("Reducto returned no usable parse output")
                result = payload["result"]
                with httpx.Client(transport=self._transport, follow_redirects=False) as download:
                    if result.get("type") == "url":
                        url = _signed_url(result.get("url"), "output")
                        payload["result"] = self._request(download, "GET", url, deadline)
                    if not isinstance(payload["result"].get("chunks"), list):
                        raise RuntimeError("Reducto returned no parsed chunks")
                    self._retain_chart_data(payload["result"], download, deadline)
            except RuntimeError:
                if self._cancelled() or self._clock() >= deadline:
                    try:
                        client.post(f"/cancel/{job_id}", timeout=10.0)
                    except httpx.HTTPError:
                        pass
                raise
        return RawResult(
            self.name,
            document_path.parent.name,
            payload,
            {"version": self.version(), "raw_cache_key": self.raw_cache_key()},
        )

    def _retain_chart_data(self, result, download, deadline):
        for chunk in result["chunks"]:
            for block in chunk.get("blocks") or []:
                if isinstance(block.get("extra"), dict):
                    block["extra"].pop("chart_reconstruction", None)
                links = block.get("chart_data")
                if not links:
                    continue
                block["chart_data"] = [
                    self._chart_download(download, link, deadline)
                    for link in (links if isinstance(links, list) else [links])
                ]

    def _chart_download(self, download, link, deadline):
        try:
            return self._request(download, "GET", _signed_url(link, "chart data"), deadline)
        except RuntimeError as error:
            self._check(deadline)
            return {"download_failed": str(error)}

    def normalize(self, raw: RawResult) -> Prediction:
        result = raw.payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("chunks"), list):
            raise ValueError("Reducto raw result has no parsed chunks")
        markdown, layout, relations, unscored = "", [], [], []
        for chunk in result["chunks"]:
            if markdown:
                markdown += "\n\n"
            start = len(markdown)
            content = chunk.get("content") or ""
            markdown += content
            cursor = 0
            for block in chunk.get("blocks") or []:
                text = block.get("content") or ""
                bounds = block.get("bbox") or {}
                page = bounds.get("original_page") or bounds.get("page")
                cls = _CLASSES.get(block.get("type"))
                if cls == ElementClass.FIGURE and _chart(block):
                    cls = ElementClass.CHART
                if cls is None or not isinstance(page, int) or page < 1:
                    if text:
                        unscored.append(markdown_text(text))
                    continue
                if block.get("type") == "Key Value" and not markdown_text(text).strip():
                    continue
                position = content.find(text, cursor) if text else -1
                if position >= 0:
                    cursor = position + len(text)
                pieces = _table_pieces(block) if cls == ElementClass.TABLE else []
                if pieces:
                    for index, (piece_page, piece_bounds, piece_text) in enumerate(pieces):
                        if index:
                            relations.append(
                                Relation(
                                    kind=RelationKind.CONTINUES_IN,
                                    source=layout[-1].id,
                                    target=f"reducto-{len(layout)}",
                                )
                            )
                        first = index == 0 and position >= 0
                        layout.append(
                            LayoutItem(
                                id=f"reducto-{len(layout)}",
                                cls=cls,
                                page=piece_page - 1,
                                text=markdown_text(piece_text),
                                box=_box(piece_bounds),
                                cells=_piece_cells(piece_text),
                                md_start=start + position if first else None,
                                md_end=start + position + len(text) if first else None,
                            )
                        )
                    continue
                layout.append(
                    LayoutItem(
                        id=f"reducto-{len(layout)}",
                        cls=cls,
                        page=page - 1,
                        text=markdown_text(text),
                        box=_box(bounds),
                        chart=_chart_json(block) if cls == ElementClass.CHART else None,
                        channel=Channel.FURNITURE
                        if cls in {ElementClass.HEADER, ElementClass.FOOTER}
                        else None,
                        md_start=start + position if position >= 0 else None,
                        md_end=start + position + len(text) if position >= 0 else None,
                    )
                )
        return Prediction(
            markdown=markdown, layout=layout, relations=relations, unscored_text=unscored
        )

    def usage(self, raw: RawResult) -> dict[str, float]:
        pages = raw.payload["usage"]["num_pages"]
        if not isinstance(pages, int):
            raise ValueError("Reducto's usage has no page count")
        blocks = (
            block
            for chunk in raw.payload["result"]["chunks"]
            for block in chunk.get("blocks") or []
        )
        charts = sum(bool(block.get("chart_data")) for block in blocks)
        return {"pages": pages, "charts": charts}


def _signed_url(url, what):
    parsed = urlsplit(url) if isinstance(url, str) else None
    if (
        not parsed
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise RuntimeError(f"Reducto returned an invalid {what} URL")
    return url


def _chart(block):
    extra = block.get("extra")
    return bool(block.get("chart_data")) or (
        isinstance(extra, dict) and extra.get("is_chart") is True
    )


def _chart_json(block):
    charts = [
        c
        for c in block.get("chart_data") or []
        if isinstance(c, dict) and "download_failed" not in c
    ]
    series, kinds = [], set()
    for chart in charts:
        for entry in chart.get("series") or []:
            if not isinstance(entry, dict):
                continue
            kinds.add(entry.get("type"))
            points = [
                ChartPoint(label=str(point["category"]), value=point["value"])
                for point in entry.get("data") or []
                if isinstance(point, dict)
                and point.get("category") is not None
                and isinstance(point.get("value"), (int, float))
                and not isinstance(point["value"], bool)
                and math.isfinite(point["value"])
            ]
            if points:
                series.append(ChartSeries(name=str(entry.get("label") or ""), points=points))
    if not charts:
        return None
    first = charts[0]
    title = first.get("chart_title")
    axis = (first.get("axes") or {}).get("y_primary") or {}
    try:
        value_axis = ValueAxis(min=axis["min"], max=axis["max"])
    except (KeyError, TypeError, ValueError):
        value_axis = None
    kinds &= set(ChartKind)
    chart = Chart(
        kind=kinds.pop() if len(kinds) == 1 else None,
        series=series,
        value_axis=value_axis if len(charts) == 1 else None,
        title=title.strip()
        if isinstance(title, str) and title.strip() and len(charts) == 1
        else None,
    )
    return chart if chart.series or chart.kind or chart.title else None


def _table_pieces(block):
    pieces = []
    for piece in block.get("merged_tables") or []:
        bounds = piece.get("bbox") if isinstance(piece, dict) else None
        if not isinstance(bounds, dict):
            return []
        page = bounds.get("original_page") or bounds.get("page")
        if not isinstance(page, int) or page < 1:
            return []
        pieces.append((page, bounds, piece.get("content") or ""))
    if len(pieces) < 2 or any(b[0] - a[0] != 1 for a, b in zip(pieces, pieces[1:], strict=False)):
        return []
    return pieces


class _HtmlCells(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.rows: list[list[tuple[list[str], int, int]]] = []
        self._depth = 0
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._depth += 1
        if self._depth == 1 and tag == "tr":
            self.rows.append([])
        elif self._depth == 1 and tag in ("td", "th") and self.rows:
            spans = dict(attrs)
            self._cell = []
            self.rows[-1].append(
                (self._cell, _span(spans.get("rowspan")), _span(spans.get("colspan")))
            )
        elif self._cell is not None:
            self._cell.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag: str) -> None:
        if self._depth == 1 and tag in ("td", "th", "tr"):
            self._cell = None
        elif self._cell is not None:
            self._cell.append(f"</{tag}>")
        if tag == "table":
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_entityref(self, name: str) -> None:
        self.handle_data(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.handle_data(f"&#{name};")


def _span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _piece_cells(html: str) -> list[TableCell] | None:
    reader = _HtmlCells()
    reader.feed(html)
    reader.close()
    taken: set[tuple[int, int]] = set()
    cells: list[TableCell] = []
    for row, written in enumerate(reader.rows):
        column = 0
        for inner, rows, columns in written:
            while (row, column) in taken:
                column += 1
            rows = min(rows, len(reader.rows) - row)
            taken.update((row + r, column + c) for r in range(rows) for c in range(columns))
            cells.append(
                TableCell(
                    text=markdown_text("".join(inner)),
                    box=None,
                    **grid_place(row, column, rows, columns),
                )
            )
            column += columns
    return cells or None


def _box(bounds):
    try:
        left, top, width, height = (float(bounds[k]) for k in ("left", "top", "width", "height"))
        return clip_box(left, top, width, height, 1, 1)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
