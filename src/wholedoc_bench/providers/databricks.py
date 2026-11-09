import base64
import html
import json
import math
import os
import re
import struct
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

import httpx

from wholedoc_bench.providers.base import Provider, RawResult, clip_box
from wholedoc_bench.providers.charts import named_chart_kind
from wholedoc_bench.providers.credentials import CredentialError, require_env
from wholedoc_bench.providers.markdown import escape_markdown
from wholedoc_bench.schema import (
    Box,
    Channel,
    Chart,
    ChartPoint,
    ChartSeries,
    ElementClass,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
)
from wholedoc_bench.scorer.utils.blocks import markdown_text
from wholedoc_bench.scorer.utils.tracks.t6_charts.charts import cell_values

HOST_VARIABLE = "DATABRICKS_HOST"
TOKEN_VARIABLE = "DATABRICKS_TOKEN"
CLIENT_ID_VARIABLE = "DATABRICKS_CLIENT_ID"
CLIENT_SECRET_VARIABLE = "DATABRICKS_CLIENT_SECRET"
VOLUME_VARIABLE = "DATABRICKS_VOLUME_PATH"
PARSE_PATH = "/api/2.0/ai-functions/ai-parse-document"
OPTIONS = {"version": "2.0", "description_element_types": "*"}
VERSION = "databricks ai_parse_document REST 2.0 descriptions=*"
TIMEOUT_SECONDS = 900.0
IMAGES_SECONDS = 300.0
CLEANUP_SECONDS = 120.0
CHECK_SECONDS = 120.0
RETRIES = 3
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
IMAGE_HEAD_BYTES = 65536
_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
_CLASSES = {
    "title": ElementClass.HEADING,
    "section_header": ElementClass.HEADING,
    "text": ElementClass.PARAGRAPH,
    "caption": ElementClass.CAPTION,
    "footnote": ElementClass.FOOTNOTE,
    "table": ElementClass.TABLE,
    "figure": ElementClass.FIGURE,
    "signature": ElementClass.FIGURE,
    "page_header": ElementClass.HEADER,
    "page_footer": ElementClass.FOOTER,
    "page_number": ElementClass.FOOTER,
}
_HEADING_LEVELS = {"title": 1, "section_header": 2}
_FURNITURE = {"page_header", "page_footer", "page_number"}
_PICTURES = {ElementClass.FIGURE, ElementClass.CHART}
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout)

ChartData = tuple[str | None, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class _Settings:
    host: str
    volume: str
    token: str | None
    client: tuple[str, str] | None


class _Failure(RuntimeError):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class DatabricksProvider(Provider):
    name: ClassVar[str] = "databricks"
    formats: ClassVar[frozenset[str]] = frozenset(_MEDIA_TYPES)
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

    def prepare(self) -> None:
        settings = _settings()
        deadline = self._clock() + CHECK_SECONDS
        probe = f"{settings.volume}/wholedoc-bench-check-{uuid.uuid4().hex}"
        try:
            with self._client(settings) as client:
                self._authorize(client, settings, deadline)
                self._call(client, "HEAD", _directory(settings.volume), deadline, "volume check")
                try:
                    self._call(client, "PUT", _directory(probe), deadline, "volume write check")
                finally:
                    self._remove(client, probe)
        except _Failure as error:
            if error.status == 429 or error.status >= 500:
                raise ValueError(f"{error}; try again later") from None
            if error.status == 404:
                raise CredentialError(
                    f"{error}: check that {HOST_VARIABLE} is the workspace and that "
                    f"{VOLUME_VARIABLE} is a folder in a volume"
                ) from None
            raise CredentialError(
                f"{error}: check the credentials, and that they may read and write files in "
                f"{VOLUME_VARIABLE}"
            ) from None
        except RuntimeError as error:
            raise ValueError(str(error)) from None

    def fetch(self, document_path: Path) -> RawResult:
        settings = _settings()
        media_type = _MEDIA_TYPES.get(document_path.suffix.lower())
        if media_type is None:
            raise RuntimeError("Databricks does not read this file format")
        data = base64.b64encode(document_path.read_bytes()).decode("ascii")
        deadline = self._clock() + TIMEOUT_SECONDS
        name = f"wholedoc-bench-{uuid.uuid4().hex}"
        folder = f"{settings.volume}/{name}"
        with self._client(settings) as client:
            self._authorize(client, settings, deadline)
            try:
                self._call(client, "PUT", _directory(folder), deadline, "image folder creation")
                response = self._call(
                    client,
                    "POST",
                    PARSE_PATH,
                    deadline,
                    "parse",
                    billed=True,
                    json={
                        "content": f"data:{media_type};base64,{data}",
                        "options": {**OPTIONS, "image_output_path": folder},
                    },
                )
                payload = _json(response, "parse")
                images = self._clock() + IMAGES_SECONDS
                self._retain_page_sizes(client, payload, folder, images)
            finally:
                self._remove(client, folder)
        if name in json.dumps(payload, ensure_ascii=False):
            raise RuntimeError("Databricks returned an image path where the adapter reads none")
        return RawResult(self.name, document_path.parent.name, payload, {"version": VERSION})

    def _client(self, settings: _Settings) -> httpx.Client:
        return httpx.Client(
            base_url=settings.host, transport=self._transport, follow_redirects=False
        )

    def _authorize(self, client: httpx.Client, settings: _Settings, deadline: float) -> None:
        token = settings.token
        if settings.client is not None:
            response = self._call(
                client,
                "POST",
                "/oidc/v1/token",
                deadline,
                "sign-in",
                auth=settings.client,
                data={"grant_type": "client_credentials", "scope": "all-apis"},
            )
            token = _json(response, "sign-in").get("access_token")
            if not isinstance(token, str) or not _header_safe(token):
                raise RuntimeError("Databricks sign-in returned no access token")
        client.headers["Authorization"] = f"Bearer {token}"

    def _call(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        deadline: float,
        what: str,
        *,
        billed: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        for attempt in range(RETRIES + 1):
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise RuntimeError(f"Databricks {what} timed out")
            try:
                timeout = remaining if billed else min(60.0, remaining)
                response = client.request(method, path, timeout=timeout, **kwargs)
            except httpx.RequestError as error:
                if (billed and not isinstance(error, _NOT_SENT)) or attempt == RETRIES:
                    raise RuntimeError(
                        f"Databricks {what} request failed or timed out; check {HOST_VARIABLE} "
                        "and the network"
                    ) from None
                self._wait(2 ** (attempt + 1), deadline, what)
                continue
            status = response.status_code
            if attempt < RETRIES and (status == 429 if billed else status in RETRY_STATUSES):
                self._wait(_retry_after(response, attempt), deadline, what)
                continue
            if not response.is_success:
                raise _Failure(f"Databricks {what} failed (HTTP {status}{_code(response)})", status)
            return response
        raise AssertionError("unreachable")

    def _wait(self, delay: float, deadline: float, what: str) -> None:
        if self._clock() + delay >= deadline:
            raise RuntimeError(f"Databricks {what} timed out")
        self._sleep(delay)

    def _retain_page_sizes(
        self, client: httpx.Client, response: dict[str, Any], folder: str, deadline: float
    ) -> None:
        document = response.get("document")
        pages = document.get("pages") if isinstance(document, dict) else None
        if not isinstance(document, dict) or not isinstance(pages, list) or not pages:
            raise RuntimeError("Databricks returned no parsed pages")
        if response.get("error_status"):
            raise RuntimeError("Databricks failed to parse some pages; check the document")
        for page in pages:
            path = _in_folder(folder, page.get("image_uri") if isinstance(page, dict) else None)
            if path is None:
                raise RuntimeError("Databricks returned no page image in the image folder")
            head = self._call(
                client,
                "GET",
                _file(path),
                deadline,
                "page image download",
                headers={"Range": f"bytes=0-{IMAGE_HEAD_BYTES - 1}"},
            ).content
            del page["image_uri"]
            page["image_width"], page["image_height"] = _image_size(head)
        for element in document.get("elements") or []:
            if isinstance(element, dict):
                element.pop("image_uri", None)

    def _remove(self, client: httpx.Client, folder: str) -> None:
        try:
            self._remove_tree(client, folder, self._clock() + CLEANUP_SECONDS)
        except Exception:
            name = folder.rsplit("/", 1)[1]
            print(
                f"warning: Databricks folder {name} may remain in the {VOLUME_VARIABLE} volume; "
                "delete it there",
                file=sys.stderr,
            )

    def _remove_tree(self, client: httpx.Client, folder: str, deadline: float) -> None:
        entries, token = [], None
        while True:
            try:
                response = self._call(
                    client,
                    "GET",
                    _directory(folder),
                    deadline,
                    "image folder listing",
                    params={"page_token": token} if token else None,
                )
            except _Failure as error:
                if error.status == 404:
                    return
                raise
            listing = _json(response, "image folder listing")
            entries += [e for e in listing.get("contents") or [] if isinstance(e, dict)]
            token = listing.get("next_page_token")
            if not token:
                break
        for entry in entries:
            name = entry.get("name") or str(entry.get("path") or "").rstrip("/").rsplit("/", 1)[-1]
            if not isinstance(name, str) or "/" in name or name in ("", ".", ".."):
                continue
            path = f"{folder}/{name}"
            try:
                if entry.get("is_directory"):
                    self._remove_tree(client, path, deadline)
                else:
                    self._call(client, "DELETE", _file(path), deadline, "image deletion")
            except RuntimeError:
                continue
        self._call(client, "DELETE", _directory(folder), deadline, "image folder deletion")

    def usage(self, raw: RawResult) -> dict[str, float]:
        return {"pages": len(_pages(raw.payload))}

    def normalize(self, raw: RawResult) -> Prediction:
        sizes = _pages(raw.payload)
        blocks: list[str] = []
        length = 0
        layout: list[LayoutItem] = []
        relations: list[Relation] = []
        page_starts: dict[int, int] = {}
        in_order = True
        for element in _elements(raw.payload):
            kind = element["type"]
            content = element.get("content")
            content = content.strip() if isinstance(content, str) else ""
            data = _chart_data(content) if kind == "figure" else None
            chart = _chart(data, element.get("description")) if data is not None else None
            furniture = kind in _FURNITURE
            pieces = _pieces(element.get("bbox"), sizes, consecutive=not furniture)
            cls = _class(kind, chart, pieces)
            block = _block(kind, content, data)
            span = None
            if block:
                start = length + (2 if blocks else 0)
                blocks.append(block)
                length = start + len(block)
                span = (start, length)
                if pieces:
                    page = pieces[0][0]
                    in_order = in_order and page >= max(page_starts, default=0)
                    page_starts.setdefault(page, start)
            text = markdown_text(_escape(content) if furniture else block)
            if not text and cls not in _PICTURES:
                continue
            for index, (page, box) in enumerate(pieces):
                first = index == 0
                if not first and not furniture:
                    relations.append(
                        Relation(
                            kind=RelationKind.CONTINUES_IN,
                            source=layout[-1].id,
                            target=f"databricks-{len(layout)}",
                        )
                    )
                layout.append(
                    LayoutItem(
                        id=f"databricks-{len(layout)}",
                        cls=cls,
                        page=page,
                        text=text if first or furniture else "",
                        box=box,
                        channel=Channel.FURNITURE if furniture else None,
                        chart=chart if first else None,
                        md_start=span[0] if span and first else None,
                        md_end=span[1] if span and first else None,
                    )
                )
        markdown = "\n\n".join(blocks)
        offsets = None
        if in_order:
            offsets, start = [0] * len(sizes), len(markdown)
            for page in reversed(range(len(sizes))):
                start = page_starts.get(page, start)
                offsets[page] = start
        return Prediction(
            markdown=markdown, layout=layout, relations=relations, page_offsets=offsets
        )


def _settings() -> _Settings:
    value = require_env(HOST_VARIABLE).strip()
    try:
        url = httpx.URL(value if "://" in value else f"https://{value}")
    except httpx.InvalidURL:
        url = None
    if (
        url is None
        or url.scheme != "https"
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
        or url.path not in ("", "/")
    ):
        raise CredentialError(
            f"{HOST_VARIABLE} must be the workspace URL, such as "
            "https://<workspace>.cloud.databricks.com"
        )
    host = f"https://{url.host}" + (f":{url.port}" if url.port else "")
    volume = require_env(VOLUME_VARIABLE).strip().rstrip("/")
    parts = volume.split("/")
    if parts[:2] != ["", "Volumes"] or len(parts) < 5 or {"", ".", ".."} & set(parts[2:]):
        raise CredentialError(
            f"{VOLUME_VARIABLE} must be a Unity Catalog volume or a folder in one, such as "
            "/Volumes/<catalog>/<schema>/<volume>"
        )
    token = os.environ.get(TOKEN_VARIABLE, "").strip()
    client_id = os.environ.get(CLIENT_ID_VARIABLE, "").strip()
    secret = os.environ.get(CLIENT_SECRET_VARIABLE, "").strip()
    if token and (client_id or secret):
        raise CredentialError(
            f"set {TOKEN_VARIABLE} or a service principal's {CLIENT_ID_VARIABLE} and "
            f"{CLIENT_SECRET_VARIABLE}, not both"
        )
    if token:
        if not _header_safe(token):
            raise CredentialError(f"{TOKEN_VARIABLE} holds characters that no token has")
        return _Settings(host, volume, token, None)
    if client_id and secret:
        return _Settings(host, volume, None, (client_id, secret))
    raise CredentialError(
        f"set {TOKEN_VARIABLE}, or a service principal's {CLIENT_ID_VARIABLE} and "
        f"{CLIENT_SECRET_VARIABLE}, before a live run. Do not put credentials in --config."
    )


def _header_safe(value: str) -> bool:
    return bool(value) and value.isascii() and value.isprintable()


def _file(path: str) -> str:
    return "/api/2.0/fs/files" + quote(path, safe="/")


def _directory(path: str) -> str:
    return "/api/2.0/fs/directories" + quote(path, safe="/")


def _in_folder(folder: str, path: Any) -> str | None:
    name = "/" + folder.rsplit("/", 1)[1] + "/"
    if not isinstance(path, str) or name not in path:
        return None
    rest = path.split(name, 1)[1]
    return None if {"", ".", ".."} & set(rest.split("/")) else f"{folder}/{rest}"


def _json(response: httpx.Response, what: str) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise RuntimeError(f"Databricks {what} returned an invalid response")
    return value


def _code(response: httpx.Response) -> str:
    try:
        code = response.json().get("error_code")
    except (ValueError, AttributeError):
        return ""
    return f", {code}" if isinstance(code, str) and _ERROR_CODE.fullmatch(code) else ""


def _retry_after(response: httpx.Response, attempt: int) -> float:
    try:
        delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
    except ValueError:
        delay = 2 ** (attempt + 1)
    if not math.isfinite(delay) or delay < 0:
        delay = 2 ** (attempt + 1)
    return min(delay, 60.0)


def _image_size(head: bytes) -> tuple[int, int]:
    size = None
    if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
        size = struct.unpack(">II", head[16:24])
    elif head[:2] == b"\xff\xd8":
        index = 2
        while index + 9 <= len(head) and head[index] == 0xFF:
            marker = head[index + 1]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", head[index + 5 : index + 9])
                size = (width, height)
                break
            index += 2 + struct.unpack(">H", head[index + 2 : index + 4])[0]
    if size is None or not all(size):
        raise RuntimeError("Databricks returned a page image of unknown size")
    return size


def _pages(payload: Any) -> list[tuple[int, int]]:
    document = payload.get("document") if isinstance(payload, dict) else None
    pages = document.get("pages") if isinstance(document, dict) else None
    if not isinstance(pages, list) or not pages:
        raise ValueError("Databricks returned no pages")
    if payload.get("error_status"):
        raise ValueError("Databricks failed to parse some pages")
    sizes: dict[int, tuple[int, int]] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Databricks returned an invalid page")
        number, width, height = page.get("id"), page.get("image_width"), page.get("image_height")
        if type(number) is not int or type(width) is not int or type(height) is not int:
            raise ValueError("Databricks returned a page without an id or image size")
        if number in sizes or width <= 0 or height <= 0:
            raise ValueError("Databricks returned a repeated page or a page image without area")
        sizes[number] = (width, height)
    if sorted(sizes) != list(range(len(sizes))):
        raise ValueError("Databricks returned incomplete pages")
    return [sizes[number] for number in range(len(sizes))]


def _elements(payload: dict[str, Any]) -> list[dict[str, Any]]:
    elements = payload["document"].get("elements")
    if not isinstance(elements, list):
        raise ValueError("Databricks returned no elements")
    ids = set()
    for element in elements:
        number = element.get("id") if isinstance(element, dict) else None
        if type(number) is not int or number in ids:
            raise ValueError("Databricks returned an element without an id of its own")
        ids.add(number)
        kind = element["type"] if "type" in element else None
        if not isinstance(kind, str) or kind not in _CLASSES:
            raise ValueError(f"Databricks returned an unknown element type: {kind!r}")
    return sorted(elements, key=lambda element: element["id"])


def _pieces(
    bbox: Any, sizes: list[tuple[int, int]], *, consecutive: bool
) -> list[tuple[int, Box | None]]:
    regions: dict[int, list[Any]] = {}
    for region in bbox if isinstance(bbox, list) else []:
        page = region.get("page_id") if isinstance(region, dict) else None
        if type(page) is int and 0 <= page < len(sizes):
            regions.setdefault(page, []).append(region.get("coord"))
    pages = sorted(regions)
    if consecutive and any(b - a != 1 for a, b in zip(pages, pages[1:], strict=False)):
        pages = pages[:1]
    return [(page, _box(regions[page], sizes[page])) for page in pages]


def _box(coords: list[Any], size: tuple[int, int]) -> Box | None:
    corners = []
    for coord in coords:
        if (
            not isinstance(coord, list)
            or len(coord) != 4
            or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in coord)
        ):
            return None
        corners.append(coord)
    if not corners:
        return None
    left, top = min(c[0] for c in corners), min(c[1] for c in corners)
    right, bottom = max(c[2] for c in corners), max(c[3] for c in corners)
    return clip_box(left, top, right - left, bottom - top, *size)


def _class(kind: str, chart: Chart | None, pieces: list[tuple[int, Box | None]]) -> ElementClass:
    if chart is not None:
        return ElementClass.CHART
    if kind == "page_number" and pieces and (box := pieces[0][1]) is not None:
        return ElementClass.HEADER if box[1] + box[3] / 2 < 0.5 else ElementClass.FOOTER
    return _CLASSES[kind]


def _block(kind: str, content: str, data: ChartData | None) -> str:
    if kind in _FURNITURE:
        return ""
    if kind in _HEADING_LEVELS:
        heading = " ".join(content.split())
        return "#" * _HEADING_LEVELS[kind] + " " + _escape(heading) if heading else ""
    if data is not None:
        title, series = data
        return _chart_table(series, title) if series else ""
    return content if kind == "table" else _escape(content)


def _escape(text: str) -> str:
    return "\n".join(escape_markdown(line) for line in text.split("\n"))


def _chart_data(content: str) -> ChartData | None:
    if not content.startswith("{"):
        return None
    try:
        data = json.loads(content)
    except ValueError:
        return None
    if not isinstance(data, dict) or not {"values", "additional_info"} & data.keys():
        return None
    values = data.get("values")
    if isinstance(values, dict) and all(map(_scalar, values.values())):
        series = {"": values} if values else {}
    elif isinstance(values, dict) and all(
        isinstance(points, dict) and all(map(_scalar, points.values()))
        for points in values.values()
    ):
        series = {str(name): points for name, points in values.items()}
    else:
        raise ValueError("Databricks returned chart data of an unknown shape")
    title = data.get("title")
    title = title.strip() if isinstance(title, str) and title.strip() != "None" else ""
    return title or None, series


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float)


def _chart(data: ChartData, description: Any) -> Chart:
    title, values = data
    series = []
    for name, points in values.items():
        chart_points = [
            ChartPoint(label=str(label), value=number)
            for label, value in points.items()
            if (number := _number(value)) is not None
        ]
        if chart_points:
            series.append(ChartSeries(name=name, points=chart_points))
    kind = named_chart_kind(description) if isinstance(description, str) else None
    return Chart(kind=kind, series=series, title=title)


def _number(value: Any) -> float | None:
    if type(value) in (int, float):
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if math.isfinite(number) else None
    if isinstance(value, str) and (numbers := cell_values(value)):
        return numbers[0]
    return None


def _chart_table(values: dict[str, dict[str, Any]], title: str | None) -> str:
    labels = list(dict.fromkeys(label for points in values.values() for label in points))

    def cell(value: Any) -> str:
        return "" if value is None else html.escape(str(value))

    caption = f"<caption>{html.escape(title)}</caption>" if title else ""
    head = "".join(f"<th>{html.escape(name)}</th>" for name in values)
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td>"
        + "".join(f"<td>{cell(points.get(label))}</td>" for points in values.values())
        + "</tr>"
        for label in labels
    )
    return f"<table>{caption}<thead><tr><th></th>{head}</tr></thead><tbody>{rows}</tbody></table>"
