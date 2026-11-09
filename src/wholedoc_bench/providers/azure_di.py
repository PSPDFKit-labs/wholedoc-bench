import itertools
import re
import string
import time
from bisect import bisect_left
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import httpx
import numpy as np

from wholedoc_bench.providers.base import Provider, RawResult, clip_box, grid_place
from wholedoc_bench.providers.credentials import CredentialError, require_env
from wholedoc_bench.schema import (
    Box,
    Channel,
    ElementClass,
    Font,
    FontClass,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
    TableCell,
)

API_VERSION = "2024-11-30"
MODEL_ID = "prebuilt-layout"
ENDPOINT_VARIABLE = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
KEY_VARIABLE = "AZURE_DOCUMENT_INTELLIGENCE_KEY"
VERSION_PARAMS = {"api-version": API_VERSION}
ANALYZE_PARAMS = {
    **VERSION_PARAMS,
    "outputContentFormat": "markdown",
    "stringIndexType": "unicodeCodePoint",
}
FEATURES = ("barcodes", "formulas", "ocrHighResolution", "styleFont")
FEATURE_FORMATS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".heif"})
_SENT_FEATURES = re.compile(r"; features ([A-Za-z+]+)$")
READ_SECONDS = 120.0
CONNECT_SECONDS = 10.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
START_RETRY_STATUSES = frozenset({429, 503})
RETRIES = 3
POLL_SECONDS = 2.0
MIN_WAIT_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 60.0
MAX_WAIT_SECONDS = 600.0
CHECK_SECONDS = 120.0

_ROLES = {
    None: ElementClass.PARAGRAPH,
    "title": ElementClass.HEADING,
    "sectionHeading": ElementClass.HEADING,
    "footnote": ElementClass.FOOTNOTE,
    "formulaBlock": ElementClass.FORMULA,
    "pageHeader": ElementClass.HEADER,
    "pageFooter": ElementClass.FOOTER,
}
_FURNITURE_ROLES = frozenset({"pageHeader", "pageFooter", "pageNumber"})
_UNSTYLED_ROLES = _FURNITURE_ROLES | {"formulaBlock"}
_HEADING_MARK = re.compile(r"#{1,6}[ \t]+")
_BACKSLASHES = re.compile(r"\\+")
_STYLE_TAGS = (("fontWeight", "bold", "b"), ("fontStyle", "italic", "i"))
_PLACEHOLDER = re.compile(r":(formula|barcode|selected|unselected):")
_MARKS = {"selected": "\u2612", "unselected": "\u2610"}
_ERROR_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")

Note = tuple[ElementClass, RelationKind, str]


class AzureDIProvider(Provider):
    name: ClassVar[str] = "azure-di"
    sends_documents: ClassVar[bool] = True
    formats: ClassVar[frozenset[str]] = frozenset(
        {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".heif",
            ".docx",
            ".xlsx",
            ".pptx",
            ".html",
        }
    )
    meters: ClassVar[frozenset[str]] = frozenset({"pages", *(f"{f}_pages" for f in FEATURES)})

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
        if unknown := config.keys() - {"features"}:
            raise ValueError(f"{cls.name}: unsupported config keys: {', '.join(sorted(unknown))}")
        features = config.get("features", list(FEATURES))
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise ValueError(f"{cls.name}: features must be a list of feature names")
        if unknown := set(features) - set(FEATURES):
            raise ValueError(
                f"{cls.name}: unsupported features: {', '.join(sorted(unknown))}; "
                f"choose from {', '.join(FEATURES)}"
            )
        return {"features": sorted(set(features))}

    def prepare(self) -> None:
        endpoint, key = _settings()
        deadline = _Deadline(self._clock, CHECK_SECONDS)
        try:
            with self._client(key) as client:
                info = f"{endpoint}/documentintelligence/info"
                response = self._send(client, "GET", info, deadline, params=VERSION_PARAMS)
        except _TimeUp:
            raise ValueError(
                f"Azure Document Intelligence did not answer within {CHECK_SECONDS:.0f} "
                "seconds; try again later."
            ) from None
        except httpx.RequestError:
            raise CredentialError(
                f"Azure Document Intelligence could not be reached; check {ENDPOINT_VARIABLE} "
                "and the network."
            ) from None
        status = response.status_code
        if status == 200:
            return
        if status == 429 or status >= 500:
            state = "is limiting requests" if status == 429 else "is unavailable"
            raise ValueError(
                f"Azure Document Intelligence {state} ({_reason(response)}); try again later."
            )
        if status in (401, 403):
            raise CredentialError(
                f"Azure Document Intelligence did not accept the key ({_reason(response)}); "
                f"check {KEY_VARIABLE} and {ENDPOINT_VARIABLE}. "
                "Do not put credentials in --config."
            )
        raise CredentialError(
            f"Azure Document Intelligence answered {_reason(response)} at the endpoint; "
            f"check {ENDPOINT_VARIABLE}."
        )

    def fetch(self, document_path: Path) -> RawResult:
        endpoint, key = _settings()
        features = self.config["features"]
        if document_path.suffix.lower() not in FEATURE_FORMATS:
            features = []
        params = {**ANALYZE_PARAMS, **({"features": ",".join(features)} if features else {})}
        try:
            payload = self._analyze(endpoint, key, params, document_path.read_bytes())
        except _TimeUp:
            raise RuntimeError(
                "Azure Document Intelligence analysis did not finish within "
                f"{MAX_WAIT_SECONDS:.0f} seconds."
            ) from None
        except httpx.RequestError:
            raise RuntimeError(
                "Azure Document Intelligence request failed; check network connectivity."
            ) from None
        result = payload["analyzeResult"]
        model = result.get("modelId") or "unknown"
        version = (
            f"azure-di {model}; API {result.get('apiVersion') or 'unknown'}; "
            f"features {'+'.join(features) or 'none'}"
        )
        return RawResult(
            provider=self.name,
            document_id=document_path.parent.name,
            payload=payload,
            metadata={"version": version, "config": dict(self.config)},
        )

    def _client(self, key: str) -> httpx.Client:
        headers = {"Ocp-Apim-Subscription-Key": key}
        return httpx.Client(transport=self._transport, headers=headers)

    def _analyze(
        self, endpoint: str, key: str, params: dict[str, str], document: bytes
    ) -> dict[str, Any]:
        url = f"{endpoint}/documentintelligence/documentModels/{MODEL_ID}:analyze"
        deadline = _Deadline(self._clock, MAX_WAIT_SECONDS)
        with self._client(key) as client:
            response = self._send(
                client,
                "POST",
                url,
                deadline,
                retry=START_RETRY_STATUSES,
                params=params,
                content=document,
                headers={"Content-Type": "application/octet-stream"},
            )
            if response.status_code != 202:
                raise RuntimeError(_failure(response))
            status_url = _status_url(response, endpoint)
            while True:
                delay = _retry_after(response, POLL_SECONDS)
                if delay > deadline.left():
                    raise _TimeUp
                self._sleep(delay)
                response = self._send(client, "GET", status_url, deadline)
                if response.status_code != 200:
                    raise RuntimeError(_failure(response))
                body = response.json()
                status = body.get("status")
                if status == "succeeded":
                    return body
                if status not in ("notStarted", "running"):
                    raise RuntimeError(
                        f"Azure Document Intelligence analysis ended as {status!r} "
                        f"({_code(body) or 'no error code'}); check the document requirements."
                    )

    def _send(
        self,
        client: httpx.Client,
        method: str,
        url: str,
        deadline: "_Deadline",
        retry: frozenset[int] = RETRY_STATUSES,
        **kwargs: Any,
    ) -> httpx.Response:
        attempt = 0
        while True:
            backoff = 2.0 * 2**attempt
            if (left := deadline.left()) <= 0:
                raise _TimeUp
            timeout = httpx.Timeout(min(READ_SECONDS, left), connect=min(CONNECT_SECONDS, left))
            try:
                response = client.request(method, url, timeout=timeout, **kwargs)
            except httpx.TransportError:
                if deadline.left() <= 0:
                    raise _TimeUp from None
                if method != "GET" or attempt == RETRIES or backoff > deadline.left():
                    raise
                wait = backoff
            else:
                wait = _retry_after(response, backoff)
                if response.status_code not in retry or attempt == RETRIES:
                    return response
                if wait > deadline.left():
                    return response
            self._sleep(wait)
            attempt += 1

    def usage(self, raw: RawResult) -> dict[str, float]:
        pages = len(raw.payload["analyzeResult"]["pages"])
        sent = _SENT_FEATURES.search(str(raw.metadata.get("version")))
        if sent is None:
            raise ValueError("the response's version does not name the features sent")
        features = set(sent[1].split("+")) - {"none"}
        return {"pages": pages, **{f"{f}_pages": pages * (f in features) for f in FEATURES}}

    def normalize(self, raw: RawResult) -> Prediction:
        result = raw.payload["analyzeResult"]
        settings = (result.get("contentFormat"), result.get("stringIndexType"))
        if settings != ("markdown", "unicodeCodePoint"):
            raise ValueError(f"expected Markdown with code point spans, got {settings}")
        markdown = _Markdown(result)
        offsets = _page_offsets(result)
        layout, relations = _Layout(result, offsets, markdown.at).read()
        return Prediction(
            markdown=markdown.text,
            layout=layout,
            relations=relations,
            page_offsets=[markdown.at(offset) for offset in offsets],
        )


class _TimeUp(Exception): ...


class _Deadline:
    def __init__(self, clock: Callable[[], float], seconds: float) -> None:
        self.clock = clock
        self.end = clock() + seconds

    def left(self) -> float:
        return self.end - self.clock()


def _settings() -> tuple[str, str]:
    value = require_env(ENDPOINT_VARIABLE).strip()
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL:
        url = None
    if (
        url is None
        or url.scheme != "https"
        or not url.host
        or url.userinfo
        or url.query
        or url.path not in ("", "/")
    ):
        raise CredentialError(
            f"{ENDPOINT_VARIABLE} must be the resource's https endpoint, such as "
            "https://<resource>.cognitiveservices.azure.com/"
        )
    endpoint = f"https://{url.host}" + (f":{url.port}" if url.port else "")
    key = require_env(KEY_VARIABLE).strip()
    if not (key.isascii() and key.isprintable()):
        raise CredentialError(
            f"{KEY_VARIABLE} holds characters that no key has; copy it again from the Azure portal."
        )
    return endpoint, key


def _status_url(response: httpx.Response, endpoint: str) -> str:
    location = response.headers.get("Operation-Location", "")
    try:
        url = httpx.URL(location)
    except httpx.InvalidURL:
        url = None
    if url is None or url.scheme != "https" or url.netloc != httpx.URL(endpoint).netloc:
        raise RuntimeError(
            "Azure Document Intelligence returned no status URL on the endpoint's host."
        )
    return location


def _retry_after(response: httpx.Response, default: float) -> float:
    after = response.headers.get("Retry-After", "")
    wait = float(after) if after.isdigit() else default
    return min(max(wait, MIN_WAIT_SECONDS), MAX_RETRY_AFTER_SECONDS)


def _code(body: Any) -> str | None:
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None
    inner = error.get("innererror")
    for code in ((inner or {}).get("code") if isinstance(inner, dict) else None, error.get("code")):
        if isinstance(code, str) and _ERROR_CODE.fullmatch(code):
            return code
    return None


def _reason(response: httpx.Response) -> str:
    try:
        code = _code(response.json())
    except ValueError:
        code = None
    return f"HTTP {response.status_code}" + (f", {code}" if code else "")


def _failure(response: httpx.Response) -> str:
    return (
        f"Azure Document Intelligence request failed ({_reason(response)}); check the key, "
        "the endpoint, the resource's limits and the document requirements."
    )


def _page_offsets(result: dict[str, Any]) -> list[int]:
    offsets: list[int] = []
    following = len(result["content"])
    for page in reversed(result["pages"]):
        spans = page.get("spans") or []
        following = min((s["offset"] for s in spans), default=following)
        offsets.append(following)
    return offsets[::-1]


class _Markdown:
    def __init__(self, result: dict[str, Any]) -> None:
        content: str = result["content"]
        tags = sorted(_style_tags(result))
        pieces, last = [], 0
        for offset, _, tag in tags:
            pieces += [content[last:offset], tag]
            last = offset
        self.text = "".join([*pieces, content[last:]])
        self.keys = [(offset, opens) for offset, opens, _ in tags]
        self.added = [0, *itertools.accumulate(len(tag) for _, _, tag in tags)]

    def at(self, offset: int) -> int:
        return offset + self.added[bisect_left(self.keys, (offset, True))]


def _style_tags(result: dict[str, Any]) -> list[tuple[int, bool, str]]:
    content: str = result["content"]
    size = len(content)

    def mask(spans: Iterator[dict[str, int]]) -> np.ndarray:
        out = np.zeros(size, dtype=bool)
        for span in spans:
            out[max(0, span["offset"]) : max(0, span["offset"] + span["length"])] = True
        return out

    paragraphs = result.get("paragraphs") or []
    pages = result["pages"]
    allowed = mask(
        s for p in paragraphs if p.get("role") not in _UNSTYLED_ROLES for s in p.get("spans") or []
    )
    allowed &= ~mask(s for t in result.get("tables") or [] for s in t.get("spans") or [])
    inline = [
        e for page in pages for kind in ("formulas", "barcodes") for e in page.get(kind) or []
    ]
    allowed &= ~mask(e["span"] for e in inline if e.get("span"))
    allowed &= np.frombuffer(content.encode("utf-32-le"), dtype="<u4") != ord("\n")
    for paragraph in paragraphs:
        if paragraph.get("role") in ("title", "sectionHeading") and paragraph.get("spans"):
            if marks := _HEADING_MARK.match(content, paragraph["spans"][0]["offset"]):
                allowed[marks.start() : marks.end()] = False
    state = np.zeros(size, dtype=np.int8)
    for bit, (key, value, _) in enumerate(_STYLE_TAGS):
        styled = mask(
            s
            for st in result.get("styles") or []
            if st.get(key) == value
            for s in st.get("spans") or []
        )
        state |= (styled & allowed).astype(np.int8) << bit
    if escapes := _escapes(content):
        state[escapes] = state[[i + 1 for i in escapes]]
    tags: list[tuple[int, bool, str]] = []
    edges = np.flatnonzero(np.diff(state, prepend=0, append=0)).tolist()
    for start, end in itertools.pairwise(edges):
        names = [tag for bit, (_, _, tag) in enumerate(_STYLE_TAGS) if state[start] >> bit & 1]
        while start < end and content[start].isspace():
            start += 1
        while end > start and content[end - 1].isspace():
            end -= 1
        if not names or start == end:
            continue
        tags.append((start, True, "".join(f"<{tag}>" for tag in names)))
        tags.append((end, False, "".join(f"</{tag}>" for tag in reversed(names))))
    return tags


def _escapes(content: str) -> list[int]:
    found: list[int] = []
    for run in _BACKSLASHES.finditer(content):
        start, end = run.span()
        found += range(start, end - 1, 2)
        if (end - start) % 2 and end < len(content) and content[end] in string.punctuation:
            found.append(end - 1)
    return found


_GENERIC = {
    "sans-serif": FontClass.SANS,
    "serif": FontClass.SERIF,
    "monospace": FontClass.MONO,
    "cursive": FontClass.SCRIPT,
    "fantasy": FontClass.DECORATIVE,
}


def _font(families: str) -> Font | None:
    names = [name.strip().strip("'\"") for name in families.split(",")]
    names = [name for name in names if name]
    generic = next((_GENERIC[n.casefold()] for n in names if n.casefold() in _GENERIC), None)
    family = next((n for n in names if n.casefold() not in _GENERIC), None)
    if family is None and generic is None:
        return None
    return Font(family=family, font_class=generic)


class _Layout:
    def __init__(
        self, result: dict[str, Any], page_offsets: list[int], at: Callable[[int], int]
    ) -> None:
        self.pages = {page["pageNumber"]: page for page in result["pages"]}
        self.page_offsets = page_offsets
        self.at = at
        self.paragraphs: list[dict[str, Any]] = result.get("paragraphs") or []
        self.tables: list[dict[str, Any]] = result.get("tables") or []
        self.figures: list[dict[str, Any]] = result.get("figures") or []
        self.content: str = result["content"]
        self.fonts = [
            (span["offset"], span["offset"] + span["length"], style["similarFontFamily"])
            for style in result.get("styles") or []
            if isinstance(style.get("similarFontFamily"), str)
            for span in style.get("spans") or []
        ]
        pages = list(self.pages.values())
        self.detected = sorted(
            [
                (element["span"]["offset"], kind, element.get("value") or "")
                for page in pages
                for kind in ("formula", "barcode")
                for element in page.get(f"{kind}s") or []
                if element.get("span")
            ]
            + [
                (mark["span"]["offset"], mark["state"], _MARKS[mark["state"]])
                for page in pages
                for mark in page.get("selectionMarks") or []
                if mark.get("span") and mark.get("state") in _MARKS
            ]
        )

    def read(self) -> tuple[list[LayoutItem], list[Relation]]:
        items: list[LayoutItem] = []
        held: set[int] = set()
        notes: dict[int, Note] = {}
        for index, table in enumerate(self.tables):
            cells = sorted(table["cells"], key=lambda c: (c["rowIndex"], c["columnIndex"]))
            held |= {i for cell in cells for i in _paragraphs(cell)}
            page = self.page_of(table)
            texts = [self.text(cell) for cell in cells]
            item = self.item(
                f"tables/{index}",
                ElementClass.TABLE,
                table,
                " ".join(text for text in texts if text),
                cells=[
                    TableCell(
                        text=text,
                        box=self.box(cell, page),
                        **grid_place(
                            cell["rowIndex"],
                            cell["columnIndex"],
                            cell.get("rowSpan", 1),
                            cell.get("columnSpan", 1),
                        ),
                    )
                    for cell, text in zip(cells, texts, strict=True)
                ],
            )
            items.append(item)
            notes |= _notes(table, item.id)
        for index, figure in enumerate(self.figures):
            inside = _paragraphs(figure) if figure.get("spans") else []
            held |= set(inside)
            text = " ".join(t for i in inside if (t := self.text(self.paragraphs[i])))
            item = self.item(f"figures/{index}", ElementClass.FIGURE, figure, text)
            items.append(item)
            notes |= _notes(figure, item.id)
        for page in self.pages.values():
            for number, barcode in enumerate(page.get("barcodes") or [], start=1):
                element = {
                    "spans": [barcode["span"]] if barcode.get("span") else [],
                    "boundingRegions": [
                        {"pageNumber": page["pageNumber"], "polygon": barcode.get("polygon")}
                    ],
                }
                item_id = f"barcodes/{page['pageNumber']}.{number}"
                items.append(
                    self.item(
                        item_id, ElementClass.MACHINE_READABLE, element, barcode.get("value") or ""
                    )
                )
                held |= {i for i, p in enumerate(self.paragraphs) if _within(p, element)}
        relations: list[Relation] = []
        for index, paragraph in enumerate(self.paragraphs):
            if index in held:
                continue
            role = paragraph.get("role")
            note = notes.get(index) if role not in _FURNITURE_ROLES else None
            if note is not None:
                cls = note[0]
            elif role == "pageNumber":
                cls = self.page_number_class(paragraph)
            elif role in _ROLES:
                cls = _ROLES[role]
            else:
                raise ValueError(f"paragraph {index}: no class mapping for Azure role {role!r}")
            channel = Channel.FURNITURE if role in _FURNITURE_ROLES else Channel.BODY
            item = self.item(f"paragraphs/{index}", cls, paragraph, self.text(paragraph), channel)
            items.append(item)
            if note is not None:
                relations.append(Relation(kind=note[1], source=item.id, target=note[2]))
        items.sort(key=lambda i: (i.page, i.md_start is None, i.md_start or 0))
        return items, relations

    def item(
        self,
        item_id: str,
        cls: ElementClass,
        element: dict[str, Any],
        text: str,
        channel: Channel = Channel.BODY,
        cells: list[TableCell] | None = None,
    ) -> LayoutItem:
        spans = element.get("spans") or []
        page = self.page_of(element)
        return LayoutItem(
            id=item_id,
            cls=cls,
            page=page,
            text=text,
            box=self.box(element, page),
            channel=channel,
            md_start=self.at(min(s["offset"] for s in spans)) if spans else None,
            md_end=self.at(max(s["offset"] + s["length"] for s in spans)) if spans else None,
            cells=cells,
            font=self.font(spans) if cells is None else None,
        )

    def font(self, spans: list[dict[str, int]]) -> Font | None:
        total = sum(span["length"] for span in spans)
        weights: dict[str, int] = {}
        for span in spans:
            start, end = span["offset"], span["offset"] + span["length"]
            for first, last, family in self.fonts:
                if (shared := min(end, last) - max(start, first)) > 0:
                    weights[family] = weights.get(family, 0) + shared
        if not total or not weights:
            return None
        family, weight = max(weights.items(), key=lambda pair: (pair[1], pair[0]))
        return _font(family) if 2 * weight >= total else None

    def text(self, element: dict[str, Any]) -> str:
        text = element.get("content") or ""
        if not _PLACEHOLDER.search(text):
            return text
        printed = [
            (found.start(), found[1], None)
            for span in element.get("spans") or []
            for found in _PLACEHOLDER.finditer(
                self.content, span["offset"], span["offset"] + span["length"]
            )
        ]
        detected = [d for d in self.detected if _holds(element, d[0])]
        values: dict[str, list[str | None]] = {}
        for _, kind, value in sorted(printed + detected, key=lambda entry: entry[0]):
            values.setdefault(kind, []).append(value)

        def substitute(match: re.Match[str]) -> str:
            queue = values.get(match[1])
            replacement = queue.pop(0) if queue else None
            return match[0] if replacement is None else replacement

        return _PLACEHOLDER.sub(substitute, text)

    def page_of(self, element: dict[str, Any]) -> int:
        if regions := element.get("boundingRegions"):
            return regions[0]["pageNumber"] - 1
        start = min((s["offset"] for s in element.get("spans") or []), default=0)
        return max((i for i, s in enumerate(self.page_offsets) if s <= start), default=0)

    def box(self, element: dict[str, Any], page: int) -> Box | None:
        info = self.pages.get(page + 1, {})
        points = [
            region["polygon"]
            for region in element.get("boundingRegions") or []
            if region["pageNumber"] == page + 1 and region.get("polygon")
        ]
        width, height = info.get("width"), info.get("height")
        if not points or not width or not height:
            return None
        xs = [x for polygon in points for x in polygon[0::2]]
        ys = [y for polygon in points for y in polygon[1::2]]
        left, top = min(xs), min(ys)
        return clip_box(left, top, max(xs) - left, max(ys) - top, width, height)

    def page_number_class(self, paragraph: dict[str, Any]) -> ElementClass:
        box = self.box(paragraph, self.page_of(paragraph))
        middle = box[1] + box[3] / 2 if box else 1.0
        return ElementClass.HEADER if middle < 0.5 else ElementClass.FOOTER


def _holds(element: dict[str, Any], offset: int) -> bool:
    return any(
        s["offset"] <= offset < s["offset"] + s["length"] for s in element.get("spans") or []
    )


def _within(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    spans = inner.get("spans") or []
    return bool(spans) and all(
        any(
            o["offset"] <= s["offset"] and s["offset"] + s["length"] <= o["offset"] + o["length"]
            for o in outer["spans"]
        )
        for s in spans
    )


def _paragraphs(element: dict[str, Any]) -> list[int]:
    prefix = "/paragraphs/"
    refs = element.get("elements") or []
    return [int(ref[len(prefix) :]) for ref in refs if ref.startswith(prefix)]


def _notes(element: dict[str, Any], item_id: str) -> dict[int, Note]:
    parts = [(element.get("caption"), ElementClass.CAPTION, RelationKind.CAPTION_OF)]
    for footnote in element.get("footnotes") or []:
        parts.append((footnote, ElementClass.FOOTNOTE, RelationKind.FOOTNOTE_OF))
    return {i: (cls, kind, item_id) for part, cls, kind in parts if part for i in _paragraphs(part)}
