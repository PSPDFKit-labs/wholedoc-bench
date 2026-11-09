import html
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import httpx

from wholedoc_bench.providers.base import Provider, RawResult, clip_box, grid_place
from wholedoc_bench.providers.charts import named_chart_kind
from wholedoc_bench.providers.credentials import require_env
from wholedoc_bench.schema import (
    Box,
    Channel,
    Chart,
    ChartKind,
    ChartPoint,
    ChartSeries,
    ElementClass,
    Font,
    FontClass,
    LayoutItem,
    Prediction,
    Relation,
    RelationKind,
    TableCell,
)
from wholedoc_bench.scorer.utils.blocks import MarkdownTable, parse_markdown_tables
from wholedoc_bench.scorer.utils.tracks.t6_charts.charts import cell_values

API_URL = "https://api.nutrient.io/extraction/parse"
API_VERSION = "2026-05-25"
ENGINE_VERSION = "latest"
API_KEY_VARIABLE = "NUTRIENT_API_KEY"
TIMEOUT_SECONDS = 330.0
RETRY_STATUSES = frozenset({429, 500, 503})
RETRIES = 3

_ROLE_TO_CLASS = {
    None: ElementClass.PARAGRAPH,
    "Text": ElementClass.PARAGRAPH,
    "Title": ElementClass.HEADING,
    "SectionHeader": ElementClass.HEADING,
    "ListItem": ElementClass.LIST_ITEM,
    "Footnote": ElementClass.FOOTNOTE,
    "Code": ElementClass.CODE,
    "Caption": ElementClass.CAPTION,
    "Header": ElementClass.HEADER,
    "Footer": ElementClass.FOOTER,
}
_SKIPPED_ROLES = frozenset({"CheckboxSelected", "CheckboxUnselected"})
_SKIPPED_TYPES = frozenset({"keyValueRegion"})
_FURNITURE = {ElementClass.HEADER, ElementClass.FOOTER}
Rows = tuple[tuple[str, ...], ...]

_CAPTION = re.compile(r"<caption[^>]*>(.*?)</caption>", re.IGNORECASE | re.DOTALL)


class NutrientProvider(Provider):
    name: ClassVar[str] = "nutrient"
    sends_documents: ClassVar[bool] = True
    formats: ClassVar[frozenset[str]] = frozenset(
        {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    meters: ClassVar[frozenset[str]] = frozenset({"credits", "pages"})

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(config)
        self._transport = transport
        self._sleep = sleep

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("provider config must be a JSON object")
        unknown = config.keys() - {"mode"}
        if unknown:
            raise ValueError(f"nutrient: unsupported config keys: {', '.join(sorted(unknown))}")
        mode = config.get("mode", "understand")
        if mode not in ("structure", "understand", "agentic"):
            raise ValueError("nutrient mode must be structure, understand or agentic")
        return {"mode": mode}

    @property
    def mode(self) -> str:
        return self.config["mode"]

    def prepare(self) -> None:
        require_env(API_KEY_VARIABLE)

    def fetch(self, document_path: Path) -> RawResult:
        key = require_env(API_KEY_VARIABLE)
        try:
            payload = self._post(document_path, key)
        except httpx.RequestError:
            raise RuntimeError("Nutrient API request failed; check network connectivity.") from None
        return RawResult(
            provider=self.name,
            document_id=document_path.parent.name,
            payload=payload,
            metadata={"version": self.version(), "config": dict(self.config)},
        )

    def version(self) -> str:
        return f"nutrient {self.mode} mode, API {API_VERSION}, engine {ENGINE_VERSION}"

    def instructions(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "output": {
                "formats": ["spatial", "markdown"],
                "includeHeadersAndFooters": False,
                "useHtmlTables": True,
                "enableSemanticBlockFormatting": True,
                "enableInlineFormatting": True,
            },
        }

    def _post(self, document_path: Path, key: str) -> Any:
        headers = {
            "Authorization": f"Bearer {key}",
            "x-nutrient-api-version": API_VERSION,
            "x-nutrient-engine-version": ENGINE_VERSION,
        }
        data = {"instructions": json.dumps(self.instructions())}
        with httpx.Client(transport=self._transport, timeout=TIMEOUT_SECONDS) as client:
            attempt = 0
            while True:
                with document_path.open("rb") as file:
                    response = client.post(
                        API_URL, headers=headers, data=data, files={"file": file}
                    )
                if response.status_code not in RETRY_STATUSES or attempt == RETRIES:
                    break
                self._sleep(_retry_delay(response, attempt))
                attempt += 1
        if response.status_code != 200:
            raise RuntimeError(
                f"Nutrient API {response.status_code}: request failed; "
                "check credentials, account limits, and document requirements."
            )
        return response.json()

    def usage(self, raw: RawResult) -> dict[str, float]:
        usage = raw.payload.get("usage") or {}
        credits = (usage.get("data_extraction_credits") or {}).get("cost")
        if isinstance(credits, int | float):
            return {"credits": credits}
        pages = (raw.payload.get("metrics") or {}).get("pagesProcessed")
        if not isinstance(pages, int):
            raise ValueError("Nutrient's response has neither credits nor a page count")
        return {"pages": pages}

    def normalize(self, raw: RawResult) -> Prediction:
        output = raw.payload["output"]
        elements: list[dict[str, Any]] = output["elements"]
        markdown_tables = {_rows(t) for t in parse_markdown_tables(output["markdown"])}
        layout = [item for e in elements if (item := _layout_item(e, markdown_tables)) is not None]
        unscored = [text for e in elements if _skipped(e) for text in _skipped_texts(e)]
        kept = {item.id for item in layout}
        relations: list[Relation] = []
        for e in elements:
            for kind, key in (
                (RelationKind.CAPTION_OF, "captionIds"),
                (RelationKind.FOOTNOTE_OF, "footnoteIds"),
            ):
                for source in e.get(key) or []:
                    if source in kept and e["id"] in kept:
                        relations.append(Relation(kind=kind, source=source, target=e["id"]))
        by_id = {e["id"]: e for e in elements}
        pieces = dict.fromkeys(
            [(e["id"], c["id"]) for e in elements if (c := e.get("continuesTo"))]
            + [(c["id"], e["id"]) for e in elements if (c := e.get("continuedFrom"))]
        )
        for source, target in pieces:
            if source in kept and target in kept and source != target:
                relations.append(
                    Relation(
                        kind=RelationKind.CONTINUES_IN,
                        source=source,
                        target=target,
                        hyphenated=(by_id[source].get("text") or "").rstrip().endswith("-"),
                    )
                )
        return Prediction(
            markdown=output["markdown"],
            layout=layout,
            relations=relations,
            unscored_text=unscored,
        )


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    after = response.headers.get("Retry-After", "")
    return float(after) if after.isdigit() else 2.0 * 2**attempt


def _layout_item(e: dict[str, Any], markdown_tables: set[Rows]) -> LayoutItem | None:
    page = e["page"]
    kind = e["type"]
    cells: list[TableCell] | None = None
    chart: Chart | None = None
    if _skipped(e):
        return None
    if kind == "table":
        cls = ElementClass.TABLE
        raw_cells = sorted(e["cells"], key=lambda c: (c["row"], c["column"]))
        text = " ".join(c["text"] for c in raw_cells)
        cells = [
            TableCell(
                text=c["text"],
                box=_page_box(c.get("bounds"), page),
                **grid_place(c["row"], c["column"], c.get("rowSpan", 1), c.get("colSpan", 1)),
            )
            for c in raw_cells
        ]
    elif kind == "chart":
        cls = ElementClass.CHART
        text = _html_table_text(e.get("htmlTable") or "")
        chart = _chart(e, markdown_tables)
    elif kind in ("machineReadable", "barcode"):
        cls = ElementClass.MACHINE_READABLE
        text = e.get("value") or ""
    elif kind == "picture":
        cls = ElementClass.FIGURE
        text = e.get("altDescription") or ""
    elif kind == "formula":
        cls = ElementClass.FORMULA
        text = e.get("latex") or ""
    elif kind == "handwriting":
        cls = ElementClass.PARAGRAPH
        text = e.get("text") or ""
    elif kind == "paragraph" and e.get("role") == "PageNumber":
        cls = _page_number_class(e["bounds"], page)
        text = e.get("text") or ""
    elif kind == "paragraph":
        role = e.get("role")
        if role not in _ROLE_TO_CLASS:
            raise ValueError(f"element {e['id']}: no class mapping for Nutrient role {role!r}")
        cls = _ROLE_TO_CLASS[role]
        text = e.get("text") or ""
    else:
        raise ValueError(f"element {e['id']}: no class mapping for Nutrient type {kind!r}")
    return LayoutItem(
        id=e["id"],
        cls=cls,
        page=page["pageIndex"],
        text=text,
        box=_page_box(e["bounds"], page),
        channel=Channel.FURNITURE if cls in _FURNITURE else Channel.BODY,
        cells=cells,
        chart=chart,
        font=_font(e.get("font")),
        format=e.get("format") if cls is ElementClass.MACHINE_READABLE else None,
        check_digits_valid=e.get("checkDigitsValid")
        if cls is ElementClass.MACHINE_READABLE
        else None,
    )


def _font(raw: Any) -> Font | None:
    if not isinstance(raw, dict):
        return None
    family = raw.get("family")
    family = family if isinstance(family, str) and family.strip() else None
    name = raw.get("class")
    name = name.strip().casefold() if isinstance(name, str) else None
    font_class = FontClass(name) if name in FontClass._value2member_map_ else None
    size = raw.get("size")
    size = (
        float(size) if isinstance(size, int | float) and math.isfinite(size) and size > 0 else None
    )
    if family is None and font_class is None and size is None:
        return None
    return Font(family=family, font_class=font_class, size=size)


def _skipped(e: dict[str, Any]) -> bool:
    return e["type"] in _SKIPPED_TYPES or e.get("role") in _SKIPPED_ROLES


def _skipped_texts(e: dict[str, Any]) -> list[str]:
    if e["type"] == "keyValueRegion":
        sides = (pair.get(side) or {} for pair in e.get("pairs") or [] for side in ("key", "value"))
        return [text for side in sides if (text := side.get("value"))]
    return [text] if (text := e.get("text")) else []


def _page_number_class(bounds: dict[str, float], page: dict[str, Any]) -> ElementClass:
    middle = bounds["y"] + bounds["height"] / 2
    return ElementClass.HEADER if middle < page["height"] / 2 else ElementClass.FOOTER


def _chart(e: dict[str, Any], markdown_tables: set[Rows]) -> Chart:
    source = e.get("htmlTable") or ""
    title = None
    if caption := _CAPTION.search(source):
        title = html.unescape(re.sub(r"<[^>]+>", "", caption[1])).strip() or None
    series: list[ChartSeries] = []
    tables = parse_markdown_tables(source)
    if tables and _rows(tables[0]) not in markdown_tables and len(tables[0].rows) > 1:
        names, *rows = tables[0].rows
        for column, name in enumerate(names[1:], start=1):
            points = [
                ChartPoint(label=row[0], value=values[0])
                for row in rows
                if column < len(row) and (values := cell_values(row[column]))
            ]
            if points:
                series.append(ChartSeries(name=name, points=points))
    kind = _classification_kind(e.get("classification")) or named_chart_kind(e.get("summary") or "")
    return Chart(kind=kind, series=series, title=title)


def _classification_kind(classification: Any) -> ChartKind | None:
    if not isinstance(classification, str):
        return None
    name = re.sub(r"[\s-]+", "_", classification.strip().lower()).removesuffix("_chart")
    return ChartKind(name) if name in ChartKind else None


def _rows(table: MarkdownTable) -> Rows:
    return tuple(tuple(cell.strip() for cell in row) for row in table.rows)


def _html_table_text(html: str) -> str:
    tables = parse_markdown_tables(html)
    return " ".join(cell for table in tables for row in table.rows for cell in row if cell)


def _page_box(bounds: dict[str, float] | None, page: dict[str, Any]) -> Box | None:
    if bounds is None:
        return None
    return clip_box(**bounds, page_width=page["width"], page_height=page["height"])
