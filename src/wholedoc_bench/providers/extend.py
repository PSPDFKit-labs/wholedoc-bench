import hashlib
import json
import mimetypes
import os
import re
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from wholedoc_bench.providers.base import Provider, RawResult, clip_box
from wholedoc_bench.providers.credentials import CredentialError
from wholedoc_bench.schema import Channel, ElementClass, LayoutItem, Prediction, TableCell
from wholedoc_bench.scorer.utils.blocks import markdown_text

API_URL = "https://api.extend.ai"
API_VERSION = "2026-02-09"
TIMEOUT_SECONDS = 900.0
CONFIG = {
    "engine": "parse_performance",
    "engineVersion": "2.0.0",
    "chunkingStrategy": {"type": "page"},
    "blockOptions": {
        "tables": {"targetFormat": "html"},
        "figures": {
            "enabled": True,
            "advancedChartExtractionEnabled": True,
            "figureImageClippingEnabled": False,
        },
    },
    "advancedOptions": {"pageRotationEnabled": False},
}
ENGINE_VERSIONS = {"parse_performance": "2.0.0", "parse_light": "1.0.0", "parse_auto": "1.0.0"}
_CLASSES = {
    "text": ElementClass.PARAGRAPH,
    "key_value": ElementClass.PARAGRAPH,
    "heading": ElementClass.HEADING,
    "section_heading": ElementClass.HEADING,
    "table": ElementClass.TABLE,
    "figure": ElementClass.FIGURE,
    "page_number": ElementClass.FOOTER,
    "header": ElementClass.HEADER,
    "footer": ElementClass.FOOTER,
    "formula": ElementClass.FORMULA,
    "barcode": ElementClass.MACHINE_READABLE,
}


class ExtendProvider(Provider):
    name = "extend"
    formats = frozenset(
        {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".webp",
            ".gif",
            ".heic",
            ".heif",
            ".docx",
            ".xlsx",
        }
    )
    meters = frozenset({"credits", "pages"})

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

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("provider config must be a JSON object")
        if unknown := config.keys() - {"engine", "agentic_text"}:
            raise ValueError(f"{cls.name}: unsupported config keys: {', '.join(sorted(unknown))}")
        engine = config.get("engine", "parse_performance")
        if not isinstance(engine, str) or engine not in ENGINE_VERSIONS:
            raise ValueError(f"{cls.name}: engine must be one of {', '.join(ENGINE_VERSIONS)}")
        agentic_text = config.get("agentic_text", False)
        if not isinstance(agentic_text, bool):
            raise ValueError(f"{cls.name}: agentic_text must be a boolean")
        return {"engine": engine, "agentic_text": agentic_text}

    def _request_config(self) -> dict[str, Any]:
        config = deepcopy(CONFIG)
        config["engine"] = self.config["engine"]
        config["engineVersion"] = ENGINE_VERSIONS[self.config["engine"]]
        if self.config["engine"] != "parse_performance":
            del config["blockOptions"]["figures"]
        if self.config["agentic_text"]:
            config["blockOptions"]["text"] = {"agentic": {"enabled": True}}
        return config

    def prepare(self) -> None:
        if error := self.configuration_error():
            raise CredentialError(error)

    def raw_cache_key(self) -> str:
        identity = {
            "api": API_VERSION,
            "config": self._request_config(),
            "workspace": os.environ.get("EXTEND_WORKSPACE_ID", ""),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        engine = self.config["engine"]
        return f"extend {API_VERSION} {engine} {ENGINE_VERSIONS[engine]} {digest}"

    def version(self) -> str:
        return f"{self.raw_cache_key()} adapter 3"

    def configuration_error(self) -> str | None:
        key = os.environ.get("EXTEND_API_KEY", "").strip()
        return "EXTEND_API_KEY is not configured" if not key or key.startswith("op://") else None

    def fetch(self, document_path: Path) -> RawResult:
        key = os.environ.get("EXTEND_API_KEY", "").strip()
        if not key or key.startswith("op://"):
            raise RuntimeError("EXTEND_API_KEY is not configured")
        if document_path.suffix.lower() not in self.formats:
            raise RuntimeError("Extend does not support this file format")
        headers = {"Authorization": f"Bearer {key}", "x-extend-api-version": API_VERSION}
        if workspace := os.environ.get("EXTEND_WORKSPACE_ID", "").strip():
            headers["x-extend-workspace-id"] = workspace
        deadline = self._clock() + TIMEOUT_SECONDS
        with httpx.Client(
            base_url=API_URL, headers=headers, transport=self._transport, follow_redirects=False
        ) as client:
            with document_path.open("rb") as stream:
                uploaded = self._request(
                    client,
                    "POST",
                    "/files/upload",
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
            file_id = _identifier(uploaded.get("id"), "file")
            result = self._request(
                client,
                "POST",
                "/parse_runs",
                deadline,
                json={"file": {"id": file_id}, "config": self._request_config()},
            )
            run_id = _identifier(result.get("id"), "pr")
            while result.get("status") in {"PENDING", "PROCESSING"}:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise RuntimeError("Extend parsing timed out after 900 seconds")
                self._sleep(min(2.0, remaining))
                result = self._request(client, "GET", f"/parse_runs/{run_id}", deadline)
            if result.get("status") != "PROCESSED":
                raise RuntimeError("Extend parsing failed; check the run in Extend Studio")
            if not isinstance(result.get("output"), dict):
                url = result.get("outputUrl")
                parsed = urlsplit(url) if isinstance(url, str) else None
                if (
                    not parsed
                    or parsed.scheme != "https"
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                ):
                    raise RuntimeError("Extend returned no usable parse output")
                with httpx.Client(transport=self._transport, follow_redirects=False) as download:
                    result["output"] = self._request(download, "GET", url, deadline)
                result["outputUrl"] = None
        return RawResult(
            self.name,
            document_path.parent.name,
            result,
            {"version": self.version(), "raw_cache_key": self.raw_cache_key()},
        )

    def _request(self, client, method, path, deadline, **kwargs):
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise RuntimeError("Extend parsing timed out after 900 seconds")
        try:
            response = client.request(method, path, timeout=min(60.0, remaining), **kwargs)
        except httpx.HTTPError:
            raise RuntimeError("Extend request failed or timed out") from None
        if not response.is_success:
            raise RuntimeError(f"Extend request failed (HTTP {response.status_code})")
        try:
            value = response.json()
        except ValueError:
            raise RuntimeError("Extend returned invalid JSON") from None
        if not isinstance(value, dict):
            raise RuntimeError("Extend returned an invalid response")
        return value

    def normalize(self, raw: RawResult) -> Prediction:
        output = raw.payload.get("output")
        if not isinstance(output, dict) or not isinstance(output.get("chunks"), list):
            raise ValueError("Extend raw result has no parsed chunks")
        pages = {p.get("number"): p for p in (output.get("metadata") or {}).get("pages") or []}
        markdown, layout, offsets, unscored = "", [], [], []
        page_numbers = []
        for chunk in output["chunks"]:
            if markdown:
                markdown += "\n\n"
            start = len(markdown)
            offsets.append(start)
            content = chunk.get("content") or ""
            markdown += content
            bounds = (chunk.get("metadata") or {}).get("pageRange") or {}
            page_numbers.append(
                bounds.get("start") if bounds.get("start") == bounds.get("end") else None
            )
            cursor = 0
            for block in chunk.get("blocks") or []:
                text = block.get("content") or ""
                cls = _CLASSES.get(block.get("type"))
                page = (block.get("metadata") or {}).get("page") or {}
                number = page.get("number")
                if cls is None or not isinstance(number, int) or number < 1:
                    if text:
                        unscored.append(markdown_text(text))
                    continue
                if block.get("type") == "key_value" and not markdown_text(text).strip():
                    continue
                if (
                    cls == ElementClass.FIGURE
                    and (block.get("details") or {}).get("figureType") == "chart"
                ):
                    cls = ElementClass.CHART
                position = content.find(text, cursor) if text else -1
                if position >= 0:
                    cursor = position + len(text)
                cells = None
                if cls == ElementClass.TABLE:
                    cells = [
                        TableCell(
                            text=markdown_text(child.get("content") or ""),
                            box=_box(child, page, pages.get(number)),
                        )
                        for child in block.get("children") or []
                        if child.get("type") in {"table_cell", "table_head"}
                    ] or None
                layout.append(
                    LayoutItem(
                        id=f"extend-{len(layout)}",
                        cls=cls,
                        page=number - 1,
                        text=markdown_text(text),
                        box=_box(block, page, pages.get(number)),
                        channel=Channel.FURNITURE
                        if cls in {ElementClass.HEADER, ElementClass.FOOTER}
                        else None,
                        cells=cells,
                        md_start=start + position if position >= 0 else None,
                        md_end=start + position + len(text) if position >= 0 else None,
                    )
                )
        complete_pages = page_numbers == list(range(1, len(page_numbers) + 1))
        return Prediction(
            markdown=markdown,
            layout=layout,
            unscored_text=unscored,
            page_offsets=offsets if complete_pages else None,
        )

    def usage(self, raw: RawResult) -> dict[str, float]:
        credits = (raw.payload.get("usage") or {}).get("credits")
        if isinstance(credits, int | float):
            return {"credits": credits}
        pages = (raw.payload.get("metrics") or {}).get("pageCount")
        if not isinstance(pages, int):
            raise ValueError("Extend's response has neither credits nor a page count")
        return {"pages": pages}


def _identifier(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(rf"{prefix}_[A-Za-z0-9_-]+", value):
        raise RuntimeError("Extend returned an invalid resource identifier")
    return value


def _box(block, page, original):
    bounds = block.get("boundingBox")
    if not isinstance(bounds, dict):
        return None
    try:
        left, top, right, bottom = (float(bounds[k]) for k in ("left", "top", "right", "bottom"))
        width, height = float(page["width"]), float(page["height"])
        if width <= 0 or height <= 0 or right <= left or bottom <= top:
            return None
        angle = (original or {}).get("rotationApplied") or 0
        if angle:
            width = float(original["originalPageWidth"]) * float(original["dpi"]) / 72
            height = float(original["originalPageHeight"]) * float(original["dpi"]) / 72
            rotation = angle % 360
            transform = {
                90: lambda x, y: (y, height - x),
                180: lambda x, y: (width - x, height - y),
                270: lambda x, y: (width - y, x),
            }.get(rotation)
            if transform is None or width <= 0 or height <= 0:
                return None
            points = [
                transform(x, y)
                for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]
            ]
            left, right = min(p[0] for p in points), max(p[0] for p in points)
            top, bottom = min(p[1] for p in points), max(p[1] for p in points)
        return clip_box(left, top, right - left, bottom - top, width, height)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
