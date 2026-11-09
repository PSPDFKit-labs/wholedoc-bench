import base64
import json
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from wholedoc_bench.cli import main
from wholedoc_bench.execution import RUN_MANIFEST
from wholedoc_bench.providers import PROVIDERS, CredentialError, DatabricksProvider, RawResult
from wholedoc_bench.providers.base import load_raw, save_raw
from wholedoc_bench.providers.databricks import _image_size
from wholedoc_bench.schema import Channel, ChartKind, ChartPoint, ElementClass, RelationKind
from wholedoc_bench.scorer.utils.blocks import markdown_text

VOLUME = "/Volumes/main/bench/scratch"
CHART = {
    "title": "Quarterly revenue",
    "values": {"Revenue": {"Q1": 12, "Q2": 15}},
    "additional_info": "None",
}
TABLE = (
    "<table><tr><th>Region</th><th>Revenue</th></tr>"
    "<tr><td>North</td><td>42</td></tr><tr><td>Metro</td><td>51</td></tr></table>"
)
CHART_TABLE = (
    "<table><caption>Quarterly revenue</caption><thead><tr><th></th><th>Revenue</th></tr></thead>"
    "<tbody><tr><td>Q1</td><td>12</td></tr><tr><td>Q2</td><td>15</td></tr></tbody></table>"
)


def element(id_: int, type_: str, content: Any, *regions: tuple[int, list[int]], **extra: Any):
    bbox = [{"coord": coord, "page_id": page} for page, coord in regions]
    return {"id": id_, "type": type_, "content": content, "bbox": bbox, **extra}


ELEMENTS = [
    element(0, "page_header", "Annual report 2026", (0, [80, 20, 720, 60])),
    element(1, "title", "Annual report", (0, [80, 100, 720, 160])),
    element(
        2,
        "text",
        "Revenue rose 12% in Q2,\nled by the North.",
        (0, [80, 200, 720, 300]),
        confidence=0.98,
    ),
    element(
        3,
        "figure",
        json.dumps(CHART),
        (0, [80, 320, 720, 800]),
        description="A bar chart showing quarterly revenue rising from Q1 to Q2.",
    ),
    element(4, "caption", "Figure 1: Quarterly revenue", (0, [80, 810, 720, 850])),
    element(5, "page_number", "1", (0, [380, 950, 420, 980])),
    element(6, "section_header", "Details", (1, [80, 50, 720, 100])),
    element(7, "table", TABLE, (1, [80, 700, 720, 980]), (2, [100, 40, 900, 200])),
    element(8, "figure", None, (2, [100, 300, 900, 700]), description="A photo of the office."),
    element(9, "footnote", "* Unaudited.", (2, [100, 720, 900, 760])),
]
SIZES = [(800, 1000), (800, 1000), (1000, 800)]


def parsed(folder: str, elements: list[dict] | None = None) -> dict[str, Any]:
    pages = [{"id": n, "image_uri": f"{folder}/pages/{n}.png"} for n in range(len(SIZES))]
    return {
        "document": {
            "pages": pages,
            "elements": deepcopy(ELEMENTS if elements is None else elements),
        },
        "metadata": {"id": "parse-1", "version": "2.0"},
    }


def saved(elements: list[dict] | None = None) -> dict[str, Any]:
    payload = parsed("/unused", elements)
    for page, (width, height) in zip(payload["document"]["pages"], SIZES, strict=True):
        del page["image_uri"]
        page.update(image_width=width, image_height=height)
    return payload


def raw(payload: dict[str, Any]) -> RawResult:
    return RawResult("databricks", "test", payload)


def png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", width, height) + b"\x08\x02"


class Workspace:
    def __init__(self, *, parse: list[httpx.Response] | None = None, response=None) -> None:
        self.requests: list[httpx.Request] = []
        self.folder: str | None = None
        self.folders: set[str] = set()
        self.files: set[str] = set()
        self.parse = parse or []
        self.response = response or parsed
        self.put_status = 204
        self.delete_status = 204
        self.refused: set[str] = set()
        self.listed = lambda path: path
        self.extra: list[dict] = []
        self.on_parse = lambda: None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/oidc/v1/token":
            return httpx.Response(200, json={"access_token": "oauth-token", "expires_in": 3600})
        if path == "/api/2.0/ai-functions/ai-parse-document":
            if self.parse:
                return self.parse.pop(0)
            self.on_parse()
            folder = json.loads(request.content)["options"]["image_output_path"]
            assert folder == self.folder
            self.files = {f"{folder}/pages/{n}.png" for n in range(len(SIZES))}
            self.files.add(f"{folder}/elements/3.png")
            return httpx.Response(200, json=self.response(folder))
        kind, _, target = path.removeprefix("/api/2.0/fs/").partition("/")
        target = "/" + target
        if request.method == "GET" and kind == "files":
            assert request.headers["range"] == "bytes=0-65535"
            return httpx.Response(206, content=png(*SIZES[int(Path(target).stem)]))
        if request.method == "GET" and kind == "directories":
            if target not in self.folders and not any(
                f.startswith(target + "/") for f in self.files
            ):
                return httpx.Response(404, json={"error_code": "NOT_FOUND"})
            children = {
                f"{target}/{f.removeprefix(target + '/').split('/')[0]}"
                for f in self.files
                if f.startswith(target + "/")
            }
            contents = [
                {
                    "path": self.listed(child),
                    "name": child.rsplit("/", 1)[1],
                    "is_directory": child not in self.files,
                }
                for child in sorted(children)
            ]
            return httpx.Response(200, json={"contents": contents + self.extra})
        if request.method == "DELETE":
            if target in self.refused:
                return httpx.Response(403, json={"error_code": "PERMISSION_DENIED"})
            if kind == "directories" and any(f.startswith(target + "/") for f in self.files):
                return httpx.Response(400, json={"error_code": "DIRECTORY_NOT_EMPTY"})
            if self.delete_status == 204:
                self.files.discard(target)
                self.folders.discard(target)
            return httpx.Response(self.delete_status)
        if request.method == "PUT":
            if "wholedoc-bench-check-" not in target:
                self.folder = target
            if self.put_status == 204:
                self.folders.add(target)
            return httpx.Response(self.put_status)
        return httpx.Response(200)

    @property
    def exists(self) -> bool:
        return self.folder in self.folders

    def calls(self) -> list[str]:
        folder = self.folder or "<none>"
        return [f"{r.method} {r.url.path.replace(folder, '<folder>')}" for r in self.requests]


@pytest.fixture
def document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc-1234.cloud.databricks.com/")
    monkeypatch.setenv("DATABRICKS_TOKEN", " dapi-secret-do-not-store\n")
    monkeypatch.setenv("DATABRICKS_VOLUME_PATH", VOLUME + "/")
    path = tmp_path / "test" / "document.pdf"
    path.parent.mkdir()
    path.write_bytes(b"%PDF-1.7 test")
    return path


def test_fetch_parses_reads_the_page_sizes_and_deletes_the_images(
    document: Path, tmp_path: Path
) -> None:
    workspace = Workspace()
    result = DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert workspace.calls() == [
        "PUT /api/2.0/fs/directories<folder>",
        "POST /api/2.0/ai-functions/ai-parse-document",
        "GET /api/2.0/fs/files<folder>/pages/0.png",
        "GET /api/2.0/fs/files<folder>/pages/1.png",
        "GET /api/2.0/fs/files<folder>/pages/2.png",
        "GET /api/2.0/fs/directories<folder>",
        "GET /api/2.0/fs/directories<folder>/elements",
        "DELETE /api/2.0/fs/files<folder>/elements/3.png",
        "DELETE /api/2.0/fs/directories<folder>/elements",
        "GET /api/2.0/fs/directories<folder>/pages",
        "DELETE /api/2.0/fs/files<folder>/pages/0.png",
        "DELETE /api/2.0/fs/files<folder>/pages/1.png",
        "DELETE /api/2.0/fs/files<folder>/pages/2.png",
        "DELETE /api/2.0/fs/directories<folder>/pages",
        "DELETE /api/2.0/fs/directories<folder>",
    ]
    assert workspace.folder is not None and workspace.folder.startswith(VOLUME + "/wholedoc-bench-")
    assert all(r.url.host == "dbc-1234.cloud.databricks.com" for r in workspace.requests)
    assert all(
        r.headers["authorization"] == "Bearer dapi-secret-do-not-store" for r in workspace.requests
    )
    body = json.loads(workspace.requests[1].content)
    assert body == {
        "content": "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.7 test").decode(),
        "options": {
            "version": "2.0",
            "description_element_types": "*",
            "image_output_path": workspace.folder,
        },
    }
    assert workspace.files == set()
    assert result.payload == saved()
    assert result.metadata == {"version": "databricks ai_parse_document REST 2.0 descriptions=*"}
    path = tmp_path / "raw.json"
    save_raw(result, path)
    assert "dapi-secret" not in path.read_text() and VOLUME not in path.read_text()
    assert load_raw(path) == result


def test_a_service_principal_signs_in_with_oauth(document: Path, monkeypatch) -> None:
    monkeypatch.delenv("DATABRICKS_TOKEN")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client-1")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "oauth-secret")
    workspace = Workspace()
    DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    sign_in, *calls = workspace.requests
    assert sign_in.url.path == "/oidc/v1/token"
    assert (
        sign_in.headers["authorization"]
        == "Basic " + base64.b64encode(b"client-1:oauth-secret").decode()
    )
    assert sign_in.content == b"grant_type=client_credentials&scope=all-apis"
    assert all(r.headers["authorization"] == "Bearer oauth-token" for r in calls)


def test_prepare_checks_the_credentials_and_writing_to_the_volume(document: Path) -> None:
    workspace = Workspace()
    DatabricksProvider(transport=httpx.MockTransport(workspace)).prepare()
    head, make, listing, remove = (f"{r.method} {r.url.path}" for r in workspace.requests)
    assert head == "HEAD /api/2.0/fs/directories" + VOLUME
    assert make.startswith(f"PUT /api/2.0/fs/directories{VOLUME}/wholedoc-bench-check-")
    assert (listing, remove) == (make.replace("PUT", "GET"), make.replace("PUT", "DELETE"))
    assert workspace.folders == set()


def test_prepare_signs_a_service_principal_in(document: Path, monkeypatch) -> None:
    monkeypatch.delenv("DATABRICKS_TOKEN")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client-1")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "oauth-secret")
    workspace = Workspace()
    DatabricksProvider(transport=httpx.MockTransport(workspace)).prepare()
    assert [r.url.path for r in workspace.requests][:2] == [
        "/oidc/v1/token",
        "/api/2.0/fs/directories" + VOLUME,
    ]
    refused = httpx.MockTransport(
        lambda _: httpx.Response(401, json={"error_code": "invalid_client"})
    )
    with pytest.raises(
        CredentialError, match=r"sign-in failed \(HTTP 401\): check the credentials"
    ):
        DatabricksProvider(transport=refused).prepare()
    nothing = httpx.MockTransport(lambda _: httpx.Response(200, json={"token_type": "Bearer"}))
    with pytest.raises(RuntimeError, match="sign-in returned no access token"):
        DatabricksProvider(transport=nothing).fetch(document)


@pytest.mark.parametrize(
    ("status", "error", "message"),
    [
        (403, CredentialError, "HTTP 403, PERMISSION_DENIED): check the credentials"),
        (400, CredentialError, "HTTP 400, PERMISSION_DENIED): check the credentials"),
        (404, CredentialError, "check that DATABRICKS_HOST is the workspace"),
        (503, ValueError, "HTTP 503, PERMISSION_DENIED); try again later"),
    ],
)
def test_prepare_says_what_to_fix(document: Path, status: int, error: type, message: str) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        body = {"error_code": "PERMISSION_DENIED", "message": "token dapi-secret-do-not-store"}
        return httpx.Response(status, json=body)

    provider = DatabricksProvider(transport=httpx.MockTransport(refuse), sleep=lambda _: None)
    with pytest.raises(error) as raised:
        provider.prepare()
    assert message in str(raised.value)
    assert "dapi-secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DATABRICKS_HOST", ""),
        ("DATABRICKS_HOST", "http://dbc-1234.cloud.databricks.com"),
        ("DATABRICKS_HOST", "https://dbc-1234.cloud.databricks.com/api"),
        ("DATABRICKS_HOST", "https://user:pass@dbc-1234.cloud.databricks.com"),
        ("DATABRICKS_VOLUME_PATH", ""),
        ("DATABRICKS_VOLUME_PATH", "/Volumes/main/bench"),
        ("DATABRICKS_VOLUME_PATH", "/Volumes/main/bench/scratch/../other"),
        ("DATABRICKS_VOLUME_PATH", "dbfs:/tmp/scratch"),
        ("DATABRICKS_TOKEN", ""),
        ("DATABRICKS_TOKEN", "dapi-sécret"),
        ("DATABRICKS_CLIENT_ID", "client-1"),
    ],
)
def test_missing_or_malformed_settings_stop_before_any_call(
    document: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    workspace = Workspace()
    provider = DatabricksProvider(transport=httpx.MockTransport(workspace))
    with pytest.raises(CredentialError) as raised:
        provider.prepare()
    with pytest.raises(CredentialError):
        provider.fetch(document)
    assert workspace.requests == []
    assert "secret" not in str(raised.value)


def test_a_host_without_a_scheme_is_https(document: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", "adb-1234.5.azuredatabricks.net")
    workspace = Workspace()
    DatabricksProvider(transport=httpx.MockTransport(workspace)).prepare()
    assert str(workspace.requests[0].url).startswith("https://adb-1234.5.azuredatabricks.net/")


def test_the_parse_is_tried_again_after_a_rate_limit_only(document: Path) -> None:
    limited = httpx.Response(429, headers={"Retry-After": "3"})
    workspace = Workspace(parse=[limited])
    sleeps = []
    DatabricksProvider(transport=httpx.MockTransport(workspace), sleep=sleeps.append).fetch(
        document
    )
    assert sleeps == [3.0]
    failed = httpx.Response(500, json={"error_code": "INTERNAL_ERROR", "message": "dapi-secret"})
    workspace = Workspace(parse=[failed])
    with pytest.raises(
        RuntimeError, match=r"^Databricks parse failed \(HTTP 500, INTERNAL_ERROR\)$"
    ):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert workspace.calls()[-1] == "DELETE /api/2.0/fs/directories<folder>"
    assert [c for c in workspace.calls() if "parse" in c] == [
        "POST /api/2.0/ai-functions/ai-parse-document"
    ]


def test_a_dropped_parse_is_not_sent_again(document: Path) -> None:
    workspace = Workspace()

    def drop(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("ai-parse-document"):
            workspace.requests.append(request)
            raise httpx.ReadTimeout("timed out")
        return workspace(request)

    with pytest.raises(RuntimeError, match="parse request failed or timed out"):
        DatabricksProvider(transport=httpx.MockTransport(drop)).fetch(document)
    assert sum(r.url.path.endswith("ai-parse-document") for r in workspace.requests) == 1


def test_failed_pages_fail_the_document(document: Path) -> None:
    def partial(folder: str) -> dict[str, Any]:
        return {**parsed(folder), "error_status": [{"error_message": "bad page", "page_id": 1}]}

    workspace = Workspace(response=partial)
    with pytest.raises(RuntimeError, match="failed to parse some pages"):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert workspace.files == set()
    with pytest.raises(ValueError, match="failed to parse some pages"):
        DatabricksProvider().normalize(raw({**saved(), "error_status": [{"page_id": 1}]}))


def test_a_page_image_outside_the_folder_is_not_read(document: Path) -> None:
    def elsewhere(folder: str) -> dict[str, Any]:
        payload = parsed(folder)
        payload["document"]["pages"][0]["image_uri"] = "/Volumes/main/other/secret.png"
        return payload

    workspace = Workspace(response=elsewhere)
    with pytest.raises(RuntimeError, match="no page image in the image folder"):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert not any("secret.png" in r.url.path for r in workspace.requests)


def test_a_folder_that_cannot_be_deleted_does_not_fail_the_parse(document: Path, capsys) -> None:
    workspace = Workspace()
    workspace.delete_status = 403
    result = DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert result.payload == saved()
    assert workspace.folder is not None
    warning = capsys.readouterr().err
    assert workspace.folder.rsplit("/", 1)[1] + " may remain" in warning
    assert VOLUME not in warning


def test_one_refused_deletion_does_not_stop_the_others(document: Path, capsys) -> None:
    workspace = Workspace()
    workspace.on_parse = lambda: workspace.refused.add(f"{workspace.folder}/pages/1.png")
    DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert {f.rsplit("/", 2)[1:] == ["pages", "1.png"] for f in workspace.files} == {True}
    assert "may remain" in capsys.readouterr().err


def test_a_failed_folder_creation_stops_before_the_parse(document: Path, capsys) -> None:
    workspace = Workspace()
    workspace.put_status = 403
    with pytest.raises(RuntimeError, match=r"image folder creation failed \(HTTP 403\)"):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert not any(r.url.path.endswith("ai-parse-document") for r in workspace.requests)
    assert capsys.readouterr().err == ""


def test_paths_outside_the_folder_are_neither_read_nor_deleted(document: Path) -> None:
    def climbing(folder: str) -> dict[str, Any]:
        payload = parsed(folder)
        payload["document"]["pages"][0]["image_uri"] = f"{folder}/../../other/secret.png"
        return payload

    workspace = Workspace(response=climbing)
    workspace.extra = [
        {"name": "..", "path": "..", "is_directory": True},
        {"name": "../keep.csv", "path": "keep.csv", "is_directory": False},
    ]
    with pytest.raises(RuntimeError, match="no page image in the image folder"):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    folder = workspace.folder
    assert folder is not None
    files_api = [r.url.path for r in workspace.requests if r.url.path.startswith("/api/2.0/fs/")]
    assert all(folder in path and ".." not in path for path in files_api)
    assert not any("keep.csv" in path for path in files_api)


def test_image_paths_are_read_by_the_folder_name(document: Path, capsys) -> None:
    def elsewhere(path: str) -> str:
        return "dbfs:" + path.replace(VOLUME, VOLUME.upper())

    def written(folder: str) -> dict[str, Any]:
        payload = parsed(folder)
        for page in payload["document"]["pages"]:
            page["image_uri"] = elsewhere(page["image_uri"])
        return payload

    workspace = Workspace(response=written)
    workspace.listed = elsewhere
    result = DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert result.payload == saved()
    assert workspace.files == set() and not workspace.exists
    assert capsys.readouterr().err == ""


def test_an_image_path_anywhere_else_in_the_response_fails_the_document(document: Path) -> None:
    def echoing(folder: str) -> dict[str, Any]:
        payload = parsed(folder)
        payload["metadata"]["image_output_path"] = folder
        return payload

    workspace = Workspace(response=echoing)
    with pytest.raises(RuntimeError, match="image path where the adapter reads none"):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert not workspace.exists


def test_the_image_reads_have_a_budget_of_their_own(document: Path) -> None:
    now = [0.0]
    workspace = Workspace()
    workspace.on_parse = lambda: now.__setitem__(0, 899.0)
    provider = DatabricksProvider(transport=httpx.MockTransport(workspace), clock=lambda: now[0])
    assert provider.fetch(document).payload == saved()
    assert workspace.files == set()


def test_a_parse_that_never_reached_databricks_is_sent_again(document: Path) -> None:
    workspace = Workspace()
    refused = []

    def connect(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("ai-parse-document") and not refused:
            refused.append(request)
            raise httpx.ConnectError("connection refused")
        return workspace(request)

    sleeps = []
    provider = DatabricksProvider(transport=httpx.MockTransport(connect), sleep=sleeps.append)
    assert provider.fetch(document).payload == saved()
    assert sleeps == [2]


def test_a_redirect_is_not_followed(document: Path) -> None:
    moved = httpx.Response(307, headers={"Location": "https://elsewhere.example/parse"})
    workspace = Workspace(parse=[moved])
    with pytest.raises(RuntimeError, match=r"parse failed \(HTTP 307\)"):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(document)
    assert {r.url.host for r in workspace.requests} == {"dbc-1234.cloud.databricks.com"}


def test_a_format_the_api_does_not_read_is_not_sent(document: Path) -> None:
    sheet = document.with_suffix(".xlsx")
    sheet.write_bytes(b"PK")
    workspace = Workspace()
    with pytest.raises(RuntimeError, match="does not read this file format"):
        DatabricksProvider(transport=httpx.MockTransport(workspace)).fetch(sheet)
    assert workspace.requests == []


def test_image_size_reads_png_and_jpeg_headers() -> None:
    assert _image_size(png(1700, 2200)) == (1700, 2200)
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + bytes(9)
    sof0 = b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 2200, 1700) + bytes(10)
    assert _image_size(b"\xff\xd8" + app0 + sof0) == (1700, 2200)
    sof2 = b"\xff\xc2" + struct.pack(">HBHH", 17, 8, 600, 800) + bytes(10)
    assert _image_size(b"\xff\xd8" + sof2) == (800, 600)
    for head in (b"GIF89a" + bytes(20), png(1700, 2200)[:20], b"\xff\xd8" + app0):
        with pytest.raises(RuntimeError, match="page image of unknown size"):
            _image_size(head)


def test_normalize_writes_markdown_and_layout_from_the_elements() -> None:
    prediction = DatabricksProvider().normalize(raw(saved()))
    blocks = [
        "# Annual report",
        "Revenue rose 12% in Q2,\nled by the North.",
        CHART_TABLE,
        "Figure 1: Quarterly revenue",
        "## Details",
        TABLE,
        "\\* Unaudited.",
    ]
    assert prediction.markdown == "\n\n".join(blocks)
    md = prediction.markdown
    assert prediction.page_offsets == [0, md.index("## Details"), md.index("\\* Unaudited.")]
    items = prediction.layout
    assert [(i.cls, i.page, i.channel) for i in items] == [
        (ElementClass.HEADER, 0, Channel.FURNITURE),
        (ElementClass.HEADING, 0, None),
        (ElementClass.PARAGRAPH, 0, None),
        (ElementClass.CHART, 0, None),
        (ElementClass.CAPTION, 0, None),
        (ElementClass.FOOTER, 0, Channel.FURNITURE),
        (ElementClass.HEADING, 1, None),
        (ElementClass.TABLE, 1, None),
        (ElementClass.TABLE, 2, None),
        (ElementClass.FIGURE, 2, None),
        (ElementClass.FOOTNOTE, 2, None),
    ]
    assert [i.text for i in items[:3]] == [
        "Annual report 2026",
        "Annual report",
        "Revenue rose 12% in Q2,\nled by the North.",
    ]
    for item in items:
        if item.md_start is not None:
            assert md[item.md_start : item.md_end] in blocks
    assert [i.md_start is None for i in items] == [
        True, False, False, False, False, True, False, False, True, True, False
    ]  # fmt: skip
    assert items[0].box == pytest.approx((0.1, 0.02, 0.8, 0.04))
    assert items[8].box == pytest.approx((0.1, 0.05, 0.8, 0.2))
    assert items[8].text == ""
    assert [(r.kind, r.source, r.target) for r in prediction.relations] == [
        (RelationKind.CONTINUES_IN, items[7].id, items[8].id)
    ]
    assert prediction.unscored_text == []


def test_a_chart_has_its_data_title_and_the_type_its_description_names() -> None:
    chart = DatabricksProvider().normalize(raw(saved())).layout[3].chart
    assert chart is not None
    assert chart.kind == ChartKind.BAR
    assert chart.title == "Quarterly revenue"
    assert [(s.name, [(p.label, p.value) for p in s.points]) for s in chart.series] == [
        ("Revenue", [("Q1", 12.0), ("Q2", 15.0)])
    ]


@pytest.mark.parametrize(
    ("content", "description", "cls", "series", "title"),
    [
        (
            {"title": "None", "values": {"Share": {"A": "12%", "B": "$1.5", "C": "n/a"}}},
            "A pie chart of shares.",
            ElementClass.CHART,
            [("Share", [("A", 12.0), ("B", 1.5)])],
            None,
        ),
        (
            {"values": {"2025": 3, "2026": 4}},
            None,
            ElementClass.CHART,
            [("", [("2025", 3.0), ("2026", 4.0)])],
            None,
        ),
        ("Sales 2026", "A line graph of monthly sales.", ElementClass.FIGURE, None, None),
        ("Our office", "A photo of the office.", ElementClass.FIGURE, None, None),
        ('{"not": "chart data"}', None, ElementClass.FIGURE, None, None),
    ],
)
def test_figure_content_decides_whether_it_is_a_chart(content, description, cls, series, title):
    content = json.dumps(content) if isinstance(content, dict) else content
    figure = element(0, "figure", content, (0, [80, 320, 720, 800]), description=description)
    item = DatabricksProvider().normalize(raw(saved([figure]))).layout[0]
    assert item.cls == cls
    if series is None:
        assert item.chart is None
    else:
        assert item.chart is not None
        assert [
            (s.name, [(p.label, p.value) for p in s.points]) for s in item.chart.series
        ] == series
        assert item.chart.title == title


def test_a_chart_without_values_writes_no_json_into_the_markdown() -> None:
    figure = element(
        0, "figure", json.dumps({"title": "Empty", "values": {}}), (0, [80, 320, 720, 800])
    )
    prediction = DatabricksProvider().normalize(raw(saved([figure])))
    assert prediction.markdown == ""
    assert (prediction.layout[0].cls, prediction.layout[0].text) == (ElementClass.CHART, "")


@pytest.mark.parametrize(
    "values", [{"Revenue": [12, 15]}, {"A": {"Q1": 1}, "B": 2}, [1, 2], {"A": {"Q1": [1]}}]
)
def test_chart_data_of_another_shape_fails_the_document(values) -> None:
    figure = element(0, "figure", json.dumps({"values": values}), (0, [80, 320, 720, 800]))
    with pytest.raises(ValueError, match="chart data of an unknown shape"):
        DatabricksProvider().normalize(raw(saved([figure])))


def test_a_page_number_in_the_top_half_is_a_header() -> None:
    number = element(0, "page_number", "iv", (0, [380, 20, 420, 60]))
    item = DatabricksProvider().normalize(raw(saved([number]))).layout[0]
    assert (item.cls, item.channel) == (ElementClass.HEADER, Channel.FURNITURE)


def test_an_element_without_a_box_keeps_its_text_in_the_markdown() -> None:
    text = {"id": 0, "type": "text", "content": "Floating"}
    prediction = DatabricksProvider().normalize(raw(saved([text])))
    assert (prediction.markdown, prediction.layout) == ("Floating", [])


def test_out_of_order_pages_give_no_page_offsets() -> None:
    elements = [
        element(0, "text", "B", (1, [0, 0, 10, 10])),
        element(1, "text", "A", (0, [0, 0, 10, 10])),
    ]
    assert DatabricksProvider().normalize(raw(saved(elements))).page_offsets is None


def test_an_unknown_element_type_fails_the_document() -> None:
    with pytest.raises(ValueError, match="unknown element type: 'checkbox'"):
        DatabricksProvider().normalize(
            raw(saved([element(0, "checkbox", "[x]", (0, [0, 0, 9, 9]))]))
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["document"]["pages"][1].pop("image_width"),
        lambda p: p["document"]["pages"][1].update(id=0),
        lambda p: p["document"]["pages"][2].update(image_height=0),
        lambda p: p["document"].pop("pages"),
    ],
)
def test_pages_without_a_valid_image_size_fail_normalization(change) -> None:
    payload = saved()
    change(payload)
    with pytest.raises(ValueError):
        DatabricksProvider().normalize(raw(payload))


def test_usage_counts_the_pages_parsed() -> None:
    assert DatabricksProvider().usage(raw(saved())) == {"pages": 3}


def test_formats_are_those_the_api_reads() -> None:
    assert ".xlsx" not in DatabricksProvider.formats
    assert {".pdf", ".docx", ".pptx", ".png", ".jpg", ".tiff"} <= DatabricksProvider.formats


def test_cli_run_and_replay_keep_no_credential_or_volume_path(
    document: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_cli import _corpus

    workspace = Workspace()

    class MockProvider(DatabricksProvider):
        def __init__(self, config=None):
            super().__init__(config, transport=httpx.MockTransport(workspace))

    monkeypatch.setitem(PROVIDERS, "databricks", MockProvider)
    corpus = _corpus(tmp_path, ".pdf")
    out = tmp_path / "out"
    args = ["run", "--provider", "databricks", "--corpus", str(corpus), "--out", str(out)]
    assert main(args) == 0
    assert json.loads((out / RUN_MANIFEST).read_text())["config"] == {}
    assert (out / "plain_continuation_demo.json").exists()
    for path in (p for p in out.rglob("*") if p.is_file()):
        assert b"dapi-secret" not in path.read_bytes() and VOLUME.encode() not in path.read_bytes()
    calls = len(workspace.requests)
    for name in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_VOLUME_PATH"):
        monkeypatch.delenv(name)
    prediction = out / "plain_continuation_demo.json"
    written = prediction.read_bytes()
    prediction.unlink()
    assert main([*args, "--from-raw"]) == 0
    assert len(workspace.requests) == calls
    assert prediction.read_bytes() == written


def normalized(*elements: dict[str, Any]):
    return DatabricksProvider().normalize(raw(saved(list(elements))))


def test_regions_on_one_page_are_enclosed_and_other_pages_are_kept_apart() -> None:
    both = element(0, "text", "Two columns", (0, [80, 100, 380, 200]), (0, [420, 150, 720, 300]))
    assert normalized(both).layout[0].box == pytest.approx((0.1, 0.1, 0.8, 0.2))
    apart = element(0, "table", TABLE, (2, [100, 40, 900, 200]), (0, [80, 700, 720, 980]))
    prediction = normalized(apart)
    assert [(i.page, i.text != "") for i in prediction.layout] == [(0, True)]
    assert prediction.relations == []
    stray = element(0, "text", "Stray", (7, [0, 0, 10, 10]))
    assert (normalized(stray).markdown, normalized(stray).layout) == ("Stray", [])


def test_a_running_header_on_every_page_is_an_item_on_each_unlinked() -> None:
    header = element(0, "page_header", "Annual report", *((n, [80, 20, 720, 60]) for n in range(3)))
    prediction = normalized(header)
    assert [(i.page, i.text, i.channel) for i in prediction.layout] == [
        (n, "Annual report", Channel.FURNITURE) for n in range(3)
    ]
    assert (prediction.markdown, prediction.relations) == ("", [])


@pytest.mark.parametrize(
    ("content", "block", "text"),
    [
        ("Key\nresults", "## Key results", "Key results"),
        ("#1 Priority", "## \\#1 Priority", "#1 Priority"),
        ("Item #", "## Item \\#", "Item #"),
        ("*Note*", "## \\*Note\\*", "*Note*"),
    ],
)
def test_a_heading_is_one_line_at_its_level(content: str, block: str, text: str) -> None:
    prediction = normalized(element(0, "section_header", content, (0, [0, 0, 9, 9])))
    assert (prediction.markdown, prediction.layout[0].text) == (block, text)


@pytest.mark.parametrize(
    "content",
    ["* Figures are unaudited.", "1. Not a list\n- nor this", "# Not a heading", "a_b *c* `d`"],
)
def test_plain_text_reads_back_as_printed(content: str) -> None:
    prediction = normalized(element(0, "footnote", content, (0, [0, 0, 9, 9])))
    assert prediction.layout[0].text == markdown_text(prediction.markdown)
    assert " ".join(prediction.layout[0].text.split()) == " ".join(content.split())


def test_elements_are_read_in_the_order_of_their_ids() -> None:
    second = element(1, "text", "Second", (0, [0, 50, 9, 59]))
    first = element(0, "text", "First", (0, [0, 0, 9, 9]))
    assert normalized(second, first).markdown == "First\n\nSecond"


def test_empty_elements_and_their_pages() -> None:
    elements = [
        element(0, "text", "  ", (0, [0, 0, 9, 9])),
        element(1, "signature", None, (0, [100, 900, 300, 950])),
        element(2, "text", "Only page one", (0, [0, 50, 9, 59])),
        element(3, "page_number", "2", (1, [0, 0, 0, 0])),
    ]
    prediction = normalized(*elements)
    assert [(i.cls, i.text) for i in prediction.layout] == [
        (ElementClass.FIGURE, ""),
        (ElementClass.PARAGRAPH, "Only page one"),
        (ElementClass.FOOTER, "2"),
    ]
    assert prediction.page_offsets == [0, len("Only page one"), len("Only page one")]


@pytest.mark.parametrize(
    "elements",
    [
        [{"id": 0, "type": ["text"], "content": "x"}],
        [element(0, "text", "a", (0, [0, 0, 9, 9])), element(0, "text", "b", (0, [0, 0, 9, 9]))],
        [{"type": "text", "content": "no id"}],
    ],
)
def test_elements_without_a_type_or_an_id_of_their_own_fail_the_document(elements) -> None:
    with pytest.raises(ValueError):
        normalized(*elements)


def test_a_live_chart_report_gives_both_charts_their_data_and_type(charts_databricks_raw) -> None:
    prediction = DatabricksProvider().normalize(charts_databricks_raw)
    charts = [i.chart for i in prediction.layout if i.chart is not None]
    assert [(c.kind, c.title) for c in charts] == [
        (ChartKind.BAR, None),
        (ChartKind.LINE, "Energy cost per tonne (EUR)"),
    ]
    assert [[s.name for s in c.series] for c in charts] == [["2025", "2026"], ["Energy"]]
    assert charts[1].series[0].points[0] == ChartPoint(label="Apr", value=11.2)
    md = prediction.markdown
    assert prediction.page_offsets == [0, md.index("## Energy costs")]
    furniture = [i.cls for i in prediction.layout if i.channel == Channel.FURNITURE]
    assert furniture == [ElementClass.HEADER, ElementClass.FOOTER] * 2
    assert "Harbour Energy" not in md


def test_a_live_two_column_report_reads_as_printed(two_column_databricks_raw) -> None:
    prediction = DatabricksProvider().normalize(two_column_databricks_raw)
    md = prediction.markdown
    assert md.startswith("# Annual Review 2026\n\n")
    footnote = next(i for i in prediction.layout if i.cls == ElementClass.FOOTNOTE)
    assert footnote.text == "* Figures are unaudited and rounded to one decimal."
    assert md[footnote.md_start : footnote.md_end].startswith("\\* Figures")
    assert [i.page for i in prediction.layout if i.cls == ElementClass.TABLE] == [1, 2]
    assert prediction.relations == []
    offsets = prediction.page_offsets
    assert offsets is not None and offsets[0] == 0
    assert md[offsets[1] :].startswith("formance of each course")
    assert md[offsets[2] :].startswith("<table>") and "Metro" in md[offsets[2] :][:200]


def test_chart_values_that_are_no_number_are_no_point() -> None:
    values = {"S": {"A": True, "B": 10**400, "C": None, "D": 3}}
    figure = element(0, "figure", json.dumps({"values": values}), (0, [80, 320, 720, 800]))
    chart = normalized(figure).layout[0].chart
    assert chart is not None
    assert [(p.label, p.value) for p in chart.series[0].points] == [("D", 3.0)]


def test_chart_json_without_values_fails_the_document() -> None:
    renamed = {"title": "Sales", "data": {"S": {"Q1": 1}}, "additional_info": "None"}
    figure = element(0, "figure", json.dumps(renamed), (0, [80, 320, 720, 800]))
    with pytest.raises(ValueError, match="chart data of an unknown shape"):
        normalized(figure)


def test_a_probe_folder_that_cannot_be_deleted_is_named(document: Path, capsys) -> None:
    workspace = Workspace()
    workspace.delete_status = 503
    provider = DatabricksProvider(transport=httpx.MockTransport(workspace), sleep=lambda _: None)
    provider.prepare()
    assert "wholedoc-bench-check-" in capsys.readouterr().err
