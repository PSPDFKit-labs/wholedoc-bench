import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wholedoc_bench.corpus.release import DOCUMENT_INDEX

RELEASE_FIXTURE = Path(__file__).parent / "fixtures" / "release"


def copy_release(destination: Path) -> Path:
    return Path(shutil.copytree(RELEASE_FIXTURE, destination))


def resum(release: Path) -> None:
    paths = sorted(
        p.relative_to(release).as_posix()
        for p in release.rglob("*")
        if p.is_file() and p.name not in ("SHA256SUMS", ".gitattributes")
    )
    lines = [f"{hashlib.sha256((release / p).read_bytes()).hexdigest()}  {p}\n" for p in paths]
    (release / "SHA256SUMS").write_text("".join(lines), newline="\n")


def rows(release: Path) -> list[dict[str, Any]]:
    text = (release / DOCUMENT_INDEX).read_text("utf-8")
    return [json.loads(line) for line in text.split("\n") if line]


def edit_rows(release: Path, change: Callable[[list[dict[str, Any]]], object]) -> None:
    found = rows(release)
    change(found)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in found)
    (release / DOCUMENT_INDEX).write_text(text, "utf-8", newline="\n")
    resum(release)


def edit_key(release: Path, change: Callable[[dict[str, Any]], None], index: int = 0) -> None:
    path = release / rows(release)[index]["groundtruth_path"]
    truth = json.loads(path.read_text("utf-8"))
    change(truth)
    data = (json.dumps(truth, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    edit_rows(release, lambda found: found[index].update(groundtruth_sha256=sha))


def edit_document(release: Path, data: bytes, index: int = 0) -> None:
    (release / rows(release)[index]["document_path"]).write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    edit_rows(release, lambda found: found[index].update(document_sha256=sha))
    edit_key(release, lambda truth: truth["document"].update(sha256=sha), index)
