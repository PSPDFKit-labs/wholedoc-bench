import importlib.util
import io
from collections.abc import Iterator
from pathlib import Path


def require_pdf_reader(extra: str) -> None:
    if importlib.util.find_spec("pypdf") is None:
        raise ValueError(f"{extra} needs pypdf: run `uv sync --extra {extra}`")


def pdf_pages(document_path: Path) -> Iterator[bytes]:
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject

    reader = PdfReader(document_path)
    root = reader.root_object
    layers = root["/OCProperties"] if "/OCProperties" in root else None
    for index in range(len(reader.pages)):
        writer = PdfWriter()
        writer.append(reader, pages=[index], import_outline=False)
        if layers is not None:
            writer.root_object[NameObject("/OCProperties")] = layers.clone(writer)
        buffer = io.BytesIO()
        writer.write(buffer)
        yield buffer.getvalue()
