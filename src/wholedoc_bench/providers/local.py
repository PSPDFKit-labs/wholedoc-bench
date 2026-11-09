import importlib.metadata
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

from wholedoc_bench.providers.base import Provider, RawResult
from wholedoc_bench.schema import Prediction

_OFFICE_AND_PDF = frozenset({".pdf", ".docx", ".xlsx"})


class _MarkdownProvider(Provider):
    sends_documents: ClassVar[bool] = False

    def normalize(self, raw: RawResult) -> Prediction:
        return Prediction(markdown=raw.payload)

    def _result(self, document_path: Path, markdown: str, version: str) -> RawResult:
        return RawResult(
            provider=self.name,
            document_id=document_path.parent.name,
            payload=markdown,
            metadata={"version": version},
        )


class PdftotextProvider(_MarkdownProvider):
    name: ClassVar[str] = "pdftotext"

    def fetch(self, document_path: Path) -> RawResult:
        tool = shutil.which("pdftotext")
        if tool is None:
            raise RuntimeError("pdftotext not found; install poppler")
        run = subprocess.run(
            [tool, "-enc", "UTF-8", str(document_path), "-"],
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        markdown = run.stdout.replace("\f", "\n\n").strip() + "\n"
        return self._result(document_path, markdown, _pdftotext_version(tool))


class PymupdfProvider(_MarkdownProvider):
    name: ClassVar[str] = "pymupdf"

    def fetch(self, document_path: Path) -> RawResult:
        import pymupdf4llm  # pyright: ignore[reportMissingImports]

        markdown = pymupdf4llm.to_markdown(str(document_path))
        assert isinstance(markdown, str)
        return self._result(document_path, markdown, _version("pymupdf"))


class MarkitdownProvider(_MarkdownProvider):
    name: ClassVar[str] = "markitdown"
    formats: ClassVar[frozenset[str]] = _OFFICE_AND_PDF

    def fetch(self, document_path: Path) -> RawResult:
        from markitdown import MarkItDown  # pyright: ignore[reportMissingImports]

        markdown = MarkItDown(enable_plugins=False).convert(str(document_path)).text_content
        return self._result(document_path, markdown, _version("markitdown"))


class DoclingProvider(_MarkdownProvider):
    name: ClassVar[str] = "docling"
    formats: ClassVar[frozenset[str]] = _OFFICE_AND_PDF
    _docling: Any = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError("provider config must be a JSON object")
        if unknown := config.keys() - {"chart_extraction"}:
            raise ValueError(f"{cls.name}: unsupported config keys: {', '.join(sorted(unknown))}")
        charts = config.get("chart_extraction", True)
        if not isinstance(charts, bool):
            raise ValueError(f"{cls.name}: chart_extraction must be true or false")
        return {"chart_extraction": charts}

    def fetch(self, document_path: Path) -> RawResult:
        document = self._converter().convert(str(document_path)).document
        markdown = document.export_to_markdown(include_picture_classification=False)
        return self._result(document_path, markdown, _version("docling"))

    def _converter(self) -> Any:
        if self._docling is None:
            from docling.datamodel.base_models import (  # pyright: ignore[reportMissingImports]
                InputFormat,
            )
            from docling.datamodel.pipeline_options import (  # pyright: ignore[reportMissingImports]
                PdfPipelineOptions,
            )
            from docling.document_converter import (  # pyright: ignore[reportMissingImports]
                DocumentConverter,
                PdfFormatOption,
            )

            if self.config["chart_extraction"]:
                options = PdfPipelineOptions(do_chart_extraction=True)
                self._docling = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
                )
            else:
                self._docling = DocumentConverter()
        return self._docling


def _version(package: str) -> str:
    return f"{package} {importlib.metadata.version(package)}"


def _pdftotext_version(tool: str) -> str:
    probe = subprocess.run([tool, "-v"], capture_output=True, encoding="utf-8")
    output = f"{probe.stdout}\n{probe.stderr}".strip()
    return output.splitlines()[0] if output else "pdftotext (version unknown)"
