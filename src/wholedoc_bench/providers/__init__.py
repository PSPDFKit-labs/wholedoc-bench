from wholedoc_bench.providers.anthropic import AnthropicProvider
from wholedoc_bench.providers.azure_di import AzureDIProvider
from wholedoc_bench.providers.base import Provider, RawResult, clip_box
from wholedoc_bench.providers.credentials import CredentialError, require_env
from wholedoc_bench.providers.databricks import DatabricksProvider
from wholedoc_bench.providers.extend import ExtendProvider
from wholedoc_bench.providers.llamaparse import LlamaParseProvider
from wholedoc_bench.providers.local import (
    DoclingProvider,
    MarkitdownProvider,
    PdftotextProvider,
    PymupdfProvider,
)
from wholedoc_bench.providers.nutrient import NutrientProvider
from wholedoc_bench.providers.openai import OpenAIProvider
from wholedoc_bench.providers.reducto import ReductoProvider
from wholedoc_bench.providers.textract import TextractProvider

PROVIDERS: dict[str, type[Provider]] = {
    p.name: p
    for p in (
        NutrientProvider,
        ReductoProvider,
        ExtendProvider,
        TextractProvider,
        AzureDIProvider,
        AnthropicProvider,
        OpenAIProvider,
        LlamaParseProvider,
        DatabricksProvider,
        PdftotextProvider,
        PymupdfProvider,
        MarkitdownProvider,
        DoclingProvider,
    )
}

__all__ = [
    "PROVIDERS",
    "AnthropicProvider",
    "AzureDIProvider",
    "CredentialError",
    "DatabricksProvider",
    "DoclingProvider",
    "LlamaParseProvider",
    "MarkitdownProvider",
    "NutrientProvider",
    "ReductoProvider",
    "ExtendProvider",
    "OpenAIProvider",
    "PdftotextProvider",
    "Provider",
    "PymupdfProvider",
    "RawResult",
    "TextractProvider",
    "clip_box",
    "require_env",
]
