import json
from pathlib import Path

import pytest

from wholedoc_bench.providers import PROVIDERS

CONFIGS = sorted((Path(__file__).parents[1] / "configs").glob("*/*.json"))


def test_the_benchmark_runs_claude_opus_and_fable_at_medium_and_high_effort() -> None:
    assert [p.name for p in CONFIGS if p.parent.name == "anthropic"] == [
        "fable-5-1-high.json",
        "fable-5-1-medium.json",
        "opus-5-5-high.json",
        "opus-5-5-medium.json",
    ]


def test_the_benchmark_runs_gpt_sol_and_luna_at_medium_and_high_effort() -> None:
    assert [p.name for p in CONFIGS if p.parent.name == "openai"] == [
        "luna-6-high.json",
        "luna-6-medium.json",
        "sol-6-1-high.json",
        "sol-6-1-medium.json",
    ]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_a_config_is_its_providers_effective_config(path: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    assert PROVIDERS[path.parent.name].validate_config(config) == config
