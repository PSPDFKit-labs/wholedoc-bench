import json
from pathlib import Path

import pytest

from tests.release_helpers import copy_release
from wholedoc_bench.cli import RUN_MANIFEST, main
from wholedoc_bench.provenance import ManifestRun, file_hash
from wholedoc_bench.providers import ExtendProvider, RawResult, ReductoProvider
from wholedoc_bench.providers.base import load_raw, raw_path

PRESETS = sorted(
    path
    for provider in ("extend", "reducto")
    for path in (Path(__file__).parents[1] / "configs" / provider).glob("*.json")
)


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_mode_release_provenance_and_offline_replay(preset, tmp_path, monkeypatch):
    name = preset.parent.name
    config = json.loads(preset.read_text())
    corpus = copy_release(tmp_path / "release")
    out = tmp_path / "out"
    calls = []
    provider_type = ExtendProvider if name == "extend" else ReductoProvider

    def fetch(self, path):
        calls.append((path.parent.name, self.config))
        field = "output" if name == "extend" else "result"
        return RawResult(
            name,
            path.parent.name,
            {field: {"chunks": [{"content": "Hello", "blocks": []}]}},
            {"version": self.version(), "raw_cache_key": self.raw_cache_key()},
        )

    monkeypatch.setenv(f"{name.upper()}_API_KEY", "fixture-key")
    monkeypatch.setattr(provider_type, "fetch", fetch)
    args = ["run", "--provider", name, "--corpus", str(corpus), "--out", str(out)]
    assert main([*args, "--config", str(preset)]) == 0
    run = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    assert run.config == config
    assert calls == [(id_, config) for id_ in sorted(run.sources)]
    hashes = {}
    for id_, attempt in run.attempts.items():
        saved = raw_path(out, id_)
        raw = load_raw(saved)
        assert attempt.status == "complete" and attempt.response == "new"
        assert attempt.raw_sha256 == file_hash(saved)
        assert raw.metadata["config"] == config
        assert raw.metadata["document_sha256"] == run.sources[id_]
        assert raw.metadata["raw_cache_key"] == provider_type(config).raw_cache_key()
        hashes[id_] = attempt.raw_sha256

    monkeypatch.delenv(f"{name.upper()}_API_KEY")
    assert main([*args, "--from-raw"]) == 0
    replay = ManifestRun.model_validate_json((out / RUN_MANIFEST).read_bytes())
    assert replay.config == config
    assert len(calls) == 2
    assert {a.response for a in replay.attempts.values()} == {"replayed"}
    assert {id_: a.raw_sha256 for id_, a in replay.attempts.items()} == hashes

    other = next(p for p in PRESETS if p.parent.name == name and p != preset)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert main([*args, "--from-raw", "--config", str(other)]) == 2
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
