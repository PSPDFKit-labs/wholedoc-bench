from dataclasses import dataclass

COVERAGE_THRESHOLD = 0.95


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    value: float


def coverage(parsed_ok: int, attempted: int) -> GateResult:
    if attempted <= 0:
        raise ValueError("attempted must be positive")
    value = parsed_ok / attempted
    return GateResult(gate="G3", passed=value >= COVERAGE_THRESHOLD, value=value)
