from collections.abc import Iterable, Mapping


def corpus_track_score(document_scores: Iterable[float | None]) -> float | None:
    values = [s for s in document_scores if s is not None]
    if not values:
        return None
    return sum(values) / len(values)


def overall(track_scores: Mapping[str, float | None]) -> float | None:
    values = [s for s in track_scores.values() if s is not None]
    if not values:
        return None
    return sum(values) / len(values)
