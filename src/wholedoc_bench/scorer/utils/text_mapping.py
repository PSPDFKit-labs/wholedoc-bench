import unicodedata
from dataclasses import dataclass

from wholedoc_bench.schema import StyleName
from wholedoc_bench.scorer.utils.prose import _SAME_MEANING, normalize


@dataclass(frozen=True)
class Origin:
    owner: str
    offset: int


@dataclass(frozen=True)
class MappedText:
    text: str
    styles: tuple[frozenset[StyleName], ...]
    origins: tuple[tuple[Origin, ...], ...]


def mapped_text(text: str, styles: tuple[frozenset[StyleName], ...], owner: str) -> MappedText:
    if not styles:
        styles = tuple(frozenset() for _ in text)
    return MappedText(text, styles, tuple((Origin(owner, i),) for i in range(len(text))))


def concatenate(parts: list[MappedText], separator: str = " ") -> MappedText:
    text: list[str] = []
    styles: list[frozenset[StyleName]] = []
    origins: list[tuple[Origin, ...]] = []
    for i, part in enumerate(parts):
        if i:
            text.append(separator)
            styles.extend(frozenset() for _ in separator)
            origins.extend(() for _ in separator)
        text.append(part.text)
        styles.extend(part.styles)
        origins.extend(part.origins)
    return MappedText("".join(text), tuple(styles), tuple(origins))


def normalize_mapped(value: MappedText) -> MappedText:
    atoms: list[tuple[str, frozenset[StyleName], tuple[Origin, ...]]] = []
    for char, styles, origins in zip(value.text, value.styles, value.origins, strict=True):
        atoms.extend((part, styles, origins) for part in unicodedata.normalize("NFKD", char))
    clusters: list[list[tuple[str, frozenset[StyleName], tuple[Origin, ...]]]] = []
    for atom in atoms:
        char = atom[0]
        if clusters and (
            unicodedata.combining(char)
            or len(unicodedata.normalize("NFC", "".join(a[0] for a in clusters[-1]) + char))
            < len(unicodedata.normalize("NFC", "".join(a[0] for a in clusters[-1]))) + 1
        ):
            clusters[-1].append(atom)
        else:
            clusters.append([atom])
    output: list[tuple[str, frozenset[StyleName], tuple[Origin, ...]]] = []
    for cluster in clusters:
        if unicodedata.combining(cluster[0][0]) == 0:
            cluster[1:] = sorted(cluster[1:], key=lambda a: unicodedata.combining(a[0]))
        else:
            cluster.sort(key=lambda a: unicodedata.combining(a[0]))
        normalized = unicodedata.normalize("NFC", "".join(a[0] for a in cluster))
        remaining = list(cluster)
        for char in normalized:
            contributors = []
            for part in unicodedata.normalize("NFD", char):
                index = next(i for i, atom in enumerate(remaining) if atom[0] == part)
                contributors.append(remaining.pop(index))
            styles = frozenset(s for atom in contributors for s in atom[1])
            origins = tuple(dict.fromkeys(o for atom in contributors for o in atom[2]))
            translated = char.translate(_SAME_MEANING)
            output.extend((c, styles, origins) for c in translated)
    collapsed: list[tuple[str, frozenset[StyleName], tuple[Origin, ...]]] = []
    for char, styles, origins in output:
        if char.isspace():
            if collapsed and collapsed[-1][0] == " ":
                before = collapsed[-1]
                collapsed[-1] = (
                    " ",
                    before[1] | styles,
                    tuple(dict.fromkeys((*before[2], *origins))),
                )
            elif collapsed:
                collapsed.append((" ", styles, origins))
        else:
            collapsed.append((char, styles, origins))
    if collapsed and collapsed[-1][0] == " ":
        collapsed.pop()
    result = MappedText(
        "".join(c[0] for c in collapsed),
        tuple(c[1] for c in collapsed),
        tuple(c[2] for c in collapsed),
    )
    if result.text != normalize(value.text):
        raise ValueError("mapped normalization disagrees with prose.normalize")
    return result
