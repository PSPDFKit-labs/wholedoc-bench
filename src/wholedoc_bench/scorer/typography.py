import re
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from wholedoc_bench.schema import Box, Element, Font, Groundtruth, LayoutItem, is_typography_element
from wholedoc_bench.scorer.utils.boxes import box_place, item_place
from wholedoc_bench.scorer.utils.pairing import reaches
from wholedoc_bench.scorer.utils.prose import normalize
from wholedoc_bench.scorer.utils.tracks.table_links import cell_tokens

MIN_WORDS_HELD = 0.5
SIZE_TOLERANCE = 0.1
SIZE_STEP = 0.5

METRIC_COMPATIBLE: tuple[tuple[str, ...], ...] = (
    ("arial", "helvetica", "arimo", "liberationsans", "nimbussans", "nimbussanl"),
    ("arialnarrow", "helveticanarrow", "liberationsansnarrow", "nimbussansnarrow"),
    (
        "timesnewroman",
        "times",
        "tinos",
        "liberationserif",
        "nimbusroman",
        "nimbusromno9l",
        "texgyretermes",
    ),
    (
        "couriernew",
        "courier",
        "cousine",
        "liberationmono",
        "nimbusmono",
        "nimbusmonl",
        "texgyrecursor",
    ),
    ("calibri", "carlito"),
    ("cambria", "caladea"),
    ("georgia", "gelasio"),
    (
        "palatino",
        "palatinolinotype",
        "bookantiqua",
        "urwpalladio",
        "urwpalladiol",
        "texgyrepagella",
    ),
    ("centuryschool", "newcenturyschlbk", "centuryschl", "c059", "texgyreschola"),
    ("itcavantgarde", "avantgarde", "urwgothic", "urwgothicl", "texgyreadventor"),
    ("itcbookman", "bookman", "urwbookman", "urwbookmanl", "texgyrebonum"),
)
_GROUP = {name: group[0] for group in METRIC_COMPATIBLE for name in group}

LOOK_ALIKE_GENRES: dict[str, tuple[str, ...]] = {
    "grotesque": (
        "arial",
        "arialnarrow",
        "arialblack",
        "helveticaneue",
        "univers",
        "akzidenzgrotesk",
        "texgyreheros",
        "texgyreheroscn",
        "freesans",
        "microsoftsansserif",
        "roboto",
        "robotocondensed",
        "heebo",
        "inter",
        "sfpro",
        "sfprotext",
        "sfprodisplay",
        "sanfrancisco",
        "franklingothic",
        "franklingothicdemi",
        "librefranklin",
        "publicsans",
        "archivo",
        "worksans",
        "karla",
        "aptos",
        "aptosnarrow",
        "spacegrotesk",
        "overpass",
        "oswald",
        "barlow",
        "bahnschrift",
        "din",
        "dinpro",
        "ibmplexsans",
        "ibmplexsanscondensed",
        "geneva",
        "swiss721",
        "pragmatica",
    ),
    "geometric": (
        "futura",
        "centurygothic",
        "avenir",
        "avenirnext",
        "avenirnextcondensed",
        "avenirltstd",
        "avenirltpro",
        "avenirnextltpro",
        "montserrat",
        "poppins",
        "raleway",
        "itcavantgarde",
        "gotham",
        "proximanova",
        "josefinsans",
        "jost",
        "leaguespartan",
        "quicksand",
        "nunito",
        "twcen",
        "dmsans",
    ),
    "humanist": (
        "calibri",
        "candara",
        "corbel",
        "segoeui",
        "segoeuivariable",
        "frutiger",
        "myriad",
        "myriadpro",
        "sourcesans",
        "sourcesanspro",
        "sourcesans3",
        "opensans",
        "opensanscondensed",
        "notosans",
        "notosansdisplay",
        "droidsans",
        "verdana",
        "tahoma",
        "trebuchetms",
        "lucidagrande",
        "lucidasans",
        "lucidasansunicode",
        "gillsans",
        "optima",
        "firasans",
        "ptsans",
        "ubuntu",
        "ubuntusans",
        "cantarell",
        "oxygen",
        "hind",
        "assistant",
        "cabin",
        "lato",
        "dejavusans",
        "linuxbiolinumo",
        "ebrima",
        "leelawadeeui",
        "gadugi",
    ),
    "oldstyle": (
        "garamond",
        "ebgaramond",
        "cormorantgaramond",
        "adobegaramond",
        "adobegaramondpro",
        "minion",
        "minionpro",
        "palatino",
        "p052",
        "bembo",
        "sabon",
        "adobejenson",
        "alegreya",
        "calisto",
        "crimsontext",
        "crimsonpro",
        "goudyoldstyle",
        "hoeflertext",
        "iowanoldstyle",
        "linuxlibertineo",
        "linuxlibertinedisplayo",
        "perpetua",
        "sitka",
        "spectral",
        "cochin",
        "bigcaslon",
        "adobecaslon",
        "adobecaslonpro",
        "bookmanoldstyle",
        "itcbookman",
        "cardo",
    ),
    "transitional": (
        "timesnewroman",
        "freeserif",
        "stix",
        "stixgeneral",
        "stixmath",
        "dejavumathtexgyre",
        "georgia",
        "cambria",
        "baskerville",
        "baskervilleoldface",
        "librebaskerville",
        "bell",
        "century",
        "centuryschool",
        "charter",
        "bitstreamcharter",
        "constantia",
        "lucidabright",
        "merriweather",
        "notoserif",
        "notoserifdisplay",
        "ptserif",
        "sourceserif",
        "sourceserif4",
        "sourceserifpro",
        "ibmplexserif",
        "dejavuserif",
        "lora",
        "sylfaen",
    ),
    "didone": (
        "didot",
        "gfsdidot",
        "bodoni",
        "playfairdisplay",
        "dmserifdisplay",
        "latinmodernroman",
        "cmr",
        "computermodern",
    ),
    "slab": (
        "rockwell",
        "robotoslab",
        "zillaslab",
        "arvo",
        "bitter",
        "clarendon",
        "museoslab",
        "americantypewriter",
    ),
    "typewriter": (
        "couriernew",
        "courierprime",
        "freemono",
        "latinmodernmono",
        "linuxlibertinemonoo",
    ),
    "monosans": (
        "consolas",
        "menlo",
        "monaco",
        "dejavusansmono",
        "sourcecodepro",
        "firacode",
        "firamono",
        "jetbrainsmono",
        "inconsolata",
        "lucidaconsole",
        "andalemono",
        "cascadiacode",
        "cascadiacodepl",
        "cascadiamono",
        "cascadiamonopl",
        "hack",
        "robotomono",
        "ubuntumono",
        "ubuntusansmono",
        "notomono",
        "notosansmono",
        "ptmono",
        "ibmplexmono",
        "spacemono",
        "sfmono",
    ),
}
_GENRE = {name: genre for genre, names in LOOK_ALIKE_GENRES.items() for name in names}

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_STYLE_WORDS = re.compile(
    r"(?:[\s_-]*(?:regular|bold|italic|oblique|light|medium|semibold|demibold|thin|book"
    r"|psmt|mt|ps))+$",
    re.IGNORECASE,
)


def family_key(name: str) -> str:
    name = _SUBSET_PREFIX.sub("", name.strip())
    name = re.split(r"[-,]", name, maxsplit=1)[0]
    name = _STYLE_WORDS.sub("", name)
    key = re.sub(r"[\W_]+", "", name).casefold()
    return _GROUP.get(key, key)


def genre(name: str) -> str | None:
    return _GENRE.get(family_key(name))


def same_family(expected: str, found: str | None) -> bool:
    if found is None:
        return False
    if family_key(expected) == family_key(found):
        return True
    look = genre(expected)
    return look is not None and look == genre(found)


def same_size(expected: float, found: float | None) -> bool:
    return found is not None and reaches(SIZE_TOLERANCE * expected, abs(found - expected))


def size_group(size: float) -> float:
    return round(size / SIZE_STEP) * SIZE_STEP


@dataclass(frozen=True)
class _Part:
    field: str
    group: Callable[[Font], Hashable]
    right: Callable[[Font, Font | None], bool]


PARTS = (
    _Part(
        "family",
        lambda f: family_key(f.family or ""),
        lambda e, f: same_family(e.family or "", f and f.family),
    ),
    _Part(
        "font_class",
        lambda f: f.font_class,
        lambda e, f: f is not None and f.font_class is e.font_class,
    ),
    _Part(
        "size",
        lambda f: size_group(f.size or 0.0),
        lambda e, f: same_size(e.size or 0.0, f and f.size),
    ),
)


@dataclass(frozen=True)
class TypographyResult:
    parts: dict[str, float]
    unpaired: tuple[str, ...]
    blocks: tuple[dict[str, Any], ...] = ()

    @property
    def score(self) -> float:
        return sum(self.parts.values()) / len(self.parts)

    def details(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "parts": self.parts,
            "unpaired": list(self.unpaired),
            "blocks": list(self.blocks),
        }


def _words(text: str) -> Counter[str]:
    return cell_tokens([[text]])


def _symbols(text: str) -> str:
    return "".join(normalize(text).split()).casefold()


@dataclass(frozen=True)
class _Text[T]:
    thing: T
    words: Counter[str]
    symbols: str
    box: Box | None


def _text[T: (Element, LayoutItem)](thing: T) -> _Text[T]:
    words = _words(thing.text)
    return _Text(thing, words, "" if words else _symbols(thing.text), thing.box)


def _holding(block: Counter[str], item: Counter[str]) -> tuple[float, float]:
    shared = sum((block & item).values())
    return shared / max(sum(block.values()), 1), shared / max(sum(item.values()), 1)


def _distance(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax + aw / 2 - bx - bw / 2) ** 2 + (ay + ah / 2 - by - bh / 2) ** 2


def _nearest[T](block: _Text[Element], tied: Sequence[_Text[T]], occurrence: int) -> _Text[T]:
    boxes = [(t, t.box) for t in tied if t.box is not None]
    if block.box is not None and len(boxes) == len(tied):
        here = block.box
        return min(boxes, key=lambda pair: _distance(here, pair[1]))[0]
    return tied[min(occurrence, len(tied) - 1)]


def _holders[T](block: _Text[Element], items: Sequence[_Text[T]], occurrence: int) -> tuple[T, ...]:
    if not block.words:
        same = [t for t in items if not t.words and t.symbols and t.symbols == block.symbols]
        return (_nearest(block, same, occurrence).thing,) if same else ()
    held = [(_holding(block.words, t.words), t) for t in items]
    best = max((h for h, _ in held if reaches(h[0], MIN_WORDS_HELD)), default=None)
    if best is not None:
        tied = [t for h, t in held if h == best]
        return (_nearest(block, tied, occurrence).thing,)
    pieces = [t for h, t in held if h[0] > 0 and reaches(h[1], MIN_WORDS_HELD)]
    joined = sum((t.words for t in pieces), Counter[str]())
    if pieces and reaches(_holding(block.words, joined)[0], MIN_WORDS_HELD):
        return tuple(t.thing for t in pieces)
    return ()


def paired_items(
    truth: Groundtruth, layout: Sequence[LayoutItem]
) -> dict[str, tuple[LayoutItem, ...]]:
    by_page: dict[int, list[_Text[LayoutItem]]] = {}
    for item in sorted(layout, key=lambda i: item_place(i, i.text)):
        by_page.setdefault(item.page, []).append(_text(item))
    blocks = sorted(
        (_text(e) for e in truth.elements if is_typography_element(e)),
        key=lambda b: (b.thing.page, *box_place(b.box)),
    )
    seen: Counter[tuple[int, Hashable]] = Counter()
    found: dict[str, tuple[LayoutItem, ...]] = {}
    for block in blocks:
        page = block.thing.page
        same = (page, frozenset(block.words.items()) if block.words else block.symbols)
        found[block.thing.id] = _holders(block, by_page.get(page, []), seen[same])
        seen[same] += 1
    return {e.id: found[e.id] for e in truth.elements if e.id in found}


def score_typography(
    truth: Groundtruth, layout: Sequence[LayoutItem], fields: frozenset[str]
) -> TypographyResult:
    items = paired_items(truth, layout)
    rows: list[tuple[Font, dict[str, bool]]] = []
    blocks: list[dict[str, Any]] = []
    for e in truth.elements:
        if not e.font or e.id not in items:
            continue
        held = items[e.id]
        right = {
            part.field: bool(held) and all(part.right(e.font, item.font) for item in held)
            for part in _given(e.font, fields)
        }
        rows.append((e.font, right))
        blocks.append({"id": e.id, "items": [item.id for item in held], **right})
    parts: dict[str, float] = {}
    for part in PARTS:
        groups: dict[Hashable, list[bool]] = {}
        for want, right in rows:
            if part.field in right:
                groups.setdefault(part.group(want), []).append(right[part.field])
        if not groups:
            continue
        parts[part.field] = sum(sum(g) / len(g) for g in groups.values()) / len(groups)
    unpaired = tuple(element_id for element_id, held in items.items() if not held)
    return TypographyResult(parts, unpaired, tuple(blocks))


def _given(font: Font, fields: frozenset[str]) -> list[_Part]:
    return [p for p in PARTS if p.field in fields and getattr(font, p.field) is not None]


def failed_typography(truth: Groundtruth, fields: frozenset[str]) -> TypographyResult:
    annotated = [e for e in truth.elements if is_typography_element(e) and e.font]
    return TypographyResult(
        {part.field: 0.0 for part in PARTS if part.field in fields},
        tuple(e.id for e in annotated),
        tuple(
            {"id": e.id, "items": [], **{p.field: False for p in _given(e.font, fields)}}
            for e in annotated
            if e.font
        ),
    )
