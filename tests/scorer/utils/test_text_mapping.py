import itertools

from wholedoc_bench.schema import StyleName
from wholedoc_bench.scorer.utils.prose import normalize
from wholedoc_bench.scorer.utils.text_mapping import Origin, mapped_text, normalize_mapped


def test_normalization_expansion_and_composition_keep_origins_and_styles():
    value = mapped_text(
        "ﬁé­",
        (
            frozenset([StyleName.BOLD]),
            frozenset([StyleName.ITALIC]),
            frozenset([StyleName.UNDERLINE]),
            frozenset([StyleName.BOLD]),
        ),
        "source",
    )
    actual = normalize_mapped(value)
    assert actual.text == "fié"
    assert actual.origins == (
        (Origin("source", 0),),
        (Origin("source", 0),),
        (Origin("source", 1), Origin("source", 2)),
    )
    assert actual.styles[2] == frozenset([StyleName.ITALIC, StyleName.UNDERLINE])


def test_mapped_normalization_agrees_with_prose_normalize():
    for combination in itertools.product("ﬁé̕­² \r", repeat=3):
        text = "".join(combination)
        assert normalize_mapped(mapped_text(text, (), "x")).text == normalize(text)
