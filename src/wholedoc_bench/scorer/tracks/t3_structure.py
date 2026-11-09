from collections.abc import Callable
from typing import Any

from wholedoc_bench.schema import Groundtruth
from wholedoc_bench.scorer.base import Source, SubMetric, Track
from wholedoc_bench.scorer.block_integrity import OutputBlock, has_blocks
from wholedoc_bench.scorer.context import DocContext
from wholedoc_bench.scorer.heading_tree import Heading, has_headings, score_heading_tree
from wholedoc_bench.scorer.relations import has_links, score_relations
from wholedoc_bench.scorer.utils.barcodes import without_payloads
from wholedoc_bench.scorer.utils.blocks import Block, BlockKind, comparable_text


def output_blocks(ctx: DocContext) -> list[OutputBlock]:
    texts: list[str] = []
    for block in ctx.blocks:
        if block.kind is BlockKind.TABLE:
            continue
        heading = _heading(block)
        texts.append(block.text if heading is None else heading.text)
    return [OutputBlock(text) for text in without_payloads(ctx.truth, texts)]


def output_headings(ctx: DocContext) -> list[Heading]:
    return [heading for block in ctx.blocks if (heading := _heading(block)) is not None]


def _heading(block: Block) -> Heading | None:
    if block.kind is BlockKind.HEADING and block.level is not None:
        return Heading(block.level, block.text.strip())
    return None


def has_outline(truth: Groundtruth, normalize: Callable[[str], str] = comparable_text) -> bool:
    return has_headings(truth) or has_blocks(truth, normalize)


class HeadingTree(SubMetric):
    name = "heading_tree"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return has_outline(truth)

    def score(self, ctx: DocContext) -> float:
        return score_heading_tree(ctx.truth, output_headings(ctx), normalize=comparable_text)


class BlockIntegrity(SubMetric):
    name = "block_integrity"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        return has_blocks(truth, normalize=comparable_text)

    def score(self, ctx: DocContext) -> float:
        return ctx.block_integrity.f1


class Relations(SubMetric):
    name = "relations"
    source = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        return has_links(truth)

    def score(self, ctx: DocContext) -> float:
        return score_relations(ctx.truth, ctx.pred, normalize=comparable_text)


class TextStyles(SubMetric):
    name = "text_styles"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        from wholedoc_bench.corpus.checks import check_style_annotations

        return check_style_annotations(truth)

    def score(self, ctx: DocContext) -> float:
        result = ctx.text_styles
        assert result.score is not None
        return result.score

    def details(self, truth: Groundtruth, ctx: DocContext | None, status: str) -> dict[str, Any]:
        from wholedoc_bench.scorer.text_styles import style_status

        return (
            ctx.text_styles.details
            if status == "scored" and ctx
            else style_status(truth, status).details
        )


class PageBreaks(SubMetric):
    name = "page_breaks"
    source = Source.MARKDOWN

    def applies(self, truth: Groundtruth) -> bool:
        from wholedoc_bench.scorer.page_breaks import has_page_breaks

        return has_page_breaks(truth)

    def score(self, ctx: DocContext) -> float:
        result = ctx.page_breaks
        assert result.score is not None
        return result.score

    def details(self, truth: Groundtruth, ctx: DocContext | None, status: str) -> dict[str, Any]:
        from wholedoc_bench.scorer.page_breaks import failed_page_breaks

        return (
            ctx.page_breaks.details
            if status == "scored" and ctx
            else failed_page_breaks(truth).details
        )


class Typography(SubMetric):
    name = "typography"
    source = Source.JSON

    def applies(self, truth: Groundtruth) -> bool:
        from wholedoc_bench.corpus.checks import font_fields

        return bool(font_fields(truth))

    def score(self, ctx: DocContext) -> float:
        return ctx.typography.score

    def details(self, truth: Groundtruth, ctx: DocContext | None, status: str) -> dict[str, Any]:
        from wholedoc_bench.corpus.checks import font_fields
        from wholedoc_bench.scorer.typography import failed_typography

        if status == "not_applicable":
            return {}
        if status == "scored" and ctx:
            return ctx.typography.details(status)
        return failed_typography(truth, font_fields(truth)).details(status)


TRACK = Track(
    id="T3",
    name="Structure and formatting",
    parts=(
        (HeadingTree(), 1.0),
        (BlockIntegrity(), 1.0),
        (Relations(), 1.0),
        (TextStyles(), 1.0),
        (PageBreaks(), 0.5),
        (Typography(), 0.5),
    ),
)
