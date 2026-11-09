from functools import cached_property
from typing import TYPE_CHECKING

from wholedoc_bench.schema import Groundtruth, Prediction
from wholedoc_bench.scorer.utils.blocks import Block, ParsedDocument, parse_markdown

if TYPE_CHECKING:
    from wholedoc_bench.scorer.block_integrity import BlockIntegrityResult
    from wholedoc_bench.scorer.page_breaks import PageBreakResult
    from wholedoc_bench.scorer.text_styles import StyleResult
    from wholedoc_bench.scorer.typography import TypographyResult


class DocContext:
    def __init__(self, truth: Groundtruth, pred: Prediction) -> None:
        self.truth = truth
        self.pred = pred

    @cached_property
    def parsed(self) -> ParsedDocument:
        return parse_markdown(self.pred.markdown)

    @property
    def document_text(self) -> str:
        return self.parsed.document_text

    @property
    def blocks(self) -> list[Block]:
        return self.parsed.blocks

    @cached_property
    def text_styles(self) -> "StyleResult":
        from wholedoc_bench.scorer.text_styles import score_text_styles

        return score_text_styles(self.truth, self.blocks)

    @cached_property
    def block_integrity(self) -> "BlockIntegrityResult":
        from wholedoc_bench.scorer.block_integrity import score_block_integrity
        from wholedoc_bench.scorer.tracks.t3_structure import output_blocks
        from wholedoc_bench.scorer.utils.prose import normalize

        return score_block_integrity(self.truth, output_blocks(self), normalize=normalize)

    @cached_property
    def page_breaks(self) -> "PageBreakResult":
        from wholedoc_bench.scorer.page_breaks import score_page_breaks
        from wholedoc_bench.scorer.tracks.t3_structure import output_blocks

        return score_page_breaks(self.truth, output_blocks(self), self.block_integrity)

    @cached_property
    def typography(self) -> "TypographyResult":
        from wholedoc_bench.corpus.checks import font_fields
        from wholedoc_bench.scorer.typography import score_typography

        return score_typography(self.truth, self.pred.layout, font_fields(self.truth))
