from wholedoc_bench.scorer.tracks.t1_text import TRACK as T1
from wholedoc_bench.scorer.tracks.t2_order import TRACK as T2
from wholedoc_bench.scorer.tracks.t3_structure import TRACK as T3
from wholedoc_bench.scorer.tracks.t4_tables import TRACK as T4
from wholedoc_bench.scorer.tracks.t5_machine_readable import TRACK as T5
from wholedoc_bench.scorer.tracks.t6_charts import TRACK as T6
from wholedoc_bench.scorer.tracks.t7_grounding import TRACK as T7

ALL_TRACKS = (T1, T2, T3, T4, T5, T6, T7)

__all__ = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "ALL_TRACKS"]
