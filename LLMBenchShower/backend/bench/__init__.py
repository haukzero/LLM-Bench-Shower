from .benchbase import BaseBench
from .c_eval import C_EvalBenchmarker
from .cmmmu import CMMMUBenchmarker
from .longbench import LongBenchBenchmarker
from .longbench_v2 import LongBenchV2Benchmarker
from .mr_gms8k import MR_GMS8KBenchmarker
from .needle_in_haystack import NeedleInHaystackBenchmarker
from typing import Dict

ALL_BENCHMARKERS = {
    "C-Eval": C_EvalBenchmarker,
    "CMMMUBench": CMMMUBenchmarker,
    "LongBench": LongBenchBenchmarker,
    "LongBenchV2": LongBenchV2Benchmarker,
    "MR-GMS8K": MR_GMS8KBenchmarker,
    "NeedleInHaystack": NeedleInHaystackBenchmarker,
}


def get_benchmarker(bench_name) -> BaseBench | None:
    return ALL_BENCHMARKERS.get(bench_name, None)


def init_all_benchmarkers() -> Dict[str, BaseBench]:
    benchmarkers: Dict[str, BaseBench] = {}
    for name, bench_cls in ALL_BENCHMARKERS.items():
        benchmarkers[name] = bench_cls()
    return benchmarkers
