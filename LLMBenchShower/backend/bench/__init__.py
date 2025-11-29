from .benchbase import BaseBench
from .longbench import LongBenchBenchmarker
from .longbench_v2 import LongBenchV2Benchmarker
from typing import Dict

# Import other benchmarkers that are implemented
try:
    from .c_eval import C_EvalBenchmarker
    _c_eval_available = True
except (ImportError, TypeError, NotImplementedError):
    _c_eval_available = False

try:
    from .cmmmu import CMMMUBenchmarker
    _cmmmu_available = True
except (ImportError, TypeError, NotImplementedError):
    _cmmmu_available = False

try:
    from .mr_gms8k import MR_GMS8KBenchmarker
    _mr_gms8k_available = True
except (ImportError, TypeError, NotImplementedError):
    _mr_gms8k_available = False

try:
    from .needle_in_haystack import NeedleInHaystackBenchmarker
    _needle_available = True
except (ImportError, TypeError, NotImplementedError):
    _needle_available = False

ALL_BENCHMARKERS = {
    "LongBench": LongBenchBenchmarker,
    "LongBenchV2": LongBenchV2Benchmarker,
}

# Add optional benchmarkers if available
if _c_eval_available:
    ALL_BENCHMARKERS["C-Eval"] = C_EvalBenchmarker
if _cmmmu_available:
    ALL_BENCHMARKERS["CMMMUBench"] = CMMMUBenchmarker
if _mr_gms8k_available:
    ALL_BENCHMARKERS["MR-GMS8K"] = MR_GMS8KBenchmarker
if _needle_available:
    ALL_BENCHMARKERS["NeedleInHaystack"] = NeedleInHaystackBenchmarker


def get_benchmarker(bench_name) -> BaseBench | None:
    return ALL_BENCHMARKERS.get(bench_name, None)


def init_all_benchmarkers() -> Dict[str, BaseBench]:
    benchmarkers: Dict[str, BaseBench] = {}
    for name, bench_cls in ALL_BENCHMARKERS.items():
        try:
            benchmarkers[name] = bench_cls()
        except TypeError as e:
            # Skip benchmarkers that are not fully implemented
            print(f"Warning: Skipping {name} - {str(e)}")
            continue
    return benchmarkers
