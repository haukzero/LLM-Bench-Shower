import gc
import torch
from collections import defaultdict
from typing import Dict, Tuple, List, Any, NamedTuple
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer
import envs
from bench import init_all_benchmarkers
from bench.utils import get_available_datasets
from model_cache import LRUModelCache


# NOTE(haukzero): dataset_name format: dataset_name/subdataset_name. For example, "LongBench/2wikimqa"
class ModelDatasetPair(NamedTuple):
    model_name: str
    dataset_name: str


_LLM_BENCHMARKER_RUNNER = None


class LLMBenchRunner:
    def __init__(self):
        self.available_datasets = get_available_datasets()
        self.bench_history: Dict[ModelDatasetPair, Dict] = defaultdict(dict)
        self.benchmarkers = init_all_benchmarkers()
        self.device_map = envs.LBS_LOCAL_DEVICE_MAP
        self.use_model_cache = envs.LBS_USE_MODEL_CACHE
        if self.use_model_cache:
            self.max_cached_local_models = envs.LBS_MAX_CACHED_LOCAL_MODELS
            self.max_gpu_utilization = envs.LBS_GPU_MAX_UTILIZATION
            self.max_cpu_utilization = envs.LBS_CPU_MAX_UTILIZATION
            self.model_cache = LRUModelCache(
                max_cached_models=self.max_cached_local_models,
                gpu_max_utilization=self.max_gpu_utilization,
                cpu_max_utilization=self.max_cpu_utilization,
                device_map=self.device_map,
            )
            self.eval_local_model_fn = self.eval_local_model_cached
        else:
            self.eval_local_model_fn = self.eval_local_model_uncached

    def _split_dataset_name(self, dataset_name: str) -> Tuple[str, str]:
        try:
            supdataset_name, subdataset_name = dataset_name.split("/")
        except ValueError:
            raise ValueError(
                f"Dataset name '{dataset_name}' is not in the correct format 'dataset_name/subdataset_name'."
            )
        if (
            supdataset_name not in self.available_datasets
            or subdataset_name not in self.available_datasets[supdataset_name]
        ):
            raise ValueError(f"Dataset {dataset_name} not found in available datasets.")
        return supdataset_name, subdataset_name

    def eval_local_model_uncached(
        self,
        model_name_or_path: str,
        dataset_name: str,
    ) -> Dict:
        if (model_name_or_path, dataset_name) in self.bench_history:
            return self.bench_history[(model_name_or_path, dataset_name)]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=self.device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        benchmark_results = self.benchmarkers[supdataset_name].evaluate_local_llm(
            model=model,
            tokenizer=tokenizer,
            subdataset_name=subdataset_name,
        )
        self.bench_history[(model_name_or_path, dataset_name)] = benchmark_results

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        return benchmark_results

    def eval_local_model_cached(
        self,
        model_name_or_path: str,
        dataset_name: str,
    ) -> Dict:
        if (model_name_or_path, dataset_name) in self.bench_history:
            return self.bench_history[(model_name_or_path, dataset_name)]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)

        # Use model cache to get model and tokenizer
        model, tokenizer = self.model_cache.get_model(model_name_or_path)

        benchmark_results = self.benchmarkers[supdataset_name].evaluate_local_llm(
            model=model,
            tokenizer=tokenizer,
            subdataset_name=subdataset_name,
        )
        self.bench_history[(model_name_or_path, dataset_name)] = benchmark_results
        return benchmark_results

    def eval_api_model(
        self,
        model_name: str,
        dataset_name: str,
        openai_api_key: str,
        base_url: str | None = None,
        *args,
        **kwargs,
    ) -> Dict:
        api_model_name = f"api::{model_name}"
        if (api_model_name, dataset_name) in self.bench_history:
            return self.bench_history[(api_model_name, dataset_name)]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)
        client = Client(api_key=openai_api_key, base_url=base_url)
        benchmark_results = self.benchmarkers[supdataset_name].evaluate_api_llm(
            client=client,
            model=model_name,
            subdataset_name=subdataset_name,
            *args,
            **kwargs,
        )
        self.bench_history[(api_model_name, dataset_name)] = benchmark_results
        return benchmark_results

    def eval_models(self, requests: List[Dict[str, Any]]) -> List[Dict]:
        results = []
        for req in requests:
            model_type: bytes = req.get("model_type", b"local")
            match model_type:
                case b"local":
                    results.append(self.eval_local_model_fn(**req))
                case b"api":
                    results.append(self.eval_api_model(**req))
                case _:
                    raise ValueError(f"Unknown model_type: {model_type}")
        return results

    def close(self):
        self.model_cache.clear_cache()
        del self.bench_history


def get_llm_bench_runner():
    global _LLM_BENCHMARKER_RUNNER
    if _LLM_BENCHMARKER_RUNNER is None:
        _LLM_BENCHMARKER_RUNNER = LLMBenchRunner()
    return _LLM_BENCHMARKER_RUNNER
