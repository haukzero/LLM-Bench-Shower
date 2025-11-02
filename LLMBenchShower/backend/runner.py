from collections import defaultdict
from typing import Dict, Tuple
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer
from bench import init_all_benchmarkers
from bench.utils import get_available_datasets

ModelDatasetPair = Tuple[str, str]  # (model_name, dataset_name)
# NOTE(haukzero): dataset_name format: dataset_name/subdataset_name. For example, "LongBench/2wikimqa"

_LLM_BENCHMARKER_RUNNER = None


class LLMBenchRunner:
    def __init__(self):
        self.available_datasets = get_available_datasets()
        self.bench_history: Dict[ModelDatasetPair, Dict] = defaultdict(dict)
        self.benchmarkers = init_all_benchmarkers()

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

    def eval_local_model(
        self,
        model_name_or_path: str,
        dataset_name: str,
        device_map: Dict | str = "auto",
    ) -> Dict:
        if (model_name_or_path, dataset_name) in self.bench_history:
            return self.bench_history[(model_name_or_path, dataset_name)]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
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


def get_llm_bench_runner():
    global _LLM_BENCHMARKER_RUNNER
    if _LLM_BENCHMARKER_RUNNER is None:
        _LLM_BENCHMARKER_RUNNER = LLMBenchRunner()
    return _LLM_BENCHMARKER_RUNNER
