import gc
import torch
import threading
import time
from collections import defaultdict
from typing import Dict, Tuple, List, Any, NamedTuple, Set
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer
import envs
from bench import init_all_benchmarkers
from bench.utils import get_available_datasets
from db import BenchmarkDatabase
from model_cache import ModelCache


# NOTE(haukzero): dataset_name format: dataset_name/subdataset_name. For example, "LongBench/2wikimqa"
class ModelDatasetPair(NamedTuple):
    model_name: str
    dataset_name: str


_LLM_BENCHMARKER_RUNNER = None


class LLMBenchRunner:
    def __init__(self):
        self.available_datasets = get_available_datasets()
        self.benchmarkers = init_all_benchmarkers()

        self.db = BenchmarkDatabase(envs.LBS_DB_PATH)
        # Keep in-memory cache for fast access during runtime
        self.bench_history: Dict[ModelDatasetPair, Dict] = defaultdict(dict)
        # Track which results need to be written to database
        self._dirty_results: Set[ModelDatasetPair] = set()
        self._dirty_lock = threading.Lock()

        # Load existing results from database into memory
        self._load_history_from_db()

        # Start background write-back thread
        self._stop_writeback = threading.Event()
        self._writeback_interval = envs.LBS_DB_WRITEBACK_S
        self._writeback_thread = threading.Thread(
            target=self._writeback_worker, daemon=True, name="DBWriteBackThread"
        )
        self._writeback_thread.start()

        self.device_map = envs.LBS_LOCAL_DEVICE_MAP
        self.use_model_cache = envs.LBS_USE_MODEL_CACHE
        if self.use_model_cache:
            self.max_cached_local_models = envs.LBS_MAX_CACHED_LOCAL_MODELS
            self.max_gpu_utilization = envs.LBS_GPU_MAX_UTILIZATION
            self.max_cpu_utilization = envs.LBS_CPU_MAX_UTILIZATION
            self.model_cache = ModelCache(
                max_cached_models=self.max_cached_local_models,
                gpu_max_utilization=self.max_gpu_utilization,
                cpu_max_utilization=self.max_cpu_utilization,
                device_map=self.device_map,
            )
            self.eval_local_model_fn = self.eval_local_model_cached
        else:
            self.eval_local_model_fn = self.eval_local_model_uncached

    def _load_history_from_db(self):
        all_results = self.db.get_all_results()
        for model_name, dataset_name, results, _, _ in all_results:
            pair = ModelDatasetPair(model_name, dataset_name)
            self.bench_history[pair] = results

    def _writeback_worker(self):
        while not self._stop_writeback.wait(timeout=self._writeback_interval):
            self._flush_dirty_results()

    def _mark_dirty(self, pair: ModelDatasetPair):
        with self._dirty_lock:
            self._dirty_results.add(pair)

    def _flush_dirty_results(self):
        with self._dirty_lock:
            if not self._dirty_results:
                return

            # Copy and clear the dirty set
            dirty_pairs = list(self._dirty_results)
            self._dirty_results.clear()

        # Prepare batch data
        batch_data = []
        for pair in dirty_pairs:
            if pair in self.bench_history:
                batch_data.append(
                    (pair.model_name, pair.dataset_name, self.bench_history[pair])
                )

        # Write to database
        if batch_data:
            try:
                self.db.save_results_batch(batch_data)
            except Exception as e:
                # If write fails, mark them as dirty again
                print(f"Warning: Failed to write results to database: {e}")
                with self._dirty_lock:
                    self._dirty_results.update(dirty_pairs)

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
        *args,
        **kwargs,
    ) -> Dict:
        pair = ModelDatasetPair(model_name_or_path, dataset_name)
        if pair in self.bench_history:
            return self.bench_history[pair]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=self.device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        benchmark_results = self.benchmarkers[supdataset_name].evaluate_local_llm(
            model=model,
            tokenizer=tokenizer,
            subdataset_name=subdataset_name,
            *args,
            **kwargs,
        )
        # Save to memory and mark for write-back
        self.bench_history[pair] = benchmark_results
        self._mark_dirty(pair)

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        return benchmark_results

    def eval_local_model_cached(
        self,
        model_name_or_path: str,
        dataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        pair = ModelDatasetPair(model_name_or_path, dataset_name)
        if pair in self.bench_history:
            return self.bench_history[pair]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)

        # Use model cache to get model and tokenizer
        model, tokenizer = self.model_cache.get_model(model_name_or_path)

        benchmark_results = self.benchmarkers[supdataset_name].evaluate_local_llm(
            model=model,
            tokenizer=tokenizer,
            subdataset_name=subdataset_name,
            *args,
            **kwargs,
        )
        # Save to memory and mark for write-back
        self.bench_history[pair] = benchmark_results
        self._mark_dirty(pair)
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
        pair = ModelDatasetPair(api_model_name, dataset_name)
        if pair in self.bench_history:
            return self.bench_history[pair]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)
        client = Client(api_key=openai_api_key, base_url=base_url)
        benchmark_results = self.benchmarkers[supdataset_name].evaluate_api_llm(
            client=client,
            model=model_name,
            subdataset_name=subdataset_name,
            *args,
            **kwargs,
        )
        # Save to memory and mark for write-back
        self.bench_history[pair] = benchmark_results
        self._mark_dirty(pair)
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

    def get_database_stats(self) -> Dict:
        return self.db.get_stats()

    def get_all_history(self) -> List[Tuple[str, str, Dict, str, str]]:
        return self.db.get_all_results()

    def clear_history(self, model_name: str = None, dataset_name: str = None) -> int:
        """Clear benchmark history.

        Args:
            model_name: If specified, only clear results for this model
            dataset_name: If specified, only clear results for this dataset

        Returns:
            Number of results cleared
        """
        if model_name and dataset_name:
            # Clear specific result
            pair = ModelDatasetPair(model_name, dataset_name)
            if pair in self.bench_history:
                del self.bench_history[pair]
            # Remove from dirty set if present
            with self._dirty_lock:
                self._dirty_results.discard(pair)
            return 1 if self.db.delete_result(model_name, dataset_name) else 0
        elif model_name or dataset_name:
            # Clear by model or dataset - need to iterate
            count = 0
            to_delete = []
            for pair in self.bench_history.keys():
                if (model_name and pair.model_name == model_name) or (
                    dataset_name and pair.dataset_name == dataset_name
                ):
                    to_delete.append(pair)
            for pair in to_delete:
                del self.bench_history[pair]
                # Remove from dirty set if present
                with self._dirty_lock:
                    self._dirty_results.discard(pair)
                if self.db.delete_result(pair.model_name, pair.dataset_name):
                    count += 1
            return count
        else:
            # Clear all
            self.bench_history.clear()
            # Clear dirty set
            with self._dirty_lock:
                self._dirty_results.clear()
            return self.db.clear_all_results()

    def close(self):
        """Clean up resources and flush all pending writes."""
        # Stop the write-back thread
        self._stop_writeback.set()
        if self._writeback_thread.is_alive():
            self._writeback_thread.join(timeout=5.0)

        # Flush any remaining dirty results
        self._flush_dirty_results()

        if self.use_model_cache:
            self.model_cache.clear_cache()


def get_llm_bench_runner():
    global _LLM_BENCHMARKER_RUNNER
    if _LLM_BENCHMARKER_RUNNER is None:
        _LLM_BENCHMARKER_RUNNER = LLMBenchRunner()
    return _LLM_BENCHMARKER_RUNNER
