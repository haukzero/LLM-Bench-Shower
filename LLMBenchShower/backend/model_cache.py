import gc
import psutil
import time
import torch
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_module
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, NamedTuple

DTYPE_TO_BYTES = {
    "float32": 4.0,
    "float": 4.0,
    "fp32": 4.0,
    "float16": 2.0,
    "fp16": 2.0,
    "half": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float8_e4m3fn": 1.0,
    "float8_e5m2": 1.0,
    "fp8": 1.0,
    "int8": 1.0,
    "uint8": 1.0,
    "int4": 0.5,
    "uint4": 0.5,
}


class ModelMeta(NamedTuple):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    estimated_memory: int
    device_map: Dict | None = None


class ModelStats:
    """Track statistics for each cached model."""

    def __init__(self, estimated_memory: int):
        self.access_count = 1
        self.last_access_time = time.time()
        self.load_time = time.time()
        self.estimated_memory = estimated_memory

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()

    def calculate_score(self, current_time: float, total_cache_memory: int) -> float:
        """
        Calculate a score for the model based on multiple factors.
        Higher score means higher priority to keep in cache.

        Factors:
        1. Access frequency: More frequent access = higher priority
        2. Recency: More recent access = higher priority
        3. Model size: Smaller models are easier to evict/reload
        4. Time in cache: Newly loaded models get a temporary boost

        Returns:
            float: Priority score (higher is better)
        """
        # Frequency component (0-1 normalized by log scale)
        # More accesses = higher score
        frequency_score = min(1.0, (self.access_count**0.5) / 10.0)

        # Recency component (decay over time)
        # Recent access within 5 minutes gets high score
        time_since_access = current_time - self.last_access_time
        recency_score = max(
            0.0, 1.0 - (time_since_access / 300.0)
        )  # 300 seconds = 5 minutes

        # Size component (inverse relationship)
        # Smaller models get lower penalty for eviction
        # Normalize by total cache capacity
        size_ratio = self.estimated_memory / total_cache_memory
        size_score = 1.0 - min(1.0, size_ratio)

        # New model boost (gradually decays)
        # Gives newly loaded models a chance to prove their worth
        time_in_cache = current_time - self.load_time
        newness_bonus = max(
            0.0, 1.0 - (time_in_cache / 60.0)
        )  # 60 seconds grace period

        # Weighted combination
        # Frequency and recency are most important
        score = (
            frequency_score * 0.4
            + recency_score * 0.35
            + size_score * 0.15
            + newness_bonus * 0.1
        )

        return score


class ModelPair(NamedTuple):
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


class ModelCache:
    def __init__(
        self,
        max_cached_models: int,
        gpu_max_utilization: float,
        cpu_max_utilization: float,
        device_map: str,
    ):
        self.max_cached_models = max_cached_models
        self.gpu_max_utilization = gpu_max_utilization
        self.cpu_max_utilization = cpu_max_utilization
        self.device_map = device_map

        self._verify_device_map()

        # GPU cache: stores models on GPU
        self.gpu_cache: Dict[str, ModelMeta] = {}
        # CPU cache: stores models offloaded to CPU
        self.cpu_cache: Dict[str, ModelMeta] = {}

        # Statistics tracking for smart eviction
        self.model_stats: Dict[str, ModelStats] = {}

        # Track GPU memory usage
        self.gpu_devices = self._get_gpu_devices()
        self.total_gpu_memory = self._get_total_gpu_memory()
        self.available_gpu_memory = self.total_gpu_memory * self.gpu_max_utilization

        # Track CPU memory usage
        self.total_cpu_memory = psutil.virtual_memory().total
        self.available_cpu_memory = self.total_cpu_memory * self.cpu_max_utilization
        self.current_cpu_usage = 0

    @property
    def num_cached_models(self) -> int:
        return len(self.gpu_cache) + len(self.cpu_cache)

    def _verify_device_map(self):
        if self.device_map == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available, cannot use device_map='auto'."
                )
            return
        if self.device_map.startswith("cuda:"):
            try:
                device_id = int(self.device_map.split(":")[-1])
                if device_id < 0 or device_id >= torch.cuda.device_count():
                    raise ValueError(
                        f"Invalid CUDA device ID {device_id} for device_map."
                    )
            except (ValueError, IndexError):
                raise ValueError(
                    f"Invalid device_map format: {self.device_map}. Expected 'auto' or 'cuda:<id>'."
                )
        else:
            raise NotImplementedError(f"Unsupported device_map: {self.device_map}")

    def _get_gpu_devices(self) -> list:
        """Get list of GPU devices based on device_map."""
        if self.device_map == "auto":
            return list(range(torch.cuda.device_count()))
        else:
            device_id = int(self.device_map.split(":")[-1])
            return [device_id]

    def _get_total_gpu_memory(self) -> int:
        """Get total GPU memory in bytes."""
        if self.device_map == "auto":
            # For auto mode, sum all GPU memory
            total = sum(
                torch.cuda.get_device_properties(i).total_memory
                for i in self.gpu_devices
            )
            return total
        else:
            # For specific device, return that device's memory
            device_id = self.gpu_devices[0] if self.gpu_devices else 0
            return torch.cuda.get_device_properties(device_id).total_memory

    def _get_bytes_per_param(self, torch_dtype: str) -> int:
        """Get bytes per parameter based on torch_dtype."""
        dtype_str = str(torch_dtype).lower().replace("torch.", "")
        return DTYPE_TO_BYTES.get(dtype_str, 2)

    def _cal_dense_params(self, config) -> int:
        """Calculate parameters for dense models."""
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        vocab_size = getattr(config, "vocab_size", 32000)
        intermediate_size = getattr(config, "intermediate_size", None)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        num_key_value_heads = getattr(config, "num_key_value_heads", None)

        # If num_key_value_heads is not specified, assume it equals num_attention_heads (MHA)
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # Calculate head dimension
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        # Embedding layer: vocab_size * hidden_size
        embedding_params = vocab_size * hidden_size

        # For each transformer layer:
        # 1. Attention: Q, K, V, O projections
        qkv_params = hidden_size * (
            num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim
        )
        o_params = num_attention_heads * head_dim * hidden_size
        attention_params = qkv_params + o_params

        # 2. MLP/FFN
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size  # Standard FFN size

        # Most models use SwiGLU or GELU with gate: 3 projections (up, gate, down)
        # up: hidden -> intermediate, gate: hidden -> intermediate, down: intermediate -> hidden
        mlp_params = (
            hidden_size * intermediate_size * 2 + intermediate_size * hidden_size
        )

        # 3. Layer norms (usually negligible, but include for completeness)
        layernorm_params = hidden_size * 2  # Two layer norms per layer

        # Total per layer
        params_per_layer = attention_params + mlp_params + layernorm_params

        # All layers
        total_layer_params = num_layers * params_per_layer

        # Output head (LM head)
        # Usually tied with embedding, but count separately for safety
        tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if tie_word_embeddings:
            lm_head_params = 0
        else:
            lm_head_params = vocab_size * hidden_size

        # Final layer norm
        final_layernorm_params = hidden_size

        total_params = (
            embedding_params
            + total_layer_params
            + lm_head_params
            + final_layernorm_params
        )

        return total_params

    def _cal_moe_params(self, config) -> int:
        """Calculate parameters for MoE models."""
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        vocab_size = getattr(config, "vocab_size", 32000)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        num_key_value_heads = getattr(config, "num_key_value_heads", None)

        # MoE specific parameters
        num_experts = getattr(config, "num_experts", 8)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", None)
        intermediate_size = getattr(config, "intermediate_size", None)

        # Which layers are MoE layers
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        # Embedding layer
        embedding_params = vocab_size * hidden_size

        # Attention parameters (same for all layers)
        qkv_params = hidden_size * (
            num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim
        )
        o_params = num_attention_heads * head_dim * hidden_size
        attention_params = qkv_params + o_params

        # MLP parameters for dense layers
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        dense_mlp_params = (
            hidden_size * intermediate_size * 2 + intermediate_size * hidden_size
        )

        # MLP parameters for MoE layers
        if moe_intermediate_size is None:
            moe_intermediate_size = intermediate_size

        # For MoE: each expert has its own MLP, plus a router
        expert_params = (
            moe_intermediate_size * hidden_size * 2
            + hidden_size * moe_intermediate_size
        )
        router_params = hidden_size * num_experts
        moe_mlp_params = num_experts * expert_params + router_params

        # Layer norms
        layernorm_params = hidden_size * 2

        # Count how many layers are MoE vs dense
        moe_layers = 0
        dense_layers = 0

        for layer_idx in range(num_layers):
            # Check if this layer is in mlp_only_layers (dense layer)
            if layer_idx in mlp_only_layers:
                dense_layers += 1
            # Check sparse step pattern
            elif decoder_sparse_step > 1 and layer_idx % decoder_sparse_step != 0:
                dense_layers += 1
            else:
                moe_layers += 1

        # Total parameters
        total_layer_params = moe_layers * (
            attention_params + moe_mlp_params + layernorm_params
        ) + dense_layers * (attention_params + dense_mlp_params + layernorm_params)

        # Output head
        tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if tie_word_embeddings:
            lm_head_params = 0
        else:
            lm_head_params = vocab_size * hidden_size

        final_layernorm_params = hidden_size

        total_params = (
            embedding_params
            + total_layer_params
            + lm_head_params
            + final_layernorm_params
        )

        # NOTE(haukzero): For MoE models during inference, only num_experts_per_tok experts are active
        # So the actual memory usage is less than total parameters
        # However, we need to load all experts, so we count total params
        # The activation memory will be lower due to sparse activation

        return total_params

    def _estimate_model_memory(self, model_name_or_path: str) -> int:
        """Estimate GPU memory required for a model in bytes."""
        config = AutoConfig.from_pretrained(model_name_or_path)

        # Get torch dtype
        torch_dtype = getattr(config, "torch_dtype", "float16")
        bytes_per_param = self._get_bytes_per_param(torch_dtype)

        # Detect if this is a MoE model
        num_experts = getattr(config, "num_experts", None)
        is_moe = num_experts is not None and num_experts > 1

        # Calculate number of parameters
        if is_moe:
            num_params = self._cal_moe_params(config)
        else:
            num_params = self._cal_dense_params(config)

        # Calculate base memory for model weights
        model_memory = num_params * bytes_per_param

        # Add overhead for:
        # - Optimizer states (not needed for inference, so skip)
        # - Activations (depends on batch size and sequence length)
        # - KV cache (depends on sequence length and batch size)
        # Conservative estimate: 20% overhead for activations and KV cache
        overhead_factor = 1.2

        # For MoE models, the activation overhead is actually lower
        # because only a subset of experts are active
        if is_moe:
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
            num_experts = getattr(config, "num_experts", 8)
            # Reduce overhead proportionally
            overhead_factor = 1.0 + 0.2 * (num_experts_per_tok / num_experts)

        estimated_memory = int(model_memory * overhead_factor)

        return estimated_memory

    def _get_current_gpu_usage(self) -> int:
        """Get current GPU memory usage in bytes."""
        if not self.gpu_devices:
            return 0

        if self.device_map == "auto":
            # Sum across all GPUs
            return sum(torch.cuda.memory_allocated(i) for i in self.gpu_devices)
        else:
            device_id = self.gpu_devices[0] if self.gpu_devices else 0
            return torch.cuda.memory_allocated(device_id)

    def _select_eviction_candidate(
        self, cache: Dict[str, ModelMeta], cache_type: str = "gpu"
    ) -> str:
        """
        Select the best candidate for eviction based on comprehensive scoring.

        Args:
            cache: The cache (gpu_cache or cpu_cache) to select from
            cache_type: Type of cache ("gpu" or "cpu")

        Returns:
            str: Model name to evict
        """
        if not cache:
            raise ValueError(
                f"Cannot select eviction candidate from empty {cache_type} cache"
            )

        current_time = time.time()

        # Calculate total memory in the cache for normalization
        total_cache_memory = sum(meta.estimated_memory for meta in cache.values())

        # Calculate scores for all models in the cache
        model_scores = {}
        for model_name in cache.keys():
            stats = self.model_stats[model_name]
            score = stats.calculate_score(current_time, total_cache_memory)
            model_scores[model_name] = score

        # Select model with lowest score (least valuable to keep)
        eviction_candidate = min(model_scores, key=model_scores.get)

        return eviction_candidate

    def _move_model_to_cpu(self, model_name: str):
        """Move a model from GPU cache to CPU cache."""
        if model_name not in self.gpu_cache:
            return

        gpu_model_meta = self.gpu_cache[model_name]
        model = gpu_model_meta.model
        tokenizer = gpu_model_meta.tokenizer
        estimated_memory = gpu_model_meta.estimated_memory

        # Save the original device_map if it exists (for models loaded with device_map)
        original_device_map = None
        if hasattr(model, "hf_device_map"):
            original_device_map = model.hf_device_map.copy()

        # Remove accelerate hooks before moving to CPU
        # This is crucial for models loaded with device_map="auto"
        try:
            # Recursively remove hooks from all submodules
            for module in model.modules():
                remove_hook_from_module(module, recurse=False)

        except (ImportError, AttributeError):
            # Fallback: manually remove hooks (works for all accelerate versions)
            for module in model.modules():
                # Remove forward hooks
                if hasattr(module, "_forward_hooks"):
                    module._forward_hooks.clear()
                if hasattr(module, "_forward_pre_hooks"):
                    module._forward_pre_hooks.clear()
                # Restore original forward method
                if hasattr(module, "_old_forward"):
                    module.forward = module._old_forward
                    delattr(module, "_old_forward")

        # Now safely move to CPU
        cpu_model = model.to("cpu")

        # Update caches
        del model
        del self.gpu_cache[model_name]
        torch.cuda.empty_cache()
        gc.collect()

        self.cpu_cache[model_name] = ModelMeta(
            model=cpu_model,
            tokenizer=tokenizer,
            estimated_memory=estimated_memory,
            device_map=original_device_map,  # Save device_map for later restoration
        )
        self.current_cpu_usage += estimated_memory

    def _evict_from_cpu(self, model_name: str):
        """Remove a model from CPU cache completely."""
        if model_name not in self.cpu_cache:
            return

        cpu_model_meta = self.cpu_cache[model_name]
        model = cpu_model_meta.model
        tokenizer = cpu_model_meta.tokenizer
        estimated_memory = cpu_model_meta.estimated_memory

        # Delete model and stats
        del model
        del tokenizer
        del self.cpu_cache[model_name]
        del self.model_stats[model_name]
        self.current_cpu_usage -= estimated_memory
        gc.collect()

    def _make_space_on_gpu(self, required_memory: int):
        """Evict models from GPU to CPU until there's enough space."""
        while self.gpu_cache:
            current_usage = self._get_current_gpu_usage()
            if current_usage + required_memory <= self.available_gpu_memory:
                return

            # Use smart eviction: select model with lowest score
            eviction_candidate = self._select_eviction_candidate(self.gpu_cache, "gpu")

            # Check if we need to make space on CPU first
            candidate_memory = self.gpu_cache[eviction_candidate].estimated_memory
            if self.current_cpu_usage + candidate_memory > self.available_cpu_memory:
                self._make_space_on_cpu(candidate_memory)

            self._move_model_to_cpu(eviction_candidate)

    def _make_space_on_cpu(self, required_memory: int):
        """Evict models from CPU until there's enough space."""
        while self.cpu_cache:
            if self.current_cpu_usage + required_memory <= self.available_cpu_memory:
                return

            # Use smart eviction: select model with lowest score from CPU cache
            eviction_candidate = self._select_eviction_candidate(self.cpu_cache, "cpu")
            self._evict_from_cpu(eviction_candidate)

    def _enforce_model_count_limit(self, total_models: int):
        """Ensure total cached models don't exceed max_cached_models."""
        while total_models > self.max_cached_models:
            # First try to evict from CPU
            if self.cpu_cache:
                eviction_candidate = self._select_eviction_candidate(
                    self.cpu_cache, "cpu"
                )
                self._evict_from_cpu(eviction_candidate)
                total_models -= 1
            # If CPU cache is empty, evict from GPU
            elif self.gpu_cache:
                eviction_candidate = self._select_eviction_candidate(
                    self.gpu_cache, "gpu"
                )
                self._move_model_to_cpu(eviction_candidate)
                # Then immediately evict from CPU
                self._evict_from_cpu(eviction_candidate)
                total_models -= 1

    def get_model(self, model_name_or_path: str) -> ModelPair:
        """
        Get a model and tokenizer, managing cache automatically.

        Args:
            model_name_or_path: Path or name of the model to load

        Returns:
            Tuple of (model, tokenizer)
        """
        # Case 1: Model is in GPU cache
        if model_name_or_path in self.gpu_cache:
            # Update access statistics
            self.model_stats[model_name_or_path].update_access()

            gpu_model_meta = self.gpu_cache[model_name_or_path]
            model = gpu_model_meta.model
            tokenizer = gpu_model_meta.tokenizer
            return ModelPair(model=model, tokenizer=tokenizer)

        # Case 2: Model is in CPU cache
        if model_name_or_path in self.cpu_cache:
            cpu_model_meta = self.cpu_cache[model_name_or_path]
            model = cpu_model_meta.model
            tokenizer = cpu_model_meta.tokenizer
            estimated_memory = cpu_model_meta.estimated_memory
            original_device_map = cpu_model_meta.device_map

            # Update access statistics
            self.model_stats[model_name_or_path].update_access()

            # Remove from CPU cache
            del self.cpu_cache[model_name_or_path]
            self.current_cpu_usage -= estimated_memory

            # Make space on GPU if needed
            self._make_space_on_gpu(estimated_memory)

            # Move model back to GPU
            # If we have the original device_map, use dispatch_model to restore it
            if original_device_map is not None:
                model = dispatch_model(model, device_map=original_device_map)
            else:
                # Fallback: simple .to() method
                target_device = self.device_map if self.device_map != "auto" else "cuda"
                model = model.to(target_device)

            # Add to GPU cache
            self.gpu_cache[model_name_or_path] = ModelMeta(
                model=model,
                tokenizer=tokenizer,
                estimated_memory=estimated_memory,
                device_map=original_device_map,
            )

            return ModelPair(model=model, tokenizer=tokenizer)

        # Case 3: Model not in cache, need to load it
        estimated_memory = self._estimate_model_memory(model_name_or_path)

        # Check if adding this model would exceed limits
        total_models = self.num_cached_models + 1
        self._enforce_model_count_limit(total_models)

        # Make space on GPU if needed
        self._make_space_on_gpu(estimated_memory)

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=self.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Save the device_map if it was created by accelerate
        original_device_map = None
        if hasattr(model, "hf_device_map"):
            original_device_map = model.hf_device_map.copy()

        # Initialize statistics for this model
        self.model_stats[model_name_or_path] = ModelStats(estimated_memory)

        # Add to GPU cache
        self.gpu_cache[model_name_or_path] = ModelMeta(
            model=model,
            tokenizer=tokenizer,
            estimated_memory=estimated_memory,
            device_map=original_device_map,
        )

        return ModelPair(model=model, tokenizer=tokenizer)

    def clear_cache(self) -> None:
        """Clear all caches."""
        # Clear GPU cache
        for model_name in list(self.gpu_cache.keys()):
            model_meta = self.gpu_cache[model_name]
            model = model_meta.model
            tokenizer = model_meta.tokenizer
            del model
            del tokenizer
            del self.gpu_cache[model_name]

        # Clear CPU cache
        for model_name in list(self.cpu_cache.keys()):
            model_meta = self.cpu_cache[model_name]
            model = model_meta.model
            tokenizer = model_meta.tokenizer
            del model
            del tokenizer
            del self.cpu_cache[model_name]

        # Clear statistics
        self.model_stats.clear()
        self.current_cpu_usage = 0

        # Force garbage collection
        torch.cuda.empty_cache()
        gc.collect()
