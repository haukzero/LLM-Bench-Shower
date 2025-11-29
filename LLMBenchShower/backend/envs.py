import os

# load model
LBS_LOCAL_DEVICE_MAP = os.getenv("LBS_LOCAL_DEVICE_MAP", "auto")

# model caching
LBS_USE_MODEL_CACHE = bool(int(os.getenv("LBS_USE_MODEL_CACHE", "1")))
LBS_MAX_CACHED_LOCAL_MODELS = int(os.getenv("LBS_MAX_CACHED_LOCAL_MODELS", "4"))
LBS_GPU_MAX_UTILIZATION = float(os.getenv("LBS_GPU_MAX_UTILIZATION", "0.5"))
LBS_CPU_MAX_UTILIZATION = float(os.getenv("LBS_CPU_MAX_UTILIZATION", "0.8"))

# database
LBS_DB_PATH = os.getenv(
    "LBS_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "benchmark_results.db",
    ),
)
LBS_DB_WRITEBACK_S = int(os.getenv("LBS_DB_WRITEBACK_S", "240"))
