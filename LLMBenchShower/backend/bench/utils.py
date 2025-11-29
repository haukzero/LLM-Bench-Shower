import json
import os
from typing import List, Dict

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../configs/")
SUB_DATASETS_PATH = os.path.join(CONFIG_DIR, "sub_datasets.json")
DEFAULT_DATASET_PATHS_PATH = os.path.join(CONFIG_DIR, "default_dataset_paths.json")
LOCAL_DATASET_PATHS_PATH = os.path.join(CONFIG_DIR, "local_dataset_paths.json")
DATASET_PATHS: Dict[str, str] | None = None


def get_sub_datasets(dataset_name: str) -> List[str]:
    with open(SUB_DATASETS_PATH, "r") as f:
        supported_datasets = json.load(f)
    return supported_datasets.get(dataset_name, [])


def get_available_datasets() -> Dict[str, List[str]]:
    with open(SUB_DATASETS_PATH, "r") as f:
        supported_datasets = json.load(f)
    return supported_datasets

def _load_dataset_paths():
    global DATASET_PATHS
    with open(DEFAULT_DATASET_PATHS_PATH, "r") as f:
        dps = json.load(f)
    if os.path.exists(LOCAL_DATASET_PATHS_PATH):
        with open(LOCAL_DATASET_PATHS_PATH, "r") as f:
            local_paths = json.load(f)
        dps |= local_paths
    DATASET_PATHS = dps

def get_dataset_path(dataset_name: str) -> str:
    global DATASET_PATHS
    if DATASET_PATHS is None:
        _load_dataset_paths()
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Dataset {dataset_name} not found in dataset paths.")
    return DATASET_PATHS[dataset_name]
