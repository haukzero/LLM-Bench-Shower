import json
import os
from typing import List, Dict

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../configs/")
SUB_DATASETS_PATH = os.path.join(CONFIG_DIR, "sub_datasets.json")
DATASET_PATHS_PATH = os.path.join(CONFIG_DIR, "dataset_paths.json")


def get_sub_datasets(dataset_name: str) -> List[str]:
    with open(SUB_DATASETS_PATH, "r") as f:
        supported_datasets = json.load(f)
    return supported_datasets.get(dataset_name, [])


def get_available_datasets() -> Dict[str, List[str]]:
    with open(SUB_DATASETS_PATH, "r") as f:
        supported_datasets = json.load(f)
    return supported_datasets


def get_dataset_path(dataset_name: str) -> str:
    with open(DATASET_PATHS_PATH, "r") as f:
        dataset_paths = json.load(f)
    return dataset_paths.get(dataset_name, "")
