"""Setup mock datasets for testing."""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List


def create_mock_longbench_dataset(test_data_dir: str, num_samples: int = 5):
    """Create mock LongBench dataset files.
    
    Args:
        test_data_dir: Path to test data directory
        num_samples: Number of samples per subdataset
    """
    dataset_dir = os.path.join(test_data_dir, "LongBench")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create sample data for a few subdatasets
    subdatasets = [
        "2wikimqa", "dureader", "gov_report", "hotpotqa", "narrativeqa"
    ]
    
    for subdataset in subdatasets:
        file_path = os.path.join(dataset_dir, f"{subdataset}.jsonl")
        with open(file_path, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                item = {
                    "question": f"Sample question {i+1} for {subdataset}",
                    "context": f"This is a long context document for testing. " * 20 + f"It contains information relevant to question {i+1}.",
                    "answers": [f"Answer to question {i+1}"],
                    "qid": f"{subdataset}-{i+1}"
                }
                f.write(json.dumps(item) + '\n')
    
    print(f"✓ Created mock LongBench dataset with {len(subdatasets)} subdatasets")
    return dataset_dir


def create_mock_longbenchv2_dataset(test_data_dir: str, num_samples: int = 5):
    """Create mock LongBenchV2 dataset files.
    
    Args:
        test_data_dir: Path to test data directory
        num_samples: Number of samples per domain
    """
    dataset_dir = os.path.join(test_data_dir, "LongBenchV2")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create sample data for a few domains
    domains = [
        "Code_Repository_Understanding",
        "Long-dialogue_History_Understanding",
        "Long_In-context_Learning",
    ]
    
    for domain in domains:
        domain_dir = os.path.join(dataset_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        file_path = os.path.join(domain_dir, "data.jsonl")
        with open(file_path, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                item = {
                    "question": f"Sample question {i+1} for {domain}",
                    "instruction": f"Instruction for {domain} task {i+1}",
                    "context": f"This is a long context for {domain}. " * 15 + f"Information for task {i+1}.",
                    "answer": f"Answer to {domain} question {i+1}",
                    "id": f"{domain}-{i+1}"
                }
                f.write(json.dumps(item) + '\n')
    
    print(f"✓ Created mock LongBenchV2 dataset with {len(domains)} domains")
    return dataset_dir


def create_mock_c_eval_dataset(test_data_dir: str, num_samples: int = 5):
    """Create mock C-Eval dataset files."""
    dataset_dir = os.path.join(test_data_dir, "C-Eval")
    os.makedirs(dataset_dir, exist_ok=True)
    
    subjects = ["math", "history", "science"]
    
    for subject in subjects:
        file_path = os.path.join(dataset_dir, f"{subject}.jsonl")
        with open(file_path, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                item = {
                    "question": f"Sample {subject} question {i+1}",
                    "A": "Option A",
                    "B": "Option B", 
                    "C": "Option C",
                    "D": "Option D",
                    "answer": "A",
                    "explanation": f"This is the explanation for {subject} question {i+1}"
                }
                f.write(json.dumps(item) + '\n')
    
    print(f"✓ Created mock C-Eval dataset with {len(subjects)} subjects")
    return dataset_dir


def setup_test_datasets(test_data_dir: str = None, num_samples: int = 5) -> str:
    """Setup all mock datasets for testing.
    
    Args:
        test_data_dir: Path to test data directory (defaults to ./tests/test_data)
        num_samples: Number of samples per subdataset
        
    Returns:
        Path to the test data directory
    """
    if test_data_dir is None:
        test_data_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_data"
        )
    
    test_data_dir = os.path.abspath(test_data_dir)
    os.makedirs(test_data_dir, exist_ok=True)
    
    print(f"\n📁 Setting up mock datasets in: {test_data_dir}")
    
    create_mock_longbench_dataset(test_data_dir, num_samples)
    create_mock_longbenchv2_dataset(test_data_dir, num_samples)
    # Only create C-Eval if needed
    # create_mock_c_eval_dataset(test_data_dir, num_samples)
    
    print(f"\n✓ All mock datasets created successfully\n")
    return test_data_dir


def update_config_for_testing(config_dir: str = None, test_data_dir: str = None) -> None:
    """Update dataset config files to point to test data.
    
    Args:
        config_dir: Path to config directory (defaults to ./LLMBenchShower/configs)
        test_data_dir: Path to test data directory (defaults to ./tests/test_data)
    """
    if config_dir is None:
        config_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "LLMBenchShower",
            "configs"
        )
    
    if test_data_dir is None:
        test_data_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_data"
        )
    
    config_dir = os.path.abspath(config_dir)
    test_data_dir = os.path.abspath(test_data_dir)
    
    # Update dataset_paths.json
    paths_config_file = os.path.join(config_dir, "dataset_paths.json")
    if os.path.exists(paths_config_file):
        with open(paths_config_file, 'r') as f:
            paths_config = json.load(f)
        
        # Backup original
        backup_file = paths_config_file + ".backup"
        if not os.path.exists(backup_file):
            shutil.copy(paths_config_file, backup_file)
        
        # Update paths
        paths_config["LongBench"] = os.path.join(test_data_dir, "LongBench")
        paths_config["LongBenchV2"] = os.path.join(test_data_dir, "LongBenchV2")
        paths_config["C-Eval"] = os.path.join(test_data_dir, "C-Eval")
        
        with open(paths_config_file, 'w') as f:
            json.dump(paths_config, f, indent=2)
        
        print(f"✓ Updated config: {paths_config_file}")


def cleanup_test_data(test_data_dir: str = None) -> None:
    """Clean up test data directory.
    
    Args:
        test_data_dir: Path to test data directory
    """
    if test_data_dir is None:
        test_data_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "test_data"
        )
    
    test_data_dir = os.path.abspath(test_data_dir)
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
        print(f"✓ Cleaned up test data directory: {test_data_dir}")
