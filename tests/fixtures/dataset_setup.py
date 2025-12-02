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
    
def create_mock_mr_GMS8K_dataset(test_data_dir: str, num_samples: int = 5):
    """Create mock MR-GMS8K dataset files.
    
    MR-GMS8K is a benchmark for evaluating meta-reasoning abilities of large language models,
    focusing on the model's ability to identify errors in mathematical solutions.
    """
    dataset_dir = os.path.join(test_data_dir, "MR-GMS8K")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create sample data for MR-GMS8K
    data = []
    
    for i in range(num_samples):
        # Alternate between correct and incorrect solutions
        is_correct = i % 2 == 0
        
        # Example math problem and solution
        if is_correct:
            # Correct solution case
            question = f"If a shirt costs $25 and is discounted by 20%, what is the sale price?"
            model_output_steps = """
1. The original price is $25.
2. The discount is 20% of $25, which is 0.20 × $25 = $5.
3. The sale price is the original price minus the discount: $25 - $5 = $20.
            """
            answer_correctness = True
            solution_correctness = True
            first_error_step = None
            error_reason = None
        else:
            # Incorrect solution case with error in step 2
            question = f"A store sells books for $15 each. If a customer buys 3 books and gets a 10% discount, how much do they pay?"
            model_output_steps = """
1. The cost of 3 books at $15 each is 3 × $15 = $45.
2. The discount is 10% of $45, which is 0.10 × $45 = $3.50.
3. The final price is $45 - $3.50 = $41.50.
            """
            # The error is in step 2: 10% of $45 is $4.50, not $3.50
            answer_correctness = False
            solution_correctness = False
            first_error_step = 2
            error_reason = "The discount calculation is incorrect. 10% of $45 is $4.50, not $3.50."
        
        item = {
            "uuid": f"mr-GMS8K-{i+1}",
            "question": question,
            "model_output_steps": model_output_steps.strip(),
            "model_output_answer_correctness": answer_correctness,
            "model_output_solution_correctness": solution_correctness,
            "model_output_solution_first_error_step": first_error_step,
            "model_output_solution_first_error_reason": error_reason
        }
        data.append(item)
    
    # Write to JSON file
    file_path = os.path.join(dataset_dir, "MR-GMS8K.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Created mock MR-GMS8K dataset with {num_samples} samples")
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
    create_mock_c_eval_dataset(test_data_dir, num_samples)
    create_mock_mr_GMS8K_dataset(test_data_dir, num_samples)
    
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
        paths_config["MR-GMS8K"] = os.path.join(test_data_dir, "MR-GMS8K")

        
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
