#!/usr/bin/env python3
"""
Quick test script - simplified version for fast testing.

This is a simpler alternative to test_benchmarks.py for quick testing
of individual benchmarkers without verbose output.

Usage:
    python quick_test.py longbench        # Test LongBench
    python quick_test.py longbench_v2     # Test LongBenchV2  
    python quick_test.py all              # Test all benchmarks
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LLMBenchShower", "backend"))

from fixtures.mock_llm import (
    create_mock_model,
    create_mock_tokenizer,
    create_mock_client
)
from fixtures.dataset_setup import (
    setup_test_datasets,
    update_config_for_testing,
)
from bench import init_all_benchmarkers
from bench.utils import get_available_datasets


def quick_test(bench_name: str = "all", num_samples: int = 3):
    """Quick test for benchmarkers.
    
    Args:
        bench_name: Benchmarker name or "all"
        num_samples: Number of samples to test
    """
    print(f"\n🚀 Quick Test - {bench_name}")
    print("=" * 50)
    
    # Setup
    test_dir = setup_test_datasets(num_samples=num_samples)
    update_config_for_testing(test_data_dir=test_dir)
    
    # Get benchmarkers
    benchmarkers = init_all_benchmarkers()
    available = get_available_datasets()
    
    # Determine which benchmarks to test
    if bench_name == "all":
        benches_to_test = list(available.keys())
    else:
        # Map common names to full names
        name_mapping = {
            "longbench": "LongBench",
            "longbench_v2": "LongBenchV2",
            "longbenchv2": "LongBenchV2",
            "c-eval": "C-Eval",
            "c_eval": "C-Eval",
        }
        full_name = name_mapping.get(bench_name.lower(), bench_name)
        benches_to_test = [full_name] if full_name in available else []
    
    if not benches_to_test:
        print(f"❌ Benchmarker '{bench_name}' not found")
        print(f"Available: {', '.join(available.keys())}")
        return
    
    # Run tests
    for bench_name in benches_to_test:
        try:
            benchmarker = benchmarkers[bench_name]
            subdatasets = available[bench_name]
            test_dataset = subdatasets[1] if len(subdatasets) > 1 else subdatasets[0]
            
            print(f"\n📝 {bench_name}/{test_dataset}")
            
            # Test local
            try:
                model = create_mock_model()
                tokenizer = create_mock_tokenizer()
                model.device = "cpu"
                
                result = benchmarker.evaluate_local_llm(
                    model=model,
                    tokenizer=tokenizer,
                    subdataset_name=test_dataset
                )
                
                processed = result.get("metrics", {}).get("processed", 0)
                total = result.get("metrics", {}).get("total", 0)
                print(f"  ✓ Local: {processed}/{total} samples")
                
            except Exception as e:
                print(f"  ❌ Local: {str(e)}")
            
            # Test API
            try:
                client = create_mock_client()
                
                result = benchmarker.evaluate_api_llm(
                    client=client,
                    model="mock-gpt4",
                    subdataset_name=test_dataset
                )
                
                processed = result.get("metrics", {}).get("processed", 0)
                total = result.get("metrics", {}).get("total", 0)
                print(f"  ✓ API:   {processed}/{total} samples")
                
            except Exception as e:
                print(f"  ❌ API:   {str(e)}")
        
        except Exception as e:
            print(f"❌ {bench_name}: {str(e)}")
    
    print("\n✓ Done!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test for benchmarks")
    parser.add_argument("bench", nargs="?", default="all", 
                       help="Benchmarker to test (default: all)")
    parser.add_argument("--samples", type=int, default=3,
                       help="Number of samples (default: 3)")
    
    args = parser.parse_args()
    quick_test(args.bench, args.samples)
