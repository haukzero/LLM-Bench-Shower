#!/usr/bin/env python3
"""
Fast testing script for LLM Benchmark classes using mock LLM.

This script allows quick testing of benchmarker implementations without
requiring actual model loading or API calls. It uses mock implementations
of transformers models and OpenAI clients.

Usage:
    python test_benchmarks.py                    # Test all benchmarks
    python test_benchmarks.py --bench longbench  # Test specific benchmark
    python test_benchmarks.py --samples 10       # Test with 10 samples
    python test_benchmarks.py --cleanup          # Clean up test data after test
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from pprint import pprint

# Add parent directory to path
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
    cleanup_test_data
)

# Import benchmarkers
from bench import init_all_benchmarkers
from bench.utils import get_available_datasets, get_dataset_path, get_sub_datasets


class BenchmarkTester:
    """Main testing class for benchmarkers with mock LLM."""
    
    def __init__(self, test_data_dir: str = None, num_samples: int = 5, use_mock: bool = True):
        """Initialize benchmark tester.
        
        Args:
            test_data_dir: Path to test data directory
            num_samples: Number of samples to test per dataset
            use_mock: Whether to use mock LLM (True) or real models (False)
        """
        self.test_data_dir = test_data_dir or os.path.join(
            os.path.dirname(__file__), "test_data"
        )
        self.num_samples = num_samples
        self.use_mock = use_mock
        self.results = {}
        
    def setup_test_environment(self):
        """Setup test datasets and configs."""
        print("\n" + "=" * 70)
        print("🚀 Setting up test environment")
        print("=" * 70)
        
        # Create mock datasets
        test_dir = setup_test_datasets(
            test_data_dir=self.test_data_dir,
            num_samples=self.num_samples
        )
        
        # Update config files to point to test data
        update_config_for_testing(test_data_dir=test_dir)
        
        return test_dir
    
    def test_local_llm_evaluation(self, benchmarker_name: str, subdataset_name: str) -> Dict:
        """Test local LLM evaluation.
        
        Args:
            benchmarker_name: Name of the benchmarker (e.g., "LongBench")
            subdataset_name: Name of the subdataset to test
            
        Returns:
            Evaluation results
        """
        print(f"\n📝 Testing local LLM evaluation: {benchmarker_name}/{subdataset_name}")
        
        try:
            # Get benchmarker
            benchmarkers = init_all_benchmarkers()
            benchmarker = benchmarkers.get(benchmarker_name)
            
            if not benchmarker:
                print(f"❌ Benchmarker '{benchmarker_name}' not found")
                return {"error": f"Benchmarker not found: {benchmarker_name}"}
            
            # Create mock model and tokenizer
            model = create_mock_model(model_name="mock-model")
            tokenizer = create_mock_tokenizer()
            
            # Set model device attribute
            model.device = "cpu"
            
            # Run evaluation
            results = benchmarker.evaluate_local_llm(
                model=model,
                tokenizer=tokenizer,
                subdataset_name=subdataset_name
            )
            
            print(f"✓ Local LLM evaluation completed")
            print(f"  - Total samples: {results['metrics']['total']}")
            print(f"  - Processed: {results['metrics']['processed']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during local LLM evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def test_api_llm_evaluation(self, benchmarker_name: str, subdataset_name: str) -> Dict:
        """Test API LLM evaluation.
        
        Args:
            benchmarker_name: Name of the benchmarker
            subdataset_name: Name of the subdataset to test
            
        Returns:
            Evaluation results
        """
        print(f"\n📡 Testing API LLM evaluation: {benchmarker_name}/{subdataset_name}")
        
        try:
            # Get benchmarker
            benchmarkers = init_all_benchmarkers()
            benchmarker = benchmarkers.get(benchmarker_name)
            
            if not benchmarker:
                print(f"❌ Benchmarker '{benchmarker_name}' not found")
                return {"error": f"Benchmarker not found: {benchmarker_name}"}
            
            # Create mock client
            client = create_mock_client(api_key="test-key")
            
            # Run evaluation
            results = benchmarker.evaluate_api_llm(
                client=client,
                model="mock-gpt-4",
                subdataset_name=subdataset_name
            )
            
            print(f"✓ API LLM evaluation completed")
            print(f"  - Total samples: {results['metrics']['total']}")
            print(f"  - Processed: {results['metrics']['processed']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during API LLM evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def test_benchmarker(self, benchmarker_name: str, test_mode: str = "all") -> Dict:
        """Test a specific benchmarker.
        
        Args:
            benchmarker_name: Name of the benchmarker to test
            test_mode: "local", "api", or "all"
            
        Returns:
            Test results
        """
        print(f"\n{'=' * 70}")
        print(f"Testing Benchmarker: {benchmarker_name}")
        print(f"{'=' * 70}")
        
        results = {
            "benchmarker": benchmarker_name,
            "local_tests": {},
            "api_tests": {}
        }
        
        try:
            # Get available subdatasets
            available = get_available_datasets()
            if benchmarker_name not in available:
                print(f"❌ Benchmarker '{benchmarker_name}' not found in available datasets")
                return results
            
            subdatasets = available[benchmarker_name]
            # Test first subdataset (or specified one)
            #print("subdatasets:", subdatasets)
            test_subdataset = subdatasets[1] if len(subdatasets) > 1 else subdatasets[0]
            
            print(f"Available subdatasets: {len(subdatasets)}")
            print(f"Testing subdataset: {test_subdataset}")
            
            # Test local evaluation
            if test_mode in ["local", "all"]:
                try:
                    results["local_tests"][test_subdataset] = self.test_local_llm_evaluation(
                        benchmarker_name, test_subdataset
                    )
                except Exception as e:
                    results["local_tests"][test_subdataset] = {"error": str(e)}
            
            # Test API evaluation
            if test_mode in ["api", "all"]:
                try:
                    results["api_tests"][test_subdataset] = self.test_api_llm_evaluation(
                        benchmarker_name, test_subdataset
                    )
                except Exception as e:
                    results["api_tests"][test_subdataset] = {"error": str(e)}
            
        except Exception as e:
            print(f"❌ Error testing benchmarker {benchmarker_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
        
        return results
    
    def run_all_tests(self, test_mode: str = "local") -> Dict:
        """Run tests for all available benchmarkers.
        
        Args:
            test_mode: "local", "api", or "all"
            
        Returns:
            All test results
        """
        try:
            available = get_available_datasets()
            print(f"\n📊 Found {len(available)} available benchmarkers:")
            for name in available.keys():
                print(f"   • {name}")
            
            all_results = {}
            for benchmarker_name in available.keys():
                results = self.test_benchmarker(benchmarker_name, test_mode)
                all_results[benchmarker_name] = results
            
            return all_results
            
        except Exception as e:
            print(f"❌ Error running tests: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def print_summary(self, results: Dict):
        """Print test results summary.
        
        Args:
            results: Test results dictionary
        """
        print(f"\n{'=' * 70}")
        print("📊 TEST SUMMARY")
        print(f"{'=' * 70}")
        
        if "error" in results:
            print(f"❌ Overall error: {results['error']}")
            return
        
        for bench_name, bench_results in results.items():
            if "error" in bench_results:
                print(f"\n❌ {bench_name}: {bench_results['error']}")
            else:
                print(f"\n✓ {bench_name}")
                
                # Local tests
                if bench_results.get("local_tests"):
                    print(f"  Local LLM tests:")
                    for dataset, test_result in bench_results["local_tests"].items():
                        if "error" in test_result:
                            print(f"    ❌ {dataset}: {test_result['error']}")
                        else:
                            processed = test_result.get("metrics", {}).get("processed", 0)
                            total = test_result.get("metrics", {}).get("total", 0)
                            print(f"    ✓ {dataset}: {processed}/{total} samples processed")
                
                # API tests
                if bench_results.get("api_tests"):
                    print(f"  API LLM tests:")
                    for dataset, test_result in bench_results["api_tests"].items():
                        if "error" in test_result:
                            print(f"    ❌ {dataset}: {test_result['error']}")
                        else:
                            processed = test_result.get("metrics", {}).get("processed", 0)
                            total = test_result.get("metrics", {}).get("total", 0)
                            print(f"    ✓ {dataset}: {processed}/{total} samples processed")
    
    def save_results(self, results: Dict, output_file: str = "test_results.json"):
        """Save test results to file.
        
        Args:
            results: Test results dictionary
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")


def main():
    """Main entry point for test script."""
    parser = argparse.ArgumentParser(
        description="Fast testing script for LLM Benchmark classes using mock LLM"
    )
    parser.add_argument(
        "--bench",
        help="Specific benchmarker to test (e.g., 'LongBench', 'LongBenchV2')",
        default=None
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples per subdataset",
        default=5
    )
    parser.add_argument(
        "--mode",
        choices=["local", "api", "all"],
        help="Test mode: local models, API models, or both",
        default="local"
    )
    parser.add_argument(
        "--output",
        help="Output file for results (JSON format)",
        default="test_results.json"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test data after tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed test output"
    )
    
    args = parser.parse_args()
    
    # Create tester
    tester = BenchmarkTester(
        num_samples=args.samples,
        use_mock=True
    )
    
    try:
        # Setup environment
        test_dir = tester.setup_test_environment()
        
        # Run tests
        if args.bench:
            results = tester.test_benchmarker(args.bench, test_mode=args.mode)
            all_results = {args.bench: results}
        else:
            all_results = tester.run_all_tests(test_mode=args.mode)
        
        # Print summary
        tester.print_summary(all_results)
        
        # Save results
        tester.save_results(all_results, args.output)
        
    finally:
        # Cleanup if requested
        if args.cleanup:
            cleanup_test_data()
    
    print("\n✓ Testing complete!\n")


if __name__ == "__main__":
    main()
