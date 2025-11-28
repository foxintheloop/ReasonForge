"""Extract failure details from benchmark results"""
import json

with open('benchmarks/results/benchmark_results_20251122_204006.json', 'r') as f:
    results = json.load(f)

failed_tests = [
    "expr_013", "expr_015", "alg_003", "alg_004",
    "analysis_015", "geom_011", "stat_010"
]

print("Failed Test Details:")
print("=" * 80)

for result in results['detailed_results']:
    if result['test_id'] in failed_tests and not result['correct']:
        print(f"\nTest ID: {result['test_id']}")
        print(f"Explanation: {result['explanation']}")
