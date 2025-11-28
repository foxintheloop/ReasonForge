"""Check test definitions for failed tests"""
import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

failed_tests = [
    "expr_013", "expr_015", "alg_003", "alg_004",
    "analysis_015", "geom_011", "stat_010"
]

print("Current Test Definitions:")
print("=" * 80)

for test in TEST_CASES:
    if test['id'] in failed_tests:
        print(f"\nTest ID: {test['id']}")
        print(f"Expected Answer: {test.get('expected_answer', 'N/A')}")
        print(f"Validation Type: {test.get('validation_type', 'N/A')}")
        print(f"Response Field: {test.get('response_field', 'N/A')}")
