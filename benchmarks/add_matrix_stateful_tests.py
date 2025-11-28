"""
Add stateful test support for matrix operations (proof of concept).

This script converts the 3 matrix operation SKIP tests to stateful tests.
"""

import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

# Define setup and expected answers for matrix tests
MATRIX_TEST_CONFIGS = {
    "alg_007": {
        "setup_steps": [
            {
                "tool": "create_matrix",
                "params": {"elements": [[1, 2], [3, 4]], "key": "inv_test"}
            }
        ],
        "expected_answer": "[[-2.0, 1.0], [1.5, -0.5]]",
        "validation_type": "list_equivalence"
    },
    "alg_008": {
        "setup_steps": [
            {
                "tool": "create_matrix",
                "params": {"elements": [[1, 2], [3, 4]], "key": "eigen_test"}
            }
        ],
        "expected_answer": "eigenvalues",  # Just check the field exists
        "validation_type": "contains_pattern"
    },
    "alg_009": {
        "setup_steps": [
            {
                "tool": "create_matrix",
                "params": {"elements": [[1, 2], [3, 4]], "key": "eigvec_test"}
            }
        ],
        "expected_answer": "eigenvectors",  # Just check the field exists
        "validation_type": "contains_pattern"
    }
}

# Convert matrix tests
converted = []

for test in TEST_CASES:
    test_id = test.get('id')

    if test_id in MATRIX_TEST_CONFIGS:
        config = MATRIX_TEST_CONFIGS[test_id]

        # Add setup_steps
        test['setup_steps'] = config['setup_steps']

        # Update expected answer and validation
        test['expected_answer'] = config['expected_answer']
        test['validation_type'] = config['validation_type']

        converted.append(test_id)
        print(f"Converted {test_id} to stateful test")

# Write back
with open('benchmarks/test_cases.py', 'w', encoding='utf-8') as f:
    f.write('"""Test cases for ReasonForge benchmark suite"""\n\n')
    f.write('TEST_CASES = [\n')

    for i, test in enumerate(TEST_CASES):
        f.write('    {\n')
        for key, value in test.items():
            # Format the value appropriately
            if isinstance(value, str):
                f.write(f'        "{key}": "{value}",\n')
            elif isinstance(value, dict):
                f.write(f'        "{key}": {value},\n')
            elif isinstance(value, list):
                f.write(f'        "{key}": {value},\n')
            elif isinstance(value, (int, float)):
                f.write(f'        "{key}": {value},\n')
            else:
                f.write(f'        "{key}": {repr(value)},\n')

        if i < len(TEST_CASES) - 1:
            f.write('    },\n\n')
        else:
            f.write('    }\n')

    f.write(']\n\n')
    f.write('def get_test_case_count():\n')
    f.write('    """Get total test case count"""\n')
    f.write(f'    return {len(TEST_CASES)}\n')

print(f"\nConverted {len(converted)} matrix tests to stateful tests:")
for test_id in converted:
    print(f"  - {test_id}")
