"""
Convert SKIP tests to stateful tests with setup_steps.

This script identifies all tests marked as SKIP and converts them to
stateful tests with appropriate setup steps.
"""

import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

# Define setup requirements for each stateful tool category
STATEFUL_TEST_SETUPS = {
    # Matrix Operations
    "alg_007": [
        {"tool": "create_matrix", "params": {"elements": [[1, 2], [3, 4]], "key": "inv_test"}}
    ],
    "alg_008": [
        {"tool": "create_matrix", "params": {"elements": [[1, 2], [3, 4]], "key": "eigen_test"}}
    ],
    "alg_009": [
        {"tool": "create_matrix", "params": {"elements": [[1, 2], [3, 4]], "key": "eigvec_test"}}
    ],

    # Optimization (these don't actually need setup - just remove SKIP)
    "alg_010": [],
    "alg_011": [],

    # Vector Field Operations
    "geom_002": [
        {"tool": "create_coordinate_system", "params": {"name": "C1", "coord_type": "cartesian"}}
    ],
    "geom_003": [
        {"tool": "create_coordinate_system", "params": {"name": "C1", "coord_type": "cartesian"}},
        {"tool": "create_vector_field", "params": {"coord_system": "C1", "components": {"i": "y", "j": "-x", "k": "0"}}}
    ],
    "geom_004": [
        {"tool": "create_coordinate_system", "params": {"name": "C1", "coord_type": "cartesian"}},
        {"tool": "create_vector_field", "params": {"coord_system": "C1", "components": {"i": "y", "j": "-x", "k": "0"}}}
    ],
    "geom_005": [],  # Doesn't need field creation

    # Metric/Tensor Operations
    "geom_009": [],  # This IS the creation tool
    "geom_010": [
        {"tool": "create_custom_metric", "params": {"components": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "coordinates": ["x", "y", "z"]}}
    ],
    "geom_011": [],  # Unit conversion - different issue
    "geom_012": [],  # Simplify units

    # Quantum State Operations
    "phys_006": [],  # This IS the creation tool
    "phys_007": [
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}}
    ],
    "phys_008": [
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}},
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [0, 1]}}
    ],
    "phys_009": [
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}}
    ],
    "phys_011": [
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}}
    ],
    "phys_012": [
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}},
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}}
    ],
    "phys_013": [],  # Pauli matrices - doesn't need setup
    "phys_015": [
        {"tool": "create_quantum_state", "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}}
    ],
}

# Convert tests
converted_count = 0
removed_skip_count = 0

for test in TEST_CASES:
    test_id = test['id']

    # Check if this test is marked as SKIP
    if test.get('expected_answer') == 'SKIP':
        # Check if we have setup steps defined for this test
        if test_id in STATEFUL_TEST_SETUPS:
            setup_steps = STATEFUL_TEST_SETUPS[test_id]

            if setup_steps:
                # Add setup_steps
                test['setup_steps'] = setup_steps
                converted_count += 1

                # Update expected answer based on tool
                # For now, use "exists" as a placeholder - will need manual adjustment
                test['expected_answer'] = 'exists'
                test['validation_type'] = 'contains_pattern'

                print(f"Converted {test_id} to stateful test with {len(setup_steps)} setup step(s)")
            else:
                # No setup needed - just remove SKIP
                test['expected_answer'] = 'result'
                test['validation_type'] = 'contains_pattern'
                removed_skip_count += 1

                print(f"Removed SKIP from {test_id} (no setup needed)")

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

print(f"\nSummary:")
print(f"  Converted to stateful: {converted_count}")
print(f"  Removed SKIP (no setup): {removed_skip_count}")
print(f"  Total processed: {converted_count + removed_skip_count}")
