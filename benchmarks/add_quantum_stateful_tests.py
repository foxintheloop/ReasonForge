"""
Add stateful test support for quantum operations.

This script converts the quantum operation SKIP tests to stateful tests.
"""

import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

# Define setup and expected answers for quantum tests
QUANTUM_TEST_CONFIGS = {
    "phys_006": {
        # This IS the creation tool - no setup needed, just remove SKIP
        "setup_steps": [],
        "expected_answer": "state_key",  # Returns a state key
        "validation_type": "contains_pattern"
    },
    "phys_007": {
        # Quantum gate operations - needs a state first
        "setup_steps": [
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
            }
        ],
        "expected_answer": "result_state",  # Check for result state in response
        "validation_type": "contains_pattern"
    },
    "phys_008": {
        # Tensor product - needs two states
        "setup_steps": [
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
            },
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [0, 1]}
            }
        ],
        "expected_answer": "tensor_product",
        "validation_type": "contains_pattern"
    },
    "phys_009": {
        # Entanglement measure - needs a state
        "setup_steps": [
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 2, "amplitudes": [0.707, 0, 0, 0.707]}
            }
        ],
        "expected_answer": "entanglement_measure",
        "validation_type": "contains_pattern"
    },
    "phys_011": {
        # Quantum measurement - needs a state
        "setup_steps": [
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
            }
        ],
        "expected_answer": "probabilities",
        "validation_type": "contains_pattern"
    },
    "phys_012": {
        # Quantum fidelity - needs two states
        "setup_steps": [
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
            },
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
            }
        ],
        "expected_answer": "fidelity",
        "validation_type": "contains_pattern"
    },
    "phys_013": {
        # Pauli matrices - no state needed, just returns matrices
        "setup_steps": [],
        "expected_answer": "matrix",
        "validation_type": "contains_pattern"
    },
    "phys_015": {
        # Quantum evolution - needs state
        "setup_steps": [
            {
                "tool": "create_quantum_state",
                "params": {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
            }
        ],
        "expected_answer": "evolved_state",
        "validation_type": "contains_pattern"
    }
}

# Convert quantum tests
converted = []

for test in TEST_CASES:
    test_id = test.get('id')

    if test_id in QUANTUM_TEST_CONFIGS:
        config = QUANTUM_TEST_CONFIGS[test_id]

        # Add setup_steps if any
        if config['setup_steps']:
            test['setup_steps'] = config['setup_steps']

        # Update expected answer and validation
        test['expected_answer'] = config['expected_answer']
        test['validation_type'] = config['validation_type']

        converted.append(test_id)
        print(f"Converted {test_id} to {'stateful' if config['setup_steps'] else 'stateless'} test")

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

print(f"\nConverted {len(converted)} quantum tests:")
for test_id in converted:
    print(f"  - {test_id}")
