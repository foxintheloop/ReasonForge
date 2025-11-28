"""
Add stateful test support for geometry operations.

This script converts the geometry SKIP tests to stateful tests.
"""

import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

# Define setup and expected answers for geometry tests
GEOMETRY_TEST_CONFIGS = {
    "geom_002": {
        # Create vector field - needs coordinate system first
        "setup_steps": [
            {
                "tool": "create_coordinate_system",
                "params": {"name": "C1", "coord_type": "cartesian"}
            }
        ],
        "expected_answer": "field_key",  # Returns field key
        "validation_type": "contains_pattern"
    },
    "geom_003": {
        # Calculate curl - needs coordinate system and vector field
        "setup_steps": [
            {
                "tool": "create_coordinate_system",
                "params": {"name": "C1", "coord_type": "cartesian"}
            },
            {
                "tool": "create_vector_field",
                "params": {"coord_system": "C1", "components": {"i": "y", "j": "-x", "k": "0"}}
            }
        ],
        "expected_answer": "curl",
        "validation_type": "contains_pattern"
    },
    "geom_004": {
        # Calculate divergence - needs coordinate system and vector field
        "setup_steps": [
            {
                "tool": "create_coordinate_system",
                "params": {"name": "C1", "coord_type": "cartesian"}
            },
            {
                "tool": "create_vector_field",
                "params": {"coord_system": "C1", "components": {"i": "x", "j": "y", "k": "z"}}
            }
        ],
        "expected_answer": "divergence",
        "validation_type": "contains_pattern"
    },
    "geom_005": {
        # Calculate gradient - just needs scalar field
        "setup_steps": [],
        "expected_answer": "gradient",
        "validation_type": "contains_pattern"
    },
    "geom_009": {
        # Create custom metric - this IS the creation tool
        "setup_steps": [],
        "expected_answer": "metric_key",
        "validation_type": "contains_pattern"
    },
    "geom_010": {
        # Print LaTeX tensor - needs metric first
        "setup_steps": [
            {
                "tool": "create_custom_metric",
                "params": {"components": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "coordinates": ["x", "y", "z"]}
            }
        ],
        "expected_answer": "latex",
        "validation_type": "contains_pattern"
    },
    "geom_011": {
        # Unit conversion - already handled, keep as SKIP
        "skip": True
    },
    "geom_012": {
        # Simplify units - no setup needed
        "setup_steps": [],
        "expected_answer": "simplified",
        "validation_type": "contains_pattern"
    }
}

# Convert geometry tests
converted = []

for test in TEST_CASES:
    test_id = test.get('id')

    if test_id in GEOMETRY_TEST_CONFIGS:
        config = GEOMETRY_TEST_CONFIGS[test_id]

        # Skip if marked to keep as SKIP
        if config.get('skip'):
            print(f"Keeping {test_id} as SKIP")
            continue

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

print(f"\nConverted {len(converted)} geometry tests:")
for test_id in converted:
    print(f"  - {test_id}")
