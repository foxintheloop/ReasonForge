"""
Fix geom_003 and geom_004 setup steps.
"""

import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

# Define fixes for geom_003 and geom_004
FIXES = {
    "geom_003": {
        "setup_steps": [
            {
                "tool": "create_coordinate_system",
                "params": {"name": "C1", "type": "Cartesian"}  # Fixed: coord_type -> type, cartesian -> Cartesian
            },
            {
                "tool": "create_vector_field",
                "params": {"coord_system": "C1", "components": {"i": "y", "j": "-x", "k": "0"}}
            }
        ]
    },
    "geom_004": {
        "setup_steps": [
            {
                "tool": "create_coordinate_system",
                "params": {"name": "C1", "type": "Cartesian"}  # Fixed: coord_type -> type, cartesian -> Cartesian
            },
            {
                "tool": "create_vector_field",
                "params": {"coord_system": "C1", "components": {"i": "x", "j": "y", "k": "z"}}
            }
        ]
    }
}

# Apply fixes
fixed_count = 0

for test in TEST_CASES:
    test_id = test.get('id')

    if test_id in FIXES:
        fix = FIXES[test_id]

        # Update setup_steps
        if 'setup_steps' in fix:
            test['setup_steps'] = fix['setup_steps']

        fixed_count += 1
        print(f"Fixed {test_id}")

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

print(f"\nFixed {fixed_count} tests")
