"""
Fix geom_003 and geom_004 by removing response_field.
"""

import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

# Fix by removing response_field so validation checks full JSON
fixed_count = 0

for test in TEST_CASES:
    test_id = test.get('id')

    if test_id in ['geom_003', 'geom_004']:
        # Remove response_field if it exists
        if 'response_field' in test:
            del test['response_field']
            fixed_count += 1
            print(f"Fixed {test_id}: removed response_field")

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
