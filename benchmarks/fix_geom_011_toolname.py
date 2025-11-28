"""Fix geom_011 - use correct tool name"""
import sys
sys.path.insert(0, 'benchmarks')

from test_cases import TEST_CASES

# Find and fix geom_011
for test in TEST_CASES:
    if test['id'] == 'geom_011':
        # Change tool name from quantity_convert_units to convert_to_units
        test['reasonforge_tool'] = 'convert_to_units'
        # Keep the rest as is - it should work now
        print(f"Fixed geom_011: Changed tool name to 'convert_to_units'")
        print(f"  Expected: {test['expected_answer']}")
        print(f"  Validation: {test['validation_type']}")
        break

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

print("Test cases updated successfully")
