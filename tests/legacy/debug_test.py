"""Quick debug test to see actual responses"""
import sys
import os
import asyncio
import json

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add packages to path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-statistics', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-physics', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-logic', 'src'))

print("=" * 80)
print("Debug Test - Check Actual Responses")
print("=" * 80)

# Test 1: Statistics
print("\n[TEST 1] Statistics - calculate_probability")
print("-" * 40)
try:
    from reasonforge_statistics import server as statistics_server

    result = asyncio.run(statistics_server.call_tool(
        "calculate_probability",
        {
            "distribution": "Normal",
            "parameters": {"mean": "0", "std": "1"},
            "operation": "expectation"
        }
    ))
    print("Response:", result[0].text)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Physics
print("\n[TEST 2] Physics - create_quantum_state")
print("-" * 40)
try:
    from reasonforge_physics import server as physics_server

    result = asyncio.run(physics_server.call_tool(
        "create_quantum_state",
        {
            "num_qubits": 1,
            "amplitudes": [1, 0]
        }
    ))
    print("Response:", result[0].text)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Logic
print("\n[TEST 3] Logic - pattern_to_equation")
print("-" * 40)
try:
    from reasonforge_logic import server as logic_server

    result = asyncio.run(logic_server.call_tool(
        "pattern_to_equation",
        {
            "x_values": [1, 2, 3, 4, 5],
            "y_values": [1, 4, 9, 16, 25]
        }
    ))
    print("Response:", result[0].text)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
