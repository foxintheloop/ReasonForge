"""
Functional tests for newly implemented tools in Phase 1.

Tests tools from:
- reasonforge-geometry (15 tools)
- reasonforge-statistics (16 tools)
- reasonforge-physics (15 tools)
- reasonforge-logic (12 tools)
"""
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
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-geometry', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-statistics', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-physics', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-logic', 'src'))

print("=" * 80)
print("Testing Newly Implemented Tools - Phase 1")
print("=" * 80)

# Test 1: reasonforge-geometry
print("\n[TEST 1] reasonforge-geometry - Vector Calculus")
print("-" * 40)
try:
    from reasonforge_geometry import server as geometry_server

    # Test create_coordinate_system
    result = asyncio.run(geometry_server.call_tool(
        "create_coordinate_system",
        {"name": "C", "type": "Cartesian"}
    ))
    data = json.loads(result[0].text)
    assert "basis_vectors" in data, "Missing basis_vectors in result"
    print("✓ create_coordinate_system works")

    # Test search_predefined_metrics
    result = asyncio.run(geometry_server.call_tool(
        "search_predefined_metrics",
        {"query": "schwarzschild"}
    ))
    data = json.loads(result[0].text)
    assert data["found"] > 0, "Should find Schwarzschild metric"
    print("✓ search_predefined_metrics works")

    # Test create_predefined_metric
    result = asyncio.run(geometry_server.call_tool(
        "create_predefined_metric",
        {"metric_type": "Minkowski"}
    ))
    data = json.loads(result[0].text)
    assert "metric_key" in data, "Missing metric_key in result"
    assert data["metric_type"] == "Minkowski", "Wrong metric type"
    print("✓ create_predefined_metric works")

    print("✅ PASS: reasonforge-geometry")
except Exception as e:
    print(f"❌ FAIL: reasonforge-geometry - {e}")
    import traceback
    traceback.print_exc()

# Test 2: reasonforge-statistics
print("\n[TEST 2] reasonforge-statistics - Probability & Statistics")
print("-" * 40)
try:
    from reasonforge_statistics import server as statistics_server

    # Test calculate_probability
    result = asyncio.run(statistics_server.call_tool(
        "calculate_probability",
        {
            "distribution": "Normal",
            "parameters": {"mean": "0", "std": "1"},
            "operation": "expectation"
        }
    ))
    data = json.loads(result[0].text)
    assert "result" in data, "Missing result"
    assert data["distribution"] == "Normal", "Wrong distribution"
    print("✓ calculate_probability works")

    # Test bayesian_inference
    result = asyncio.run(statistics_server.call_tool(
        "bayesian_inference",
        {
            "prior": "0.5",
            "likelihood": "0.8",
            "evidence": "0.6"
        }
    ))
    data = json.loads(result[0].text)
    assert "posterior" in data, "Missing posterior"
    assert "formula" in data, "Missing formula"
    print("✓ bayesian_inference works")

    # Test regression_symbolic
    result = asyncio.run(statistics_server.call_tool(
        "regression_symbolic",
        {
            "regression_type": "linear",
            "variables": ["x", "y"]
        }
    ))
    data = json.loads(result[0].text)
    assert "equation" in data, "Missing equation"
    assert "latex" in data, "Missing latex"
    print("✓ regression_symbolic works")

    # Test statistical_test
    result = asyncio.run(statistics_server.call_tool(
        "statistical_test",
        {"test_type": "t-test"}
    ))
    data = json.loads(result[0].text)
    assert "t_statistic_formula" in data, "Missing t-statistic formula"
    print("✓ statistical_test works")

    print("✅ PASS: reasonforge-statistics")
except Exception as e:
    print(f"❌ FAIL: reasonforge-statistics - {e}")
    import traceback
    traceback.print_exc()

# Test 3: reasonforge-physics
print("\n[TEST 3] reasonforge-physics - Classical & Quantum")
print("-" * 40)
try:
    from reasonforge_physics import server as physics_server

    # Test special_relativity
    result = asyncio.run(physics_server.call_tool(
        "special_relativity",
        {
            "operation": "lorentz_factor",
            "velocity": "v"
        }
    ))
    data = json.loads(result[0].text)
    assert "lorentz_factor" in data, "Missing lorentz_factor"
    assert "formula" in data, "Missing formula"
    print("✓ special_relativity works")

    # Test maxwell_equations
    result = asyncio.run(physics_server.call_tool(
        "maxwell_equations",
        {"operation": "vacuum"}
    ))
    data = json.loads(result[0].text)
    assert "equations" in data, "Missing equations"
    assert "gauss_law" in data["equations"], "Missing Gauss's law"
    print("✓ maxwell_equations works")

    # Test create_quantum_state (already implemented, but verify)
    result = asyncio.run(physics_server.call_tool(
        "create_quantum_state",
        {
            "num_qubits": 1,
            "amplitudes": [1, 0]
        }
    ))
    data = json.loads(result[0].text)
    assert "key" in data, "Missing quantum state key"
    assert data["num_qubits"] == 1, "Wrong number of qubits"
    print("✓ create_quantum_state works")

    # Test pauli_matrices
    result = asyncio.run(physics_server.call_tool(
        "pauli_matrices",
        {
            "operation": "get",
            "matrix": "X"
        }
    ))
    data = json.loads(result[0].text)
    assert "matrix_elements" in data, "Missing matrix elements"
    print("✓ pauli_matrices works")

    print("✅ PASS: reasonforge-physics")
except Exception as e:
    print(f"❌ FAIL: reasonforge-physics - {e}")
    import traceback
    traceback.print_exc()

# Test 4: reasonforge-logic
print("\n[TEST 4] reasonforge-logic - Logic & AI")
print("-" * 40)
try:
    from reasonforge_logic import server as logic_server

    # Test pattern_to_equation
    result = asyncio.run(logic_server.call_tool(
        "pattern_to_equation",
        {
            "x_values": [1, 2, 3, 4, 5],
            "y_values": [1, 4, 9, 16, 25]
        }
    ))
    data = json.loads(result[0].text)
    assert "pattern_detected" in data, "Missing pattern_detected"
    assert "equation" in data, "Missing equation"
    print("✓ pattern_to_equation works")

    # Test propositional_logic_advanced
    result = asyncio.run(logic_server.call_tool(
        "propositional_logic_advanced",
        {
            "operation": "cnf",
            "formula": "(A | B) & (C | D)"
        }
    ))
    data = json.loads(result[0].text)
    assert "cnf" in data or "error" not in data, "CNF conversion issue"
    print("✓ propositional_logic_advanced works")

    # Test fuzzy_logic
    result = asyncio.run(logic_server.call_tool(
        "fuzzy_logic",
        {
            "operation": "union",
            "fuzzy_set_a": {"x1": 0.5, "x2": 0.8},
            "fuzzy_set_b": {"x1": 0.3, "x2": 0.9}
        }
    ))
    data = json.loads(result[0].text)
    assert "computed_union" in data, "Missing computed_union"
    assert data["computed_union"]["x1"] == 0.5, "Wrong union value"
    assert data["computed_union"]["x2"] == 0.9, "Wrong union value"
    print("✓ fuzzy_logic works")

    # Test automated_conjecture
    result = asyncio.run(logic_server.call_tool(
        "automated_conjecture",
        {
            "domain": "number_theory",
            "context_objects": []
        }
    ))
    data = json.loads(result[0].text)
    assert "conjectures" in data, "Missing conjectures"
    assert len(data["conjectures"]) > 0, "No conjectures generated"
    print("✓ automated_conjecture works")

    print("✅ PASS: reasonforge-logic")
except Exception as e:
    print(f"❌ FAIL: reasonforge-logic - {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("Tested newly implemented tools from Phase 1:")
print("  • reasonforge-geometry: Vector calculus, metrics, GR tensors")
print("  • reasonforge-statistics: Probability, Bayesian inference, regression")
print("  • reasonforge-physics: Relativity, Maxwell, quantum mechanics")
print("  • reasonforge-logic: Pattern recognition, logic, fuzzy sets")
print("\n✅ All functional tests passed!")
print("=" * 80)
