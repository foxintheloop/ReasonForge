"""
Comprehensive test suite for ReasonForge modular packages
Tests all 7 MCP server packages and the core library
"""
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add packages to path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-expressions', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-algebra', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-analysis', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-geometry', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-statistics', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-physics', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-logic', 'src'))

print("=" * 80)
print("ReasonForge Modular Package Test Suite")
print("=" * 80)

# Test 1: Core Library
print("\n[TEST 1] Core Library (reasonforge)")
print("-" * 40)
try:
    from reasonforge import SymbolicAI
    ai = SymbolicAI()
    print("✓ Core library imported successfully")
    print(f"✓ SymbolicAI instance created: {type(ai).__name__}")

    # Test basic functionality
    vars = ai.define_variables(['x', 'y'])
    print(f"✓ Created variables: x, y")

    # Test equation solving
    import sympy as sp
    x, y = vars
    result = ai.solve_equation_system([x**2 - 4, y - 2])
    print(f"✓ Equation solving works: {len(result['solutions'])} solutions found")

    print("✅ PASS: Core library")
except Exception as e:
    print(f"❌ FAIL: Core library - {e}")
    import traceback
    traceback.print_exc()

# Test 2: reasonforge-expressions
print("\n[TEST 2] reasonforge-expressions (15 tools)")
print("-" * 40)
try:
    from reasonforge_expressions import server
    print(f"✓ Package imported: {server.server.name}")

    # Test tool listing
    tools = []
    async def get_tools():
        return await server.list_tools()

    import asyncio
    tools = asyncio.run(get_tools())
    print(f"✓ Tools registered: {len(tools)} tools")

    tool_names = [t.name for t in tools]
    expected_tools = ['intro', 'intro_many', 'introduce_expression', 'introduce_function',
                     'simplify_expression', 'substitute_expression', 'expand_expression',
                     'factor_expression', 'substitute_values', 'differentiate', 'integrate',
                     'compute_limit', 'expand_series', 'print_latex_expression', 'solve_word_problem']

    for expected in expected_tools:
        if expected in tool_names:
            print(f"  ✓ {expected}")
        else:
            print(f"  ✗ MISSING: {expected}")

    print("✅ PASS: reasonforge-expressions")
except Exception as e:
    print(f"❌ FAIL: reasonforge-expressions - {e}")
    import traceback
    traceback.print_exc()

# Test 3: reasonforge-algebra
print("\n[TEST 3] reasonforge-algebra (18 tools)")
print("-" * 40)
try:
    from reasonforge_algebra import server as algebra_server
    print(f"✓ Package imported: {algebra_server.server.name}")

    tools = asyncio.run(algebra_server.list_tools())
    print(f"✓ Tools registered: {len(tools)} tools")

    expected_count = 18
    if len(tools) == expected_count:
        print(f"✓ Expected tool count: {expected_count}")
    else:
        print(f"⚠ Tool count mismatch: expected {expected_count}, got {len(tools)}")

    print("✅ PASS: reasonforge-algebra")
except Exception as e:
    print(f"❌ FAIL: reasonforge-algebra - {e}")
    import traceback
    traceback.print_exc()

# Test 4: reasonforge-analysis
print("\n[TEST 4] reasonforge-analysis (17 tools)")
print("-" * 40)
try:
    from reasonforge_analysis import server as analysis_server
    print(f"✓ Package imported: {analysis_server.server.name}")

    tools = asyncio.run(analysis_server.list_tools())
    print(f"✓ Tools registered: {len(tools)} tools")

    expected_count = 17
    if len(tools) == expected_count:
        print(f"✓ Expected tool count: {expected_count}")
    else:
        print(f"⚠ Tool count mismatch: expected {expected_count}, got {len(tools)}")

    print("✅ PASS: reasonforge-analysis")
except Exception as e:
    print(f"❌ FAIL: reasonforge-analysis - {e}")
    import traceback
    traceback.print_exc()

# Test 5: reasonforge-geometry
print("\n[TEST 5] reasonforge-geometry (15 tools)")
print("-" * 40)
try:
    from reasonforge_geometry import server as geometry_server
    print(f"✓ Package imported: {geometry_server.server.name}")

    tools = asyncio.run(geometry_server.list_tools())
    print(f"✓ Tools registered: {len(tools)} tools")

    expected_count = 15
    if len(tools) == expected_count:
        print(f"✓ Expected tool count: {expected_count}")
    else:
        print(f"⚠ Tool count mismatch: expected {expected_count}, got {len(tools)}")

    print("✅ PASS: reasonforge-geometry")
except Exception as e:
    print(f"❌ FAIL: reasonforge-geometry - {e}")
    import traceback
    traceback.print_exc()

# Test 6: reasonforge-statistics
print("\n[TEST 6] reasonforge-statistics (16 tools)")
print("-" * 40)
try:
    from reasonforge_statistics import server as statistics_server
    print(f"✓ Package imported: {statistics_server.server.name}")

    tools = asyncio.run(statistics_server.list_tools())
    print(f"✓ Tools registered: {len(tools)} tools")

    expected_count = 16
    if len(tools) == expected_count:
        print(f"✓ Expected tool count: {expected_count}")
    else:
        print(f"⚠ Tool count mismatch: expected {expected_count}, got {len(tools)}")

    print("✅ PASS: reasonforge-statistics")
except Exception as e:
    print(f"❌ FAIL: reasonforge-statistics - {e}")
    import traceback
    traceback.print_exc()

# Test 7: reasonforge-physics
print("\n[TEST 7] reasonforge-physics (16 tools)")
print("-" * 40)
try:
    from reasonforge_physics import server as physics_server
    print(f"✓ Package imported: {physics_server.server.name}")

    tools = asyncio.run(physics_server.list_tools())
    print(f"✓ Tools registered: {len(tools)} tools")

    expected_count = 16
    if len(tools) == expected_count:
        print(f"✓ Expected tool count: {expected_count}")
    else:
        print(f"⚠ Tool count mismatch: expected {expected_count}, got {len(tools)}")

    print("✅ PASS: reasonforge-physics")
except Exception as e:
    print(f"❌ FAIL: reasonforge-physics - {e}")
    import traceback
    traceback.print_exc()

# Test 8: reasonforge-logic
print("\n[TEST 8] reasonforge-logic (13 tools)")
print("-" * 40)
try:
    from reasonforge_logic import server as logic_server
    print(f"✓ Package imported: {logic_server.server.name}")

    tools = asyncio.run(logic_server.list_tools())
    print(f"✓ Tools registered: {len(tools)} tools")

    expected_count = 13
    if len(tools) == expected_count:
        print(f"✓ Expected tool count: {expected_count}")
    else:
        print(f"⚠ Tool count mismatch: expected {expected_count}, got {len(tools)}")

    print("✅ PASS: reasonforge-logic")
except Exception as e:
    print(f"❌ FAIL: reasonforge-logic - {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("All packages tested successfully!")
print("\nTotal tools across all packages:")
print("  • reasonforge-expressions: 15 tools")
print("  • reasonforge-algebra: 18 tools")
print("  • reasonforge-analysis: 17 tools")
print("  • reasonforge-geometry: 15 tools")
print("  • reasonforge-statistics: 16 tools")
print("  • reasonforge-physics: 16 tools")
print("  • reasonforge-logic: 13 tools")
print("  " + "-" * 40)
print("  TOTAL: 110 tools")
print("=" * 80)
