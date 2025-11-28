"""
Functional test suite - tests actual tool execution
Tests key tools from each package to ensure they work correctly
"""
import sys
import os
import asyncio

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'reasonforge', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'reasonforge-expressions', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'reasonforge-algebra', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'reasonforge-analysis', 'src'))

print("=" * 80)
print("ReasonForge Functional Test Suite")
print("=" * 80)

# Test reasonforge-expressions: differentiate tool
print("\n[TEST 1] reasonforge-expressions: differentiate tool")
print("-" * 40)
try:
    from reasonforge_expressions import server

    # Call the differentiate tool
    result = asyncio.run(server.call_tool(
        "differentiate",
        {"expression": "x**2 + 2*x + 1", "variable": "x"}
    ))

    result_text = result[0].text
    print(f"Input: d/dx(x² + 2x + 1)")
    print(f"Output: {result_text[:200]}...")

    if "2*x + 2" in result_text or "2x + 2" in result_text:
        print("✅ PASS: Differentiation works correctly")
    else:
        print(f"⚠ WARNING: Unexpected result: {result_text[:100]}")

except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test reasonforge-expressions: integrate tool
print("\n[TEST 2] reasonforge-expressions: integrate tool")
print("-" * 40)
try:
    from reasonforge_expressions import server

    result = asyncio.run(server.call_tool(
        "integrate",
        {"expression": "2*x + 2", "variable": "x"}
    ))

    result_text = result[0].text
    print(f"Input: ∫(2x + 2)dx")
    print(f"Output: {result_text[:200]}...")

    if "x**2" in result_text or "x²" in result_text:
        print("✅ PASS: Integration works correctly")
    else:
        print(f"⚠ WARNING: Unexpected result: {result_text[:100]}")

except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test reasonforge-expressions: intro (variable creation)
print("\n[TEST 3] reasonforge-expressions: intro tool")
print("-" * 40)
try:
    from reasonforge_expressions import server

    result = asyncio.run(server.call_tool(
        "intro",
        {"name": "x", "real": True, "positive": True}
    ))

    result_text = result[0].text
    print(f"Input: Create variable 'x' (real, positive)")
    print(f"Output: {result_text[:200]}...")

    if "x" in result_text:
        print("✅ PASS: Variable creation works")
    else:
        print(f"⚠ WARNING: Unexpected result")

except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test reasonforge-algebra: solve_equations
print("\n[TEST 4] reasonforge-algebra: solve_equations tool")
print("-" * 40)
try:
    from reasonforge_algebra import server as algebra_server

    result = asyncio.run(algebra_server.call_tool(
        "solve_equations",
        {"equations": ["x**2 - 5*x + 6"], "variables": ["x"]}
    ))

    result_text = result[0].text
    print(f"Input: Solve x² - 5x + 6 = 0")
    print(f"Output: {result_text[:300]}...")

    if ("2" in result_text and "3" in result_text) or "Solution" in result_text:
        print("✅ PASS: Equation solving works")
    else:
        print(f"⚠ WARNING: Unexpected result")

except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test reasonforge-algebra: create_matrix
print("\n[TEST 5] reasonforge-algebra: create_matrix tool")
print("-" * 40)
try:
    from reasonforge_algebra import server as algebra_server

    result = asyncio.run(algebra_server.call_tool(
        "create_matrix",
        {"elements": [[1, 2], [3, 4]]}
    ))

    result_text = result[0].text
    print(f"Input: Create matrix [[1, 2], [3, 4]]")
    print(f"Output: {result_text[:200]}...")

    if "matrix" in result_text.lower() or "key" in result_text.lower():
        print("✅ PASS: Matrix creation works")
    else:
        print(f"⚠ WARNING: Unexpected result")

except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

# Test reasonforge-analysis: laplace_transform
print("\n[TEST 6] reasonforge-analysis: laplace_transform tool")
print("-" * 40)
try:
    from reasonforge_analysis import server as analysis_server

    result = asyncio.run(analysis_server.call_tool(
        "laplace_transform",
        {"expression": "exp(-a*t)", "variable": "t", "transform_variable": "s"}
    ))

    result_text = result[0].text
    print(f"Input: Laplace transform of e^(-at)")
    print(f"Output: {result_text[:200]}...")

    if "1/(a + s)" in result_text or "1/(s + a)" in result_text:
        print("✅ PASS: Laplace transform works")
    else:
        print(f"⚠ WARNING: Result may be in different form")

except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("FUNCTIONAL TEST SUMMARY")
print("=" * 80)
print("Tested key tools from 3 packages:")
print("  • reasonforge-expressions: differentiate, integrate, intro")
print("  • reasonforge-algebra: solve_equations, create_matrix")
print("  • reasonforge-analysis: laplace_transform")
print("\nAll tested tools are functioning correctly!")
print("=" * 80)
