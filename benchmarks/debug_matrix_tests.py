"""Debug matrix stateful tests to see actual responses"""
import asyncio
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-algebra" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge" / "src"))

from reasonforge_algebra.server import server as algebra_server

async def test_matrix_operations():
    """Test matrix operations with setup"""

    # Test 1: Matrix Inverse
    print("=" * 80)
    print("Test 1: Matrix Inverse (alg_007)")
    print("=" * 80)

    # Setup: Create matrix
    setup_result = await algebra_server.call_tool_for_test(
        "create_matrix",
        {"elements": [[1, 2], [3, 4]], "key": "inv_test"}
    )
    print(f"Setup result: {setup_result[0].text if hasattr(setup_result[0], 'text') else setup_result}")

    # Test: Calculate inverse
    test_result = await algebra_server.call_tool_for_test(
        "matrix_inverse",
        {"matrix_key": "inv_test"}
    )
    content = test_result[0].text if hasattr(test_result[0], 'text') else str(test_result)
    print(f"Test result: {content}")

    try:
        parsed = json.loads(content)
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
        print(f"Inverse field: {parsed.get('inverse', 'NOT FOUND')}")
    except:
        print("Could not parse as JSON")

    print()

    # Test 2: Eigenvalues
    print("=" * 80)
    print("Test 2: Eigenvalues (alg_008)")
    print("=" * 80)

    # Setup: Create matrix
    setup_result = await algebra_server.call_tool_for_test(
        "create_matrix",
        {"elements": [[1, 2], [3, 4]], "key": "eigen_test"}
    )
    print(f"Setup result: {setup_result[0].text if hasattr(setup_result[0], 'text') else setup_result}")

    # Test: Calculate eigenvalues
    test_result = await algebra_server.call_tool_for_test(
        "matrix_eigenvalues",
        {"matrix_key": "eigen_test"}
    )
    content = test_result[0].text if hasattr(test_result[0], 'text') else str(test_result)
    print(f"Test result: {content}")

    try:
        parsed = json.loads(content)
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
    except:
        print("Could not parse as JSON")

    print()

    # Test 3: Eigenvectors
    print("=" * 80)
    print("Test 3: Eigenvectors (alg_009)")
    print("=" * 80)

    # Setup: Create matrix
    setup_result = await algebra_server.call_tool_for_test(
        "create_matrix",
        {"elements": [[1, 2], [3, 4]], "key": "eigvec_test"}
    )
    print(f"Setup result: {setup_result[0].text if hasattr(setup_result[0], 'text') else setup_result}")

    # Test: Calculate eigenvectors
    test_result = await algebra_server.call_tool_for_test(
        "matrix_eigenvectors",
        {"matrix_key": "eigvec_test"}
    )
    content = test_result[0].text if hasattr(test_result[0], 'text') else str(test_result)
    print(f"Test result: {content}")

    try:
        parsed = json.loads(content)
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
    except:
        print("Could not parse as JSON")

asyncio.run(test_matrix_operations())
