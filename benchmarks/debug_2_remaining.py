"""
Debug the 2 remaining test failures.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add packages to path (same as benchmark_runner.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-geometry" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-physics" / "src"))

async def test_geom_002():
    """Test geom_002 - create vector field."""
    from reasonforge_geometry.server import server

    print("=" * 80)
    print("Testing geom_002: Create vector field")
    print("=" * 80)

    try:
        # Setup: Create coordinate system
        print("\n[Setup] Creating coordinate system...")
        setup_result = await server.call_tool_for_test(
            "create_coordinate_system",
            {"name": "C1", "type": "cartesian"}
        )
        print(f"Setup result: {setup_result[0].text if setup_result else 'None'}")

        # Main test: Create vector field
        print("\n[Main] Creating vector field...")
        result = await server.call_tool_for_test(
            "create_vector_field",
            {"coord_system": "C1", "components": ["x", "y", "0"]}
        )

        content = result[0].text if result else str(result)
        print(f"\nRaw response:\n{content}")

        parsed = json.loads(content)
        print(f"\nParsed JSON:\n{json.dumps(parsed, indent=2)}")

        # Check what we're looking for
        print(f"\nSearching for 'field_' in: {content}")
        print(f"Contains 'field_': {'field_' in content}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def test_phys_008():
    """Test phys_008 - tensor product."""
    from reasonforge_physics.server import server

    print("\n" + "=" * 80)
    print("Testing phys_008: Tensor product")
    print("=" * 80)

    try:
        # Setup: Create two quantum states
        print("\n[Setup] Creating quantum states...")
        state1 = await server.call_tool_for_test(
            "create_quantum_state",
            {"state_vector": [1, 0], "basis": "computational"}
        )
        print(f"State 1: {state1[0].text if state1 else 'None'}")

        state2 = await server.call_tool_for_test(
            "create_quantum_state",
            {"state_vector": [0, 1], "basis": "computational"}
        )
        print(f"State 2: {state2[0].text if state2 else 'None'}")

        # Main test: Tensor product
        print("\n[Main] Computing tensor product...")
        result = await server.call_tool_for_test(
            "tensor_product",
            {"state1": "quantum_state_0", "state2": "quantum_state_1"}
        )

        content = result[0].text if result else str(result)
        print(f"\nRaw response:\n{content}")

        parsed = json.loads(content)
        print(f"\nParsed JSON:\n{json.dumps(parsed, indent=2)}")

        # Check what we're looking for
        print(f"\nSearching for 'TensorProduct' in: {content}")
        print(f"Contains 'TensorProduct': {'TensorProduct' in content}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run both tests."""
    await test_geom_002()
    await test_phys_008()


if __name__ == "__main__":
    asyncio.run(main())
