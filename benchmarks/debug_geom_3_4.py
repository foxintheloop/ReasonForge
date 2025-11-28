"""
Debug geom_003 and geom_004.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-geometry" / "src"))

async def test_geom_003():
    """Test geom_003 - calculate curl."""
    from reasonforge_geometry.server import server

    print("=" * 80)
    print("Testing geom_003: Calculate curl")
    print("=" * 80)

    try:
        # Setup: Create coordinate system
        print("\n[Setup 1] Creating coordinate system...")
        result1 = await server.call_tool_for_test(
            "create_coordinate_system",
            {"name": "C1", "type": "Cartesian"}
        )
        print(f"Result: {result1[0].text if result1 else 'None'}")

        # Setup: Create vector field
        print("\n[Setup 2] Creating vector field...")
        result2 = await server.call_tool_for_test(
            "create_vector_field",
            {"coord_system": "C1", "components": {"i": "y", "j": "-x", "k": "0"}}
        )
        print(f"Result: {result2[0].text if result2 else 'None'}")

        # Main test: Calculate curl
        print("\n[Main] Calculating curl...")
        result = await server.call_tool_for_test(
            "calculate_curl",
            {"field_name": "field_0"}
        )

        content = result[0].text if result else str(result)
        print(f"\nRaw response:\n{content}")

        parsed = json.loads(content)
        print(f"\nParsed JSON:\n{json.dumps(parsed, indent=2)}")

        # Check curl field
        if "curl" in parsed:
            curl_value = parsed["curl"]
            print(f"\ncurl field: {curl_value}")
            print(f"curl field contains 'curl': {'curl' in str(curl_value)}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def test_geom_004():
    """Test geom_004 - calculate divergence."""
    from reasonforge_geometry.server import server

    print("\n" + "=" * 80)
    print("Testing geom_004: Calculate divergence")
    print("=" * 80)

    try:
        # Setup: Create coordinate system
        print("\n[Setup 1] Creating coordinate system...")
        result1 = await server.call_tool_for_test(
            "create_coordinate_system",
            {"name": "C2", "type": "Cartesian"}
        )
        print(f"Result: {result1[0].text if result1 else 'None'}")

        # Setup: Create vector field
        print("\n[Setup 2] Creating vector field...")
        result2 = await server.call_tool_for_test(
            "create_vector_field",
            {"coord_system": "C2", "components": {"i": "x", "j": "y", "k": "z"}}
        )
        print(f"Result: {result2[0].text if result2 else 'None'}")

        # Main test: Calculate divergence
        print("\n[Main] Calculating divergence...")
        result = await server.call_tool_for_test(
            "calculate_divergence",
            {"field_name": "field_1"}
        )

        content = result[0].text if result else str(result)
        print(f"\nRaw response:\n{content}")

        parsed = json.loads(content)
        print(f"\nParsed JSON:\n{json.dumps(parsed, indent=2)}")

        # Check divergence field
        if "divergence" in parsed:
            div_value = parsed["divergence"]
            print(f"\ndivergence field: {div_value}")
            print(f"divergence field contains 'divergence': {'divergence' in str(div_value)}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run both tests."""
    await test_geom_003()
    await test_geom_004()


if __name__ == "__main__":
    asyncio.run(main())
