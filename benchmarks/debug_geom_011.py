"""Debug geom_011 unit conversion test"""
import asyncio
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-geometry" / "src"))

from reasonforge_geometry.server import server as geometry_server

async def test_unit_conversion():
    """Test the actual unit conversion"""
    tool_name = "quantity_convert_units"
    params = {"expression": "5", "from_unit": "meter", "to_unit": "centimeter"}

    print("Testing unit conversion:")
    print(f"Tool: {tool_name}")
    print(f"Params: {params}")
    print()

    try:
        result = await geometry_server.call_tool_for_test(tool_name, params)

        # Extract content
        if isinstance(result, list) and len(result) > 0:
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        else:
            content = str(result)

        print(f"Raw result: {content}")
        print()

        # Try to parse JSON
        try:
            parsed = json.loads(content)
            print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
            print()
            print(f"'converted' field: {parsed.get('converted', 'NOT FOUND')}")
        except json.JSONDecodeError:
            print(f"Could not parse as JSON")

        # Check if 500 is in the content
        print()
        print(f"Contains '500': {'500' in content}")
        print(f"Contains '500 centimeter': {'500 centimeter' in content}")
        print(f"Contains 'centimeter': {'centimeter' in content}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_unit_conversion())
