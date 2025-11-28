"""Test a single failing test to see what's happening."""

import asyncio
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-analysis" / "src"))

from reasonforge_analysis.server import server as analysis_server

async def test():
    # Test analysis_008 directly
    result = await analysis_server.call_tool_for_test(
        "fourier_transform",
        {'expression': 'exp(-x**2)', 'variable': 'x', 'transform_variable': 'k'}
    )

    content = result[0].text if hasattr(result[0], 'text') else str(result[0])

    print("Response:")
    print(content)
    print("\nChecking for 'sqrt(pi)':")
    print(f"Contains 'sqrt(pi)': {'sqrt(pi)' in content}")
    print(f"Contains 'pi': {'pi' in content}")

    # Try parsing
    parsed = json.loads(content)
    print(f"\nParsed keys: {list(parsed.keys())}")
    print(f"Transform field: {parsed.get('transform')}")

asyncio.run(test())
