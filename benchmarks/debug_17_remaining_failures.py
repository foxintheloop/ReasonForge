"""
Debug the 17 remaining failures to see actual responses.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-algebra" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-analysis" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-geometry" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-statistics" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-physics" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-logic" / "src"))

from reasonforge_algebra.server import server as algebra_server
from reasonforge_analysis.server import server as analysis_server
from reasonforge_geometry.server import server as geometry_server
from reasonforge_statistics.server import server as statistics_server
from reasonforge_physics.server import server as physics_server
from reasonforge_logic.server import server as logic_server

# Map categories to servers
SERVERS = {
    "algebra": algebra_server,
    "analysis": analysis_server,
    "geometry": geometry_server,
    "statistics": statistics_server,
    "physics": physics_server,
    "logic": logic_server
}

# Failing tests from benchmark
FAILING_TESTS = {
    "analysis_008": "analysis",
    "stat_003": "statistics",
    "stat_005": "statistics",
    "stat_007": "statistics",
    "stat_013": "statistics",
    "stat_014": "statistics",
    "phys_001": "physics",
    "phys_002": "physics",
    "phys_003": "physics",
    "phys_008": "physics",
    "phys_015": "physics",
    "phys_016": "physics",
    "logic_001": "logic",
    "logic_002": "logic",
    "logic_009": "logic",
    "logic_010": "logic",
    "logic_011": "logic"
}

sys.path.insert(0, 'benchmarks')
from test_cases import TEST_CASES

async def debug_test(test_case, server):
    """Debug a single test case"""
    test_id = test_case['id']
    print(f"\n{'='*80}")
    print(f"Test: {test_id} - {test_case['problem'][:60]}")
    print(f"{'='*80}")

    try:
        # Execute test
        tool_name = test_case['reasonforge_tool']
        params = test_case['reasonforge_params']

        result = await server.call_tool_for_test(tool_name, params)
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])

        print(f"\nTool: {tool_name}")
        print(f"Params: {params}")
        print(f"Response length: {len(content)}")
        print(f"Response preview: {content[:400]}")

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"\nParsed JSON keys: {list(parsed.keys())}")
            print(f"Full response:\n{json.dumps(parsed, indent=2)[:600]}")
        except:
            print("\nNot JSON format - raw content:")
            print(content[:600])

        print(f"\nExpected to contain: '{test_case['expected_answer']}'")
        print(f"Validation type: {test_case['validation_type']}")
        print(f"Contains expected? {test_case['expected_answer'] in content}")

        # Check if response_field exists
        if 'response_field' in test_case:
            print(f"WARNING: Still has response_field: {test_case['response_field']}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    """Debug all failing tests"""

    # Group tests by category
    for test in TEST_CASES:
        test_id = test.get('id')
        if test_id in FAILING_TESTS:
            category = FAILING_TESTS[test_id]
            server = SERVERS.get(category)

            if server:
                await debug_test(test, server)
                await asyncio.sleep(0.1)  # Brief pause between tests

asyncio.run(main())
