"""
Debug all failing tests to see actual responses.

This script runs each failing test and shows what it actually returns.
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

# Failing tests from benchmark (test_id: category)
FAILING_TESTS = {
    "alg_010": "algebra",
    "alg_011": "algebra",
    "analysis_003": "analysis",
    "analysis_008": "analysis",
    "analysis_012": "analysis",
    "analysis_013": "analysis",
    "geom_002": "geometry",
    "geom_005": "geometry",
    "geom_009": "geometry",
    "geom_012": "geometry",
    "stat_002": "statistics",
    "stat_003": "statistics",
    "stat_005": "statistics",
    "stat_007": "statistics",
    "stat_013": "statistics",
    "stat_014": "statistics",
    "phys_001": "physics",
    "phys_002": "physics",
    "phys_003": "physics",
    "phys_005": "physics",
    "phys_006": "physics",
    "phys_007": "physics",
    "phys_008": "physics",
    "phys_011": "physics",
    "phys_012": "physics",
    "phys_013": "physics",
    "phys_015": "physics",
    "phys_016": "physics",
    "logic_001": "logic",
    "logic_002": "logic",
    "logic_005": "logic",
    "logic_007": "logic",
    "logic_008": "logic",
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
        # Execute setup steps if any
        if 'setup_steps' in test_case and test_case['setup_steps']:
            print("Setup steps:")
            for step in test_case['setup_steps']:
                result = await server.call_tool_for_test(step['tool'], step['params'])
                content = result[0].text if hasattr(result[0], 'text') else str(result[0])
                print(f"  {step['tool']}: {content[:100]}...")

        # Execute test
        tool_name = test_case['reasonforge_tool']
        params = test_case['reasonforge_params']

        result = await server.call_tool_for_test(tool_name, params)
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])

        print(f"\nTool: {tool_name}")
        print(f"Params: {params}")
        print(f"Response: {content[:200]}")

        # Try to parse as JSON
        try:
            parsed = json.loads(content)
            print(f"\nParsed JSON keys: {list(parsed.keys())}")
            print(f"Full response:\n{json.dumps(parsed, indent=2)[:500]}")
        except:
            print("\nNot JSON format")

        print(f"\nExpected to contain: '{test_case['expected_answer']}'")
        print(f"Validation type: {test_case['validation_type']}")
        print(f"Contains expected? {test_case['expected_answer'] in content}")

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
