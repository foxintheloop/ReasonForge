"""
Smoke test to verify all tools are accessible and count is correct.
"""

from reasonforge_mcp.advanced_tools import get_advanced_tool_definitions
from reasonforge_mcp.hybrid_tools import get_hybrid_tool_definitions
from reasonforge_mcp.logic_tools import get_logic_tool_definitions
from reasonforge_mcp.quantum_tools import get_quantum_tool_definitions
from reasonforge_mcp.data_science_tools import get_data_science_tool_definitions
from reasonforge_mcp.visualization_tools import get_visualization_tool_definitions
from reasonforge_mcp.physics_tools import get_physics_tool_definitions
from reasonforge_mcp.numerical_hybrid_tools import get_numerical_hybrid_tool_definitions

def main():
    print("=" * 60)
    print("REASONFORGE MCP SERVER - SMOKE TEST")
    print("=" * 60)

    # Get all tool definitions
    advanced_tools = get_advanced_tool_definitions()
    hybrid_tools = get_hybrid_tool_definitions()
    logic_tools = get_logic_tool_definitions()
    quantum_tools = get_quantum_tool_definitions()
    data_science_tools = get_data_science_tool_definitions()
    visualization_tools = get_visualization_tool_definitions()
    physics_tools = get_physics_tool_definitions()
    numerical_hybrid_tools = get_numerical_hybrid_tool_definitions()

    # Count original tools (from server.py) - reduced from 15 to 12 after removing duplicates
    original_tool_count = 12

    # Calculate total
    total_tools = original_tool_count + len(advanced_tools) + len(hybrid_tools) + len(logic_tools) + len(quantum_tools) + len(data_science_tools) + len(visualization_tools) + len(physics_tools) + len(numerical_hybrid_tools)

    print(f"\n[OK] All imports successful!")
    print(f"\nTool counts by module:")
    print(f"  - Original tools (server.py):  {original_tool_count}")
    print(f"  - Advanced tools:              {len(advanced_tools)}")
    print(f"  - Hybrid tools:                {len(hybrid_tools)}")
    print(f"  - Logic tools:                 {len(logic_tools)}")
    print(f"  - Quantum tools:               {len(quantum_tools)}")
    print(f"  - Data Science tools:          {len(data_science_tools)}")
    print(f"  - Visualization tools:         {len(visualization_tools)}")
    print(f"  - Physics tools:               {len(physics_tools)}")
    print(f"  - Numerical Hybrid tools:      {len(numerical_hybrid_tools)}")
    print(f"  " + "-" * 40)
    print(f"  TOTAL:                         {total_tools}")

    # Verify no duplicates between modules
    all_tool_names = []
    all_tool_names.extend([t.name for t in advanced_tools])
    all_tool_names.extend([t.name for t in hybrid_tools])
    all_tool_names.extend([t.name for t in logic_tools])
    all_tool_names.extend([t.name for t in quantum_tools])
    all_tool_names.extend([t.name for t in data_science_tools])
    all_tool_names.extend([t.name for t in visualization_tools])
    all_tool_names.extend([t.name for t in physics_tools])
    all_tool_names.extend([t.name for t in numerical_hybrid_tools])

    if len(all_tool_names) != len(set(all_tool_names)):
        print(f"\n[ERROR] Duplicate tool names found!")
        duplicates = [name for name in all_tool_names if all_tool_names.count(name) > 1]
        print(f"  Duplicates: {set(duplicates)}")
    else:
        print(f"\n[OK] No duplicate tool names in non-original modules")

    # Verify removed duplicate tools are not in advanced/hybrid/logic modules
    removed_tools = ['basic_simplify', 'logical_reasoning', 'perform_matrix_operation']
    found_removed = [tool for tool in removed_tools if tool in all_tool_names]
    if found_removed:
        print(f"[ERROR] Removed duplicate tools found in modules: {found_removed}")
    else:
        print(f"[OK] All duplicate tools successfully removed")

    # Check for simplify_expression in advanced tools
    if 'simplify_expression' in all_tool_names:
        print(f"[OK] 'simplify_expression' found in advanced tools")
    else:
        print(f"[WARNING] 'simplify_expression' not found in advanced tools")

    print(f"\n" + "=" * 60)
    if total_tools == 113:
        print(f"SUCCESS! All {total_tools} tools are accessible.")
    else:
        print(f"WARNING: Expected 113 tools, but found {total_tools}")
    print("=" * 60)

    return total_tools

if __name__ == "__main__":
    main()
