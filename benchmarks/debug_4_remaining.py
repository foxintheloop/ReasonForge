"""Debug the 4 remaining failures after SKIP fixes."""

import asyncio
import json
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-geometry" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-physics" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-logic" / "src"))

from reasonforge_geometry.server import server as geometry_server
from reasonforge_physics.server import server as physics_server
from reasonforge_logic.server import server as logic_server

async def debug_geom_002():
    print("\n" + "="*80)
    print("geom_002: Create vector field")
    print("="*80)

    # Setup: create coordinate system
    result1 = await geometry_server.call_tool_for_test(
        "create_coordinate_system",
        {"name": "C1", "type": "cartesian"}
    )
    print(f"Setup result: {result1[0].text[:200]}")

    # Main test
    result2 = await geometry_server.call_tool_for_test(
        "create_vector_field",
        {"coord_system": "C1", "components": {"i": "y", "j": "-x", "k": "0"}}
    )
    content = result2[0].text
    print(f"\nResponse: {content}")
    print(f"Contains 'field_key': {'field_key' in content}")

async def debug_phys_008():
    print("\n" + "="*80)
    print("phys_008: Tensor product")
    print("="*80)

    # Setup: create states
    result1 = await physics_server.call_tool_for_test(
        "create_quantum_state",
        {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
    )
    print(f"State 1: {result1[0].text}")

    result2 = await physics_server.call_tool_for_test(
        "create_quantum_state",
        {"state_type": "pure", "num_qubits": 1, "amplitudes": [0, 1]}
    )
    print(f"State 2: {result2[0].text}")

    # Main test
    result3 = await physics_server.call_tool_for_test(
        "tensor_product_states",
        {"state_keys": ["quantum_state_0", "quantum_state_1"]}
    )
    content = result3[0].text
    print(f"\nResponse: {content}")
    print(f"Contains 'tensor': {'tensor' in content}")

async def debug_phys_015():
    print("\n" + "="*80)
    print("phys_015: Quantum evolution")
    print("="*80)

    # Setup: create state
    result1 = await physics_server.call_tool_for_test(
        "create_quantum_state",
        {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
    )
    print(f"State: {result1[0].text}")

    # Main test
    result2 = await physics_server.call_tool_for_test(
        "quantum_evolution",
        {"state_key": "quantum_state_0", "hamiltonian": "[[1, 0], [0, -1]]", "time": "t"}
    )
    content = result2[0].text
    print(f"\nResponse: {content}")
    print(f"Contains 'exp': {'exp' in content}")

async def debug_logic_011():
    print("\n" + "="*80)
    print("logic_011: Modal logic")
    print("="*80)

    result = await logic_server.call_tool_for_test(
        "modal_logic",
        {"logic_type": "alethic", "formula": "P"}
    )
    content = result[0].text
    print(f"\nResponse: {content}")
    print(f"Contains 'Box': {'Box' in content}")

async def main():
    await debug_geom_002()
    await debug_phys_008()
    await debug_phys_015()
    await debug_logic_011()

asyncio.run(main())
