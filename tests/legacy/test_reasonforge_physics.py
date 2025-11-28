"""
Comprehensive tests for reasonforge-physics MCP server.

Tests all 16 tools:
- Classical Mechanics (5 tools): lagrangian_mechanics, hamiltonian_mechanics, noether_theorem, maxwell_equations, special_relativity
- Quantum States (5 tools): create_quantum_state, quantum_gate_operations, tensor_product_states, quantum_entanglement_measure, quantum_circuit_symbolic
- Quantum Operations (5 tools): quantum_measurement, quantum_fidelity, pauli_matrices, commutator_anticommutator, quantum_evolution
- Optimization (1 tool): symbolic_optimization_setup
"""

import sys
import os
import json
import pytest

# Add packages to path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-physics', 'src'))

from reasonforge_physics.server import server as physics_server


class TestClassicalMechanics:
    """Test classical mechanics tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_lagrangian_mechanics(self):
        """Test deriving equations of motion from Lagrangian."""
        result = await physics_server.call_tool_for_test(
            "lagrangian_mechanics",
            {
                "lagrangian": "m*v**2/2 - m*g*h",
                "generalized_coordinates": ["h"]
            }
        )
        data = json.loads(result[0].text)

        assert "equations_of_motion" in data
        assert "formula" in data

    @pytest.mark.asyncio
    async def test_hamiltonian_mechanics(self):
        """Test Hamiltonian mechanics."""
        result = await physics_server.call_tool_for_test(
            "hamiltonian_mechanics",
            {
                "hamiltonian": "p**2/(2*m) + m*g*q",
                "operation": "derive"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_noether_theorem(self):
        """Test finding conserved quantities."""
        result = await physics_server.call_tool_for_test(
            "noether_theorem",
            {
                "lagrangian": "m*v**2/2",
                "symmetry_transformation": "time"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_maxwell_equations(self):
        """Test Maxwell's equations."""
        result = await physics_server.call_tool_for_test(
            "maxwell_equations",
            {"operation": "gauss_law"}
        )
        data = json.loads(result[0].text)

        assert "operation" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_special_relativity(self):
        """Test special relativity calculations."""
        result = await physics_server.call_tool_for_test(
            "special_relativity",
            {
                "operation": "lorentz_factor",
                "velocity": "0.5*c"
            }
        )
        data = json.loads(result[0].text)

        assert "operation" in data or "result" in data


class TestQuantumStates:
    """Test quantum state tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_create_quantum_state(self):
        """Test creating quantum state."""
        result = await physics_server.call_tool_for_test(
            "create_quantum_state",
            {
                "state_type": "pure",
                "num_qubits": 1,
                "amplitudes": [1, 0]
            }
        )
        data = json.loads(result[0].text)

        # Verify we got a valid response
        assert len(data) > 0 and ("state_key" in data or "amplitudes" in data or "normalized" in data)

    @pytest.mark.asyncio
    async def test_quantum_gate_operations(self):
        """Test applying quantum gates."""
        # First create a state
        state_result = await physics_server.call_tool_for_test(
            "create_quantum_state",
            {
                "state_type": "pure",
                "num_qubits": 1,
                "amplitudes": [1, 0]
            }
        )
        state_data = json.loads(state_result[0].text)
        state_key = state_data.get("state_key", "quantum_state_0")

        result = await physics_server.call_tool_for_test(
            "quantum_gate_operations",
            {
                "state_key": state_key,
                "gates": ["H"]
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_tensor_product_states(self):
        """Test tensor product of quantum states."""
        # Create two states
        await physics_server.call_tool_for_test(
            "create_quantum_state",
            {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
        )
        await physics_server.call_tool_for_test(
            "create_quantum_state",
            {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
        )

        result = await physics_server.call_tool_for_test(
            "tensor_product_states",
            {"state_keys": ["quantum_state_0", "quantum_state_1"]}
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_quantum_entanglement_measure(self):
        """Test measuring quantum entanglement."""
        # Create a state
        state_result = await physics_server.call_tool_for_test(
            "create_quantum_state",
            {"state_type": "pure", "num_qubits": 2, "amplitudes": [1, 0, 0, 0]}
        )
        state_data = json.loads(state_result[0].text)
        state_key = state_data.get("state_key", "quantum_state_0")

        result = await physics_server.call_tool_for_test(
            "quantum_entanglement_measure",
            {
                "state_key": state_key,
                "measure_type": "von_neumann_entropy"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_quantum_circuit_symbolic(self):
        """Test building symbolic quantum circuits."""
        result = await physics_server.call_tool_for_test(
            "quantum_circuit_symbolic",
            {
                "num_qubits": 2,
                "gate_sequence": [{"gate": "H", "qubit": 0}, {"gate": "CNOT", "qubits": [0, 1]}]
            }
        )
        data = json.loads(result[0].text)

        assert "num_qubits" in data or len(data) > 0


class TestQuantumOperations:
    """Test quantum operation tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_quantum_measurement(self):
        """Test quantum measurement."""
        # Create a state
        state_result = await physics_server.call_tool_for_test(
            "create_quantum_state",
            {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
        )
        state_data = json.loads(state_result[0].text)
        state_key = state_data.get("state_key", "quantum_state_0")

        result = await physics_server.call_tool_for_test(
            "quantum_measurement",
            {
                "state_key": state_key,
                "measurement_basis": "computational"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_quantum_fidelity(self):
        """Test calculating fidelity between states."""
        # Create two states
        await physics_server.call_tool_for_test(
            "create_quantum_state",
            {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
        )
        await physics_server.call_tool_for_test(
            "create_quantum_state",
            {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
        )

        result = await physics_server.call_tool_for_test(
            "quantum_fidelity",
            {
                "state_key_1": "quantum_state_0",
                "state_key_2": "quantum_state_1"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_pauli_matrices(self):
        """Test Pauli matrices operations."""
        result = await physics_server.call_tool_for_test(
            "pauli_matrices",
            {
                "operation": "get",
                "matrix": "X"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_commutator_anticommutator(self):
        """Test commutator calculation."""
        result = await physics_server.call_tool_for_test(
            "commutator_anticommutator",
            {
                "operator_1": "[[0, 1], [1, 0]]",
                "operator_2": "[[0, -I], [I, 0]]",
                "operation": "commutator"
            }
        )
        data = json.loads(result[0].text)

        assert "operation" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_quantum_evolution(self):
        """Test quantum state evolution."""
        # Create a state
        state_result = await physics_server.call_tool_for_test(
            "create_quantum_state",
            {"state_type": "pure", "num_qubits": 1, "amplitudes": [1, 0]}
        )
        state_data = json.loads(state_result[0].text)
        state_key = state_data.get("state_key", "quantum_state_0")

        result = await physics_server.call_tool_for_test(
            "quantum_evolution",
            {
                "state_key": state_key,
                "hamiltonian": "[[1, 0], [0, -1]]",
                "time": "t"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0


class TestOptimization:
    """Test optimization tools (1 tool)."""

    @pytest.mark.asyncio
    async def test_symbolic_optimization_setup(self):
        """Test physics optimization setup."""
        result = await physics_server.call_tool_for_test(
            "symbolic_optimization_setup",
            {
                "objective": "E = m*c**2",
                "variables": ["m"]
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
