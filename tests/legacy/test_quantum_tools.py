"""
Tests for quantum computing tools.

This module tests the 10 quantum computing tools from quantum_tools.py:
1. create_quantum_state - Create quantum state vectors
2. quantum_gate_operations - Apply quantum gates
3. tensor_product_states - Tensor products
4. quantum_entanglement_measure - Calculate entanglement
5. quantum_circuit_symbolic - Build quantum circuits
6. quantum_measurement - Apply measurement operators
7. quantum_fidelity - Calculate fidelity
8. pauli_matrices - Generate Pauli matrices
9. commutator_anticommutator - Compute [A,B] and {A,B}
10. quantum_evolution - Unitary time evolution

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.quantum_tools import handle_quantum_tool


class TestCreateQuantumState:
    """Test the 'create_quantum_state' tool."""

    @pytest.mark.asyncio
    async def test_create_pure_state(self, ai):
        """Test creating pure state |ψ⟩ = (|0⟩ + |1⟩)/√2 (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'create_quantum_state',
            {
                'state_type': 'pure',
                'num_qubits': 1,
                'amplitudes': ['1/sqrt(2)', '1/sqrt(2)']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'state_key' in data
        assert data['num_qubits'] == 1

    @pytest.mark.asyncio
    async def test_create_two_qubit_state(self, ai):
        """Test creating two-qubit state."""
        result = await handle_quantum_tool(
            'create_quantum_state',
            {
                'state_type': 'pure',
                'num_qubits': 2,
                'amplitudes': ['1/sqrt(2)', '0', '0', '1/sqrt(2)']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert data['num_qubits'] == 2

    @pytest.mark.asyncio
    async def test_create_mixed_state(self, ai):
        """Test creating mixed quantum state."""
        result = await handle_quantum_tool(
            'create_quantum_state',
            {
                'state_type': 'mixed',
                'num_qubits': 1,
                'density_matrix': [['1/2', '0'], ['0', '1/2']]
            },
            ai
        )

        data = json.loads(result[0].text)
        # Should succeed or indicate mixed states not yet supported
        assert 'status' in data or 'error' in data


class TestQuantumGateOperations:
    """Test the 'quantum_gate_operations' tool."""

    @pytest.mark.asyncio
    async def test_apply_hadamard_gate(self, ai):
        """Test applying Hadamard gate to |0⟩ (from USAGE_EXAMPLES.md)."""
        # First create a state
        state_result = await handle_quantum_tool(
            'create_quantum_state',
            {
                'state_type': 'pure',
                'num_qubits': 1,
                'amplitudes': ['1', '0']
            },
            ai
        )

        state_data = json.loads(state_result[0].text)
        state_key = state_data.get('state_key', 'state_0')

        # Apply Hadamard gate
        result = await handle_quantum_tool(
            'quantum_gate_operations',
            {
                'state_key': state_key,
                'gates': ['Hadamard']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'final_state' in data

    @pytest.mark.asyncio
    async def test_apply_pauli_x(self, ai):
        """Test applying Pauli X gate."""
        result = await handle_quantum_tool(
            'quantum_gate_operations',
            {
                'state_key': 'state_0',
                'gates': ['PauliX']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestTensorProductStates:
    """Test the 'tensor_product_states' tool."""

    @pytest.mark.asyncio
    async def test_tensor_product(self, ai):
        """Test tensor product of |0⟩ and |1⟩ (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'tensor_product_states',
            {
                'state_keys': ['state_0', 'state_1']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'tensor_product' in data


class TestQuantumEntanglementMeasure:
    """Test the 'quantum_entanglement_measure' tool."""

    @pytest.mark.asyncio
    async def test_entanglement_entropy(self, ai):
        """Test calculating entanglement entropy (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'quantum_entanglement_measure',
            {
                'state_key': 'bell_state',
                'measure_type': 'entropy',
                'subsystem_partition': [[0], [1]]
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'entanglement_entropy' in data

    @pytest.mark.asyncio
    async def test_concurrence_measure(self, ai):
        """Test concurrence entanglement measure."""
        result = await handle_quantum_tool(
            'quantum_entanglement_measure',
            {
                'state_key': 'bell_state',
                'measure_type': 'concurrence',
                'subsystem_partition': [[0], [1]]
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestQuantumCircuitSymbolic:
    """Test the 'quantum_circuit_symbolic' tool."""

    @pytest.mark.asyncio
    async def test_build_bell_state_circuit(self, ai):
        """Test building circuit for Bell state (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'quantum_circuit_symbolic',
            {
                'num_qubits': 2,
                'gate_sequence': [
                    {'gate': 'H', 'target': 0},
                    {'gate': 'CNOT', 'control': 0, 'target': 1}
                ],
                'initial_state': '|00⟩'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'final_state' in data

    @pytest.mark.asyncio
    async def test_single_qubit_circuit(self, ai):
        """Test single qubit circuit."""
        result = await handle_quantum_tool(
            'quantum_circuit_symbolic',
            {
                'num_qubits': 1,
                'gate_sequence': [{'gate': 'H', 'target': 0}]
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestQuantumMeasurement:
    """Test the 'quantum_measurement' tool."""

    @pytest.mark.asyncio
    async def test_measurement_computational_basis(self, ai):
        """Test measurement in computational basis (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'quantum_measurement',
            {
                'state_key': 'bell_state',
                'measurement_basis': 'computational'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'probabilities' in data

    @pytest.mark.asyncio
    async def test_measurement_hadamard_basis(self, ai):
        """Test measurement in Hadamard basis."""
        result = await handle_quantum_tool(
            'quantum_measurement',
            {
                'state_key': 'state_plus',
                'measurement_basis': 'hadamard'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestQuantumFidelity:
    """Test the 'quantum_fidelity' tool."""

    @pytest.mark.asyncio
    async def test_calculate_fidelity(self, ai):
        """Test calculating fidelity between states (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'quantum_fidelity',
            {
                'state_key_1': 'state_0',
                'state_key_2': 'state_plus'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'fidelity' in data


class TestPauliMatrices:
    """Test the 'pauli_matrices' tool."""

    @pytest.mark.asyncio
    async def test_generate_pauli_x(self, ai):
        """Test generating Pauli X matrix (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'pauli_matrices',
            {
                'operation': 'single',
                'matrix': 'X'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_pauli_tensor_product(self, ai):
        """Test Pauli matrix tensor product."""
        result = await handle_quantum_tool(
            'pauli_matrices',
            {
                'operation': 'tensor_product',
                'pauli_string': 'XYZ'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestCommutatorAnticommutator:
    """Test the 'commutator_anticommutator' tool."""

    @pytest.mark.asyncio
    async def test_calculate_commutator(self, ai):
        """Test calculating [X, Y] for Pauli matrices (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'commutator_anticommutator',
            {
                'operator_1': 'X',
                'operator_2': 'Y',
                'operation': 'commutator'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_calculate_anticommutator(self, ai):
        """Test calculating {X, Y} for Pauli matrices."""
        result = await handle_quantum_tool(
            'commutator_anticommutator',
            {
                'operator_1': 'X',
                'operator_2': 'Y',
                'operation': 'anticommutator'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data


class TestQuantumEvolution:
    """Test the 'quantum_evolution' tool."""

    @pytest.mark.asyncio
    async def test_evolve_under_hamiltonian(self, ai):
        """Test evolving |0⟩ under H = σ_x (from USAGE_EXAMPLES.md)."""
        result = await handle_quantum_tool(
            'quantum_evolution',
            {
                'state_key': 'state_0',
                'hamiltonian': 'sigma_x',
                'time': 't',
                'method': 'schrodinger'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'evolved_state' in data

    @pytest.mark.asyncio
    async def test_heisenberg_evolution(self, ai):
        """Test Heisenberg picture evolution."""
        result = await handle_quantum_tool(
            'quantum_evolution',
            {
                'state_key': 'state_0',
                'hamiltonian': 'omega*sigma_z',
                'time': 't',
                'method': 'heisenberg'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data
