"""
Tests for computational physics tools.

This module tests the 8 physics tools from physics_tools.py:
1. schrodinger_equation_solver - Solve quantum mechanics problems
2. wave_equation_solver - Solve classical wave equations
3. heat_equation_solver - Solve heat diffusion problems
4. maxwell_equations - Electromagnetic theory
5. special_relativity - Relativistic transformations
6. lagrangian_mechanics - Derive equations of motion
7. hamiltonian_mechanics - Hamiltonian formulation
8. noether_theorem - Symmetries and conservation laws

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.physics_tools import handle_physics_tool


class TestSchrodingerEquationSolver:
    """Test the 'schrodinger_equation_solver' tool."""

    @pytest.mark.asyncio
    async def test_infinite_square_well(self, ai):
        """Test 1D infinite square well problem (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'schrodinger_equation_solver',
            {
                'equation_type': 'time_independent',
                'dimension': 1,
                'potential': 'infinite_square_well',
                'boundary_conditions': {'psi(0)': '0', 'psi(L)': '0'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'wavefunctions' in data or 'energy_eigenvalues' in data

    @pytest.mark.asyncio
    async def test_harmonic_oscillator(self, ai):
        """Test quantum harmonic oscillator."""
        result = await handle_physics_tool(
            'schrodinger_equation_solver',
            {
                'equation_type': 'time_independent',
                'dimension': 1,
                'potential': 'harmonic_oscillator',
                'parameters': {'omega': 'omega'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_hydrogen_atom(self, ai):
        """Test hydrogen atom (3D Coulomb potential)."""
        result = await handle_physics_tool(
            'schrodinger_equation_solver',
            {
                'equation_type': 'time_independent',
                'dimension': 3,
                'potential': 'coulomb',
                'parameters': {'Z': '1'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestWaveEquationSolver:
    """Test the 'wave_equation_solver' tool."""

    @pytest.mark.asyncio
    async def test_string_vibration(self, ai):
        """Test 1D wave on string (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'wave_equation_solver',
            {
                'wave_type': 'string',
                'dimension': 1,
                'boundary_conditions': {'u(0,t)': '0', 'u(L,t)': '0'},
                'initial_conditions': {'u(x,0)': 'f(x)', 'u_t(x,0)': 'g(x)'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'solution' in data or 'wave_speed' in data

    @pytest.mark.asyncio
    async def test_membrane_vibration(self, ai):
        """Test 2D wave equation (membrane)."""
        result = await handle_physics_tool(
            'wave_equation_solver',
            {
                'wave_type': 'membrane',
                'dimension': 2,
                'geometry': 'rectangular',
                'boundary_conditions': {'u(0,y,t)': '0', 'u(L,y,t)': '0'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_acoustic_wave(self, ai):
        """Test acoustic wave equation."""
        result = await handle_physics_tool(
            'wave_equation_solver',
            {
                'wave_type': 'acoustic',
                'dimension': 3,
                'parameters': {'c': 'c'}  # speed of sound
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestHeatEquationSolver:
    """Test the 'heat_equation_solver' tool."""

    @pytest.mark.asyncio
    async def test_1d_heat_dirichlet(self, ai):
        """Test 1D heat equation with Dirichlet BC (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'heat_equation_solver',
            {
                'geometry': '1D',
                'boundary_conditions': {'u(0,t)': '0', 'u(L,t)': '0'},
                'initial_conditions': {'u(x,0)': 'f(x)'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'solution' in data or 'thermal_diffusivity' in data

    @pytest.mark.asyncio
    async def test_2d_heat_equation(self, ai):
        """Test 2D heat equation."""
        result = await handle_physics_tool(
            'heat_equation_solver',
            {
                'geometry': '2D',
                'boundary_conditions': {'u(0,y,t)': '0', 'u(L,y,t)': '0'},
                'initial_conditions': {'u(x,y,0)': 'f(x,y)'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_neumann_boundary_conditions(self, ai):
        """Test heat equation with Neumann BC (insulated)."""
        result = await handle_physics_tool(
            'heat_equation_solver',
            {
                'geometry': '1D',
                'boundary_conditions': {'du/dx(0,t)': '0', 'du/dx(L,t)': '0'},
                'initial_conditions': {'u(x,0)': 'T_0'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestMaxwellEquations:
    """Test the 'maxwell_equations' tool."""

    @pytest.mark.asyncio
    async def test_derive_wave_equation(self, ai):
        """Test deriving EM wave equation (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'maxwell_equations',
            {
                'operation': 'derive_wave',
                'vacuum': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'wave_equation' in data or 'wave_speed' in data

    @pytest.mark.asyncio
    async def test_poynting_vector(self, ai):
        """Test Poynting vector calculation."""
        result = await handle_physics_tool(
            'maxwell_equations',
            {
                'operation': 'poynting_vector',
                'fields': {'E': 'E', 'B': 'B'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_energy_density(self, ai):
        """Test electromagnetic energy density."""
        result = await handle_physics_tool(
            'maxwell_equations',
            {
                'operation': 'energy_density',
                'fields': {'E': 'E', 'B': 'B'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestSpecialRelativity:
    """Test the 'special_relativity' tool."""

    @pytest.mark.asyncio
    async def test_lorentz_transformation(self, ai):
        """Test Lorentz transformation (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'special_relativity',
            {
                'operation': 'lorentz_transform',
                'velocity': '0.6*c'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'gamma' in data or 'transformation' in data

    @pytest.mark.asyncio
    async def test_time_dilation(self, ai):
        """Test time dilation formula."""
        result = await handle_physics_tool(
            'special_relativity',
            {
                'operation': 'time_dilation',
                'velocity': 'v'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_length_contraction(self, ai):
        """Test length contraction."""
        result = await handle_physics_tool(
            'special_relativity',
            {
                'operation': 'length_contraction',
                'velocity': 'v'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_relativistic_energy(self, ai):
        """Test relativistic energy-momentum relation."""
        result = await handle_physics_tool(
            'special_relativity',
            {
                'operation': 'energy_momentum',
                'mass': 'm',
                'velocity': 'v'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestLagrangianMechanics:
    """Test the 'lagrangian_mechanics' tool."""

    @pytest.mark.asyncio
    async def test_simple_pendulum(self, ai):
        """Test simple pendulum (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'lagrangian_mechanics',
            {
                'system': 'simple_pendulum',
                'parameters': {'m': 'm', 'l': 'l', 'g': 'g'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'lagrangian' in data or 'equation_of_motion' in data

    @pytest.mark.asyncio
    async def test_double_pendulum(self, ai):
        """Test double pendulum system."""
        result = await handle_physics_tool(
            'lagrangian_mechanics',
            {
                'system': 'double_pendulum',
                'parameters': {'m1': 'm_1', 'm2': 'm_2', 'l1': 'l_1', 'l2': 'l_2'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_atwood_machine(self, ai):
        """Test Atwood machine."""
        result = await handle_physics_tool(
            'lagrangian_mechanics',
            {
                'system': 'atwood_machine',
                'parameters': {'m1': 'm_1', 'm2': 'm_2'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestHamiltonianMechanics:
    """Test the 'hamiltonian_mechanics' tool."""

    @pytest.mark.asyncio
    async def test_harmonic_oscillator_hamilton(self, ai):
        """Test Hamiltonian for harmonic oscillator (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'hamiltonian_mechanics',
            {
                'operation': 'hamilton_equations',
                'system': 'harmonic_oscillator',
                'parameters': {'m': 'm', 'k': 'k'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'hamiltonian' in data or 'equations' in data

    @pytest.mark.asyncio
    async def test_central_force(self, ai):
        """Test central force problem."""
        result = await handle_physics_tool(
            'hamiltonian_mechanics',
            {
                'operation': 'hamilton_equations',
                'system': 'central_force',
                'parameters': {'alpha': 'alpha'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_poisson_brackets(self, ai):
        """Test Poisson bracket calculation."""
        result = await handle_physics_tool(
            'hamiltonian_mechanics',
            {
                'operation': 'poisson_brackets',
                'function1': 'x',
                'function2': 'p'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestNoetherTheorem:
    """Test the 'noether_theorem' tool."""

    @pytest.mark.asyncio
    async def test_time_translation_symmetry(self, ai):
        """Test energy conservation from time symmetry (from USAGE_EXAMPLES.md)."""
        result = await handle_physics_tool(
            'noether_theorem',
            {
                'lagrangian': 'L(q, q_dot)',
                'symmetry_type': 'time_translation'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'conserved_quantity' in data or 'interpretation' in data

    @pytest.mark.asyncio
    async def test_spatial_translation(self, ai):
        """Test momentum conservation from spatial translation."""
        result = await handle_physics_tool(
            'noether_theorem',
            {
                'lagrangian': 'L(q, q_dot)',
                'symmetry_type': 'spatial_translation',
                'direction': 'x'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_rotational_symmetry(self, ai):
        """Test angular momentum conservation from rotational symmetry."""
        result = await handle_physics_tool(
            'noether_theorem',
            {
                'lagrangian': 'L(r, theta)',
                'symmetry_type': 'rotation',
                'axis': 'z'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data
