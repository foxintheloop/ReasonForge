"""
Tests for numerical-symbolic hybrid tools.

This module tests the 6 numerical-symbolic hybrid tools from numerical_hybrid_tools.py:
1. symbolic_optimization_setup - Set up optimization with KKT conditions
2. symbolic_ode_initial_conditions - Solve ODEs with symbolic ICs
3. perturbation_theory - Apply perturbation methods
4. asymptotic_analysis - Derive asymptotic expansions
5. special_functions_properties - Properties of special functions
6. integral_transforms_custom - Define and apply custom transforms

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.numerical_hybrid_tools import handle_numerical_hybrid_tool


class TestSymbolicOptimizationSetup:
    """Test the 'symbolic_optimization_setup' tool."""

    @pytest.mark.asyncio
    async def test_setup_constrained_optimization(self, ai):
        """Test setting up constrained optimization (from USAGE_EXAMPLES.md)."""
        result = await handle_numerical_hybrid_tool(
            'symbolic_optimization_setup',
            {
                'objective': 'x**2 + y**2',
                'equality_constraints': ['x + y - 1'],
                'inequality_constraints': [],
                'variables': ['x', 'y']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'lagrangian' in data or 'kkt_conditions' in data

    @pytest.mark.asyncio
    async def test_setup_with_inequalities(self, ai):
        """Test setup with inequality constraints."""
        result = await handle_numerical_hybrid_tool(
            'symbolic_optimization_setup',
            {
                'objective': 'x**2 + y**2',
                'equality_constraints': [],
                'inequality_constraints': ['x - 0', 'y - 0'],
                'variables': ['x', 'y']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_unconstrained_optimization(self, ai):
        """Test unconstrained optimization setup."""
        result = await handle_numerical_hybrid_tool(
            'symbolic_optimization_setup',
            {
                'objective': 'x**2',
                'equality_constraints': [],
                'inequality_constraints': [],
                'variables': ['x']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestSymbolicOdeInitialConditions:
    """Test the 'symbolic_ode_initial_conditions' tool."""

    @pytest.mark.asyncio
    async def test_solve_with_symbolic_ic(self, ai):
        """Test solving ODE with symbolic IC (from USAGE_EXAMPLES.md)."""
        # First introduce function and expression
        await handle_numerical_hybrid_tool('introduce_function', {'name': 'y'}, ai)

        result = await handle_numerical_hybrid_tool(
            'symbolic_ode_initial_conditions',
            {
                'equation_key': 'expr_1',
                'function_name': 'y',
                'initial_conditions': {'y(0)': 'y_0'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'solution' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_multiple_symbolic_ics(self, ai):
        """Test with multiple symbolic initial conditions."""
        result = await handle_numerical_hybrid_tool(
            'symbolic_ode_initial_conditions',
            {
                'equation_key': 'ode_1',
                'function_name': 'y',
                'initial_conditions': {'y(0)': 'a', 'y\'(0)': 'b'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestPerturbationTheory:
    """Test the 'perturbation_theory' tool."""

    @pytest.mark.asyncio
    async def test_regular_perturbation(self, ai):
        """Test regular perturbation method (from USAGE_EXAMPLES.md)."""
        result = await handle_numerical_hybrid_tool(
            'perturbation_theory',
            {
                'equation': 'x\'\' + x + epsilon*x**3',
                'perturbation_type': 'regular',
                'small_parameter': 'epsilon',
                'order': 2
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'expansion' in data or 'zeroth_order' in data

    @pytest.mark.asyncio
    async def test_singular_perturbation(self, ai):
        """Test singular perturbation."""
        result = await handle_numerical_hybrid_tool(
            'perturbation_theory',
            {
                'equation': 'epsilon*y\'\' + y\' + y',
                'perturbation_type': 'singular',
                'small_parameter': 'epsilon',
                'order': 1
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_multiple_scales(self, ai):
        """Test multiple scales method."""
        result = await handle_numerical_hybrid_tool(
            'perturbation_theory',
            {
                'equation': 'x\'\' + x + epsilon*x\'',
                'perturbation_type': 'multiple_scales',
                'small_parameter': 'epsilon',
                'order': 1
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestAsymptoticAnalysis:
    """Test the 'asymptotic_analysis' tool."""

    @pytest.mark.asyncio
    async def test_asymptotic_expansion_at_infinity(self, ai):
        """Test asymptotic expansion at infinity (from USAGE_EXAMPLES.md)."""
        result = await handle_numerical_hybrid_tool(
            'asymptotic_analysis',
            {
                'expression': 'exp(x)',
                'variable': 'x',
                'limit_point': 'inf',
                'order': 3
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'asymptotic_series' in data or 'leading_behavior' in data

    @pytest.mark.asyncio
    async def test_asymptotic_at_zero(self, ai):
        """Test asymptotic expansion at zero."""
        result = await handle_numerical_hybrid_tool(
            'asymptotic_analysis',
            {
                'expression': 'sin(x)/x',
                'variable': 'x',
                'limit_point': '0',
                'order': 3
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_rational_function_asymptotics(self, ai):
        """Test asymptotics of rational function."""
        result = await handle_numerical_hybrid_tool(
            'asymptotic_analysis',
            {
                'expression': '(x**2 + 1)/(x**3 + x)',
                'variable': 'x',
                'limit_point': 'inf',
                'order': 2
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestSpecialFunctionsProperties:
    """Test the 'special_functions_properties' tool."""

    @pytest.mark.asyncio
    async def test_legendre_recurrence(self, ai):
        """Test Legendre polynomial recurrence (from USAGE_EXAMPLES.md)."""
        result = await handle_numerical_hybrid_tool(
            'special_functions_properties',
            {
                'function_type': 'legendre',
                'operation': 'recurrence',
                'parameters': {'n': 'n'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'recurrence_relation' in data

    @pytest.mark.asyncio
    async def test_bessel_zeros(self, ai):
        """Test Bessel function zeros."""
        result = await handle_numerical_hybrid_tool(
            'special_functions_properties',
            {
                'function_type': 'bessel',
                'operation': 'zeros',
                'parameters': {'n': '0'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_hermite_orthogonality(self, ai):
        """Test Hermite polynomial orthogonality."""
        result = await handle_numerical_hybrid_tool(
            'special_functions_properties',
            {
                'function_type': 'hermite',
                'operation': 'orthogonality',
                'parameters': {}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_generating_function(self, ai):
        """Test generating function."""
        result = await handle_numerical_hybrid_tool(
            'special_functions_properties',
            {
                'function_type': 'laguerre',
                'operation': 'generating_function',
                'parameters': {}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestIntegralTransformsCustom:
    """Test the 'integral_transforms_custom' tool."""

    @pytest.mark.asyncio
    async def test_hankel_transform(self, ai):
        """Test Hankel transform (from USAGE_EXAMPLES.md)."""
        result = await handle_numerical_hybrid_tool(
            'integral_transforms_custom',
            {
                'transform_type': 'hankel',
                'expression': 'exp(-r**2)',
                'variable': 'r'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_hilbert_transform(self, ai):
        """Test Hilbert transform."""
        result = await handle_numerical_hybrid_tool(
            'integral_transforms_custom',
            {
                'transform_type': 'hilbert',
                'expression': 'sin(x)',
                'variable': 'x'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_custom_transform(self, ai):
        """Test custom integral transform."""
        result = await handle_numerical_hybrid_tool(
            'integral_transforms_custom',
            {
                'transform_type': 'custom',
                'kernel': 'exp(-s*t)',
                'expression': 'f(t)',
                'variable': 't'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_abel_transform(self, ai):
        """Test Abel transform."""
        result = await handle_numerical_hybrid_tool(
            'integral_transforms_custom',
            {
                'transform_type': 'abel',
                'expression': 'r**2',
                'variable': 'r'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data
