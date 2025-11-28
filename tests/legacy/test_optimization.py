"""
Tests for optimization extension tools.

This module tests the 5 optimization tools from advanced_tools.py:
1. lagrange_multipliers - Constrained optimization
2. linear_programming - LP problems with simplex method
3. convex_optimization - Convexity verification and solving
4. calculus_of_variations - Euler-Lagrange equations
5. dynamic_programming - Bellman equations

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestLagrangeMultipliers:
    """Test the 'lagrange_multipliers' tool."""

    @pytest.mark.asyncio
    async def test_maximize_xy_on_unit_circle(self, ai):
        """Test maximize xy subject to x^2 + y^2 = 1 (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'lagrange_multipliers',
            {
                'objective': 'x*y',
                'constraints': ['x**2 + y**2 - 1'],
                'variables': ['x', 'y']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'lagrangian' in data
        assert 'critical_points' in data
        # Should find critical points involving sqrt(2)/2

    @pytest.mark.asyncio
    async def test_minimize_distance_to_line(self, ai):
        """Test constrained optimization with linear constraint."""
        result = await handle_advanced_tool(
            'lagrange_multipliers',
            {
                'objective': 'x**2 + y**2',
                'constraints': ['x + y - 2'],
                'variables': ['x', 'y']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'critical_points' in data

    @pytest.mark.asyncio
    async def test_multiple_constraints(self, ai):
        """Test optimization with multiple constraints."""
        result = await handle_advanced_tool(
            'lagrange_multipliers',
            {
                'objective': 'x + y + z',
                'constraints': ['x**2 + y**2 - 1', 'z - 1'],
                'variables': ['x', 'y', 'z']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'lagrangian' in data


class TestLinearProgramming:
    """Test the 'linear_programming' tool."""

    @pytest.mark.asyncio
    async def test_minimize_with_inequality_constraints(self, ai):
        """Test minimize 3x + 4y subject to constraints (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'linear_programming',
            {
                'objective': '3*x + 4*y',
                'constraints': ['x + y - 5', '2*x + y - 8', 'x', 'y'],
                'variables': ['x', 'y'],
                'minimize': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'optimal_solution' in data or 'solution' in data

    @pytest.mark.asyncio
    async def test_maximize_profit(self, ai):
        """Test maximization problem."""
        result = await handle_advanced_tool(
            'linear_programming',
            {
                'objective': '5*x + 4*y',
                'constraints': ['x + 2*y - 10', '2*x + y - 10', 'x', 'y'],
                'variables': ['x', 'y'],
                'minimize': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'

    @pytest.mark.asyncio
    async def test_single_variable_lp(self, ai):
        """Test single variable linear programming."""
        result = await handle_advanced_tool(
            'linear_programming',
            {
                'objective': '2*x',
                'constraints': ['x - 5', 'x'],
                'variables': ['x'],
                'minimize': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'


class TestConvexOptimization:
    """Test the 'convex_optimization' tool."""

    @pytest.mark.asyncio
    async def test_verify_convex_quadratic(self, ai):
        """Test verify x^2 is convex and find minimum (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'convex_optimization',
            {
                'objective': 'x**2',
                'constraints': [],
                'variables': ['x'],
                'check_convexity': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'is_convex' in data
        assert data['is_convex'] is True

    @pytest.mark.asyncio
    async def test_convex_with_constraints(self, ai):
        """Test convex optimization with constraints."""
        result = await handle_advanced_tool(
            'convex_optimization',
            {
                'objective': 'x**2 + y**2',
                'constraints': ['x + y - 1'],
                'variables': ['x', 'y'],
                'check_convexity': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'

    @pytest.mark.asyncio
    async def test_nonconvex_function(self, ai):
        """Test detection of non-convex function."""
        result = await handle_advanced_tool(
            'convex_optimization',
            {
                'objective': 'x**3',
                'constraints': [],
                'variables': ['x'],
                'check_convexity': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        # Should detect as non-convex


class TestCalculusOfVariations:
    """Test the 'calculus_of_variations' tool."""

    @pytest.mark.asyncio
    async def test_shortest_path_geodesic(self, ai):
        """Test finding shortest path/geodesic (from USAGE_EXAMPLES.md)."""
        # Introduce function for differential calculus
        await handle_advanced_tool('introduce_function', {'name': 'y'}, ai)

        result = await handle_advanced_tool(
            'calculus_of_variations',
            {
                'functional': 'sqrt(1 + Derivative(y(x), x)**2)',
                'function_name': 'y',
                'independent_var': 'x'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'euler_lagrange' in data or 'equation' in data

    @pytest.mark.asyncio
    async def test_brachistochrone_problem(self, ai):
        """Test brachistochrone problem functional."""
        await handle_advanced_tool('introduce_function', {'name': 'y'}, ai)

        result = await handle_advanced_tool(
            'calculus_of_variations',
            {
                'functional': 'sqrt((1 + Derivative(y(x), x)**2)/y(x))',
                'function_name': 'y',
                'independent_var': 'x'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'

    @pytest.mark.asyncio
    async def test_simple_functional(self, ai):
        """Test simple functional variation."""
        await handle_advanced_tool('introduce_function', {'name': 'f'}, ai)

        result = await handle_advanced_tool(
            'calculus_of_variations',
            {
                'functional': 'Derivative(f(x), x)**2',
                'function_name': 'f',
                'independent_var': 'x'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'


class TestDynamicProgramming:
    """Test the 'dynamic_programming' tool."""

    @pytest.mark.asyncio
    async def test_inventory_problem(self, ai):
        """Test Bellman equation for inventory problem (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'dynamic_programming',
            {
                'value_function': 'V(s)',
                'state_variables': ['s'],
                'decision_variables': ['a'],
                'transition': 's - d + a'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'bellman_equation' in data or 'equation' in data

    @pytest.mark.asyncio
    async def test_optimal_control(self, ai):
        """Test dynamic programming for control problem."""
        result = await handle_advanced_tool(
            'dynamic_programming',
            {
                'value_function': 'V(x)',
                'state_variables': ['x'],
                'decision_variables': ['u'],
                'transition': 'x + u'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'

    @pytest.mark.asyncio
    async def test_multistage_decision(self, ai):
        """Test multistage decision problem."""
        result = await handle_advanced_tool(
            'dynamic_programming',
            {
                'value_function': 'V(x, t)',
                'state_variables': ['x', 't'],
                'decision_variables': ['a'],
                'transition': 'f(x, a)'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
