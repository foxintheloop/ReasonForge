"""
Tests for Advanced Solver Tools (6 tools).

Tools tested:
1. solve_algebraically - Advanced algebraic equation solving
2. solve_linear_system - Linear system solver
3. solve_nonlinear_system - Nonlinear system solver
4. introduce_function - Define symbolic functions for ODEs/PDEs
5. dsolve_ode - Ordinary differential equation solver
6. pdsolve_pde - Partial differential equation solver
"""

import pytest
import json
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestSolveAlgebraically:
    """Test the 'solve_algebraically' tool for advanced equation solving."""

    @pytest.mark.asyncio
    async def test_solve_algebraically_simple_polynomial(self, ai):
        """Test solving a simple polynomial equation."""
        # Setup
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 - 4', 'key': 'eq1'},
            ai
        )

        # Test
        result = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'eq1',
                'variable': 'x',
                'domain': 'real'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'solutions' in data
        assert data['variable'] == 'x'
        assert data['domain'] == 'real'
        # Solutions should contain -2 and 2
        assert '-2' in data['solutions'] or '2' in data['solutions']

    @pytest.mark.asyncio
    async def test_solve_algebraically_complex_domain(self, ai):
        """Test solving with complex domain."""
        await handle_advanced_tool('intro', {'name': 'z'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'z**2 + 1', 'key': 'eq2'},
            ai
        )

        result = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'eq2',
                'variable': 'z',
                'domain': 'complex'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'solutions' in data
        assert data['domain'] == 'complex'
        # Should find complex solutions i and -i

    @pytest.mark.asyncio
    async def test_solve_algebraically_integers_domain(self, ai):
        """Test solving with integers domain."""
        await handle_advanced_tool('intro', {'name': 'n'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'n - 5', 'key': 'eq3'},
            ai
        )

        result = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'eq3',
                'variable': 'n',
                'domain': 'integers'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'solutions' in data
        assert '5' in data['solutions']

    @pytest.mark.asyncio
    async def test_solve_algebraically_no_solutions(self, ai):
        """Test solving an equation with no real solutions."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 + 1', 'key': 'no_sol'},
            ai
        )

        result = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'no_sol',
                'variable': 'x',
                'domain': 'real'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'solutions' in data
        # Empty set or explicit no solutions

    @pytest.mark.asyncio
    async def test_solve_algebraically_invalid_expression_key(self, ai):
        """Test solving with non-existent expression key."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        result = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'nonexistent',
                'variable': 'x',
                'domain': 'real'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestSolveLinearSystem:
    """Test the 'solve_linear_system' tool."""

    @pytest.mark.asyncio
    async def test_solve_linear_system_2x2(self, ai):
        """Test solving a 2x2 linear system."""
        # Setup
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x + y - 3', 'key': 'eq1'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x - y - 1', 'key': 'eq2'},
            ai
        )

        # Test
        result = await handle_advanced_tool(
            'solve_linear_system',
            {
                'expression_keys': ['eq1', 'eq2'],
                'variables': ['x', 'y']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'result_key' in data
        assert 'solutions' in data
        # Solution should be x=2, y=1

    @pytest.mark.asyncio
    async def test_solve_linear_system_3x3(self, ai):
        """Test solving a 3x3 linear system."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y', 'z']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x + y + z - 6', 'key': 'eq1'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x - y + z - 2', 'key': 'eq2'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': '2*x + y - z - 1', 'key': 'eq3'},
            ai
        )

        result = await handle_advanced_tool(
            'solve_linear_system',
            {
                'expression_keys': ['eq1', 'eq2', 'eq3'],
                'variables': ['x', 'y', 'z']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'solutions' in data


class TestSolveNonlinearSystem:
    """Test the 'solve_nonlinear_system' tool."""

    @pytest.mark.asyncio
    async def test_solve_nonlinear_system_simple(self, ai):
        """Test solving a simple nonlinear system."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 + y**2 - 25', 'key': 'eq1'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x - y', 'key': 'eq2'},
            ai
        )

        result = await handle_advanced_tool(
            'solve_nonlinear_system',
            {
                'expression_keys': ['eq1', 'eq2'],
                'variables': ['x', 'y']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'result_key' in data
        assert 'solutions' in data


class TestIntroduceFunction:
    """Test the 'introduce_function' tool for defining symbolic functions."""

    @pytest.mark.asyncio
    async def test_introduce_function_basic(self, ai):
        """Test introducing a basic function."""
        result = await handle_advanced_tool(
            'introduce_function',
            {'name': 'y'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['function'] == 'y'
        assert 'y' in ai.functions

    @pytest.mark.asyncio
    async def test_introduce_multiple_functions(self, ai):
        """Test introducing multiple functions."""
        functions = ['f', 'g', 'h']

        for func_name in functions:
            result = await handle_advanced_tool(
                'introduce_function',
                {'name': func_name},
                ai
            )
            data = json.loads(result[0].text)

            assert data['status'] == 'success'
            assert func_name in ai.functions

    @pytest.mark.asyncio
    async def test_introduce_function_greek_letters(self, ai):
        """Test introducing functions with Greek letter names."""
        result = await handle_advanced_tool(
            'introduce_function',
            {'name': 'psi'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'psi' in ai.functions


class TestDsolveODE:
    """Test the 'dsolve_ode' tool for solving ordinary differential equations."""

    @pytest.mark.asyncio
    async def test_dsolve_ode_first_order_simple(self, ai):
        """Test solving a simple first-order ODE: dy/dt = y."""
        # Setup
        await handle_advanced_tool('intro', {'name': 't'}, ai)
        await handle_advanced_tool('introduce_function', {'name': 'y'}, ai)

        # Create ODE: Derivative(y(t), t) - y(t) = 0
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'Derivative(y(t), t) - y(t)', 'key': 'ode1'},
            ai
        )

        # Solve
        result = await handle_advanced_tool(
            'dsolve_ode',
            {
                'equation_key': 'ode1',
                'function_name': 'y'
            },
            ai
        )
        data = json.loads(result[0].text)

        # Debug
        import sys
        print(f"DEBUG ODE: {data}", file=sys.stderr)

        assert data['status'] == 'success' or 'error' in data
        assert 'result_key' in data
        assert 'solution' in data
        # Solution should involve exp(t)

    @pytest.mark.asyncio
    async def test_dsolve_ode_with_initial_condition(self, ai):
        """Test solving an ODE with initial conditions."""
        await handle_advanced_tool('intro', {'name': 't'}, ai)
        await handle_advanced_tool('introduce_function', {'name': 'y'}, ai)

        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'Derivative(y(t), t) - y(t)', 'key': 'ode2'},
            ai
        )

        result = await handle_advanced_tool(
            'dsolve_ode',
            {
                'equation_key': 'ode2',
                'function_name': 'y',
                'ics': {'y(0)': '1'}
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'solution' in data


    @pytest.mark.asyncio
    async def test_dsolve_ode_invalid_equation(self, ai):
        """Test solving with invalid equation key."""
        await handle_advanced_tool('introduce_function', {'name': 'y'}, ai)

        result = await handle_advanced_tool(
            'dsolve_ode',
            {
                'equation_key': 'nonexistent',
                'function_name': 'y'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestPdsolvePDE:
    """Test the 'pdsolve_pde' tool for solving partial differential equations."""

    @pytest.mark.asyncio
    async def test_pdsolve_pde_simple(self, ai):
        """Test solving a simple PDE."""
        # Setup
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool('introduce_function', {'name': 'u'}, ai)

        # Simple PDE: du/dx = 0
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'Derivative(u(x, y), x)', 'key': 'pde1'},
            ai
        )

        # Solve
        result = await handle_advanced_tool(
            'pdsolve_pde',
            {
                'equation_key': 'pde1',
                'function_name': 'u'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'result_key' in data
        assert 'solution' in data

    @pytest.mark.asyncio
    async def test_pdsolve_pde_invalid_function(self, ai):
        """Test PDE solving with non-existent function."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x + y', 'key': 'pde_test'},
            ai
        )

        result = await handle_advanced_tool(
            'pdsolve_pde',
            {
                'equation_key': 'pde_test',
                'function_name': 'nonexistent_func'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data
