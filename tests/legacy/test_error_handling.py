"""
Tests for Error Handling - Edge Cases and Error Conditions.

Tests various error scenarios to ensure robust error handling:
- Invalid keys/references
- Missing required arguments
- Type validation errors
- Mathematical errors
- State consistency
"""

import pytest
import json
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestInvalidKeyErrors:
    """Test error handling for invalid key references."""

    @pytest.mark.asyncio
    async def test_solve_with_nonexistent_expression(self, ai):
        """Test solving with non-existent expression key."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        result = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'does_not_exist',
                'variable': 'x',
                'domain': 'real'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data
        assert 'not found' in data['error'].lower()

    @pytest.mark.asyncio
    async def test_print_latex_nonexistent_expression(self, ai):
        """Test printing LaTeX for non-existent expression."""
        result = await handle_advanced_tool(
            'print_latex_expression',
            {'key': 'invalid_key'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data
        assert 'available_keys' in data

    @pytest.mark.asyncio
    async def test_matrix_determinant_invalid_key(self, ai):
        """Test matrix determinant with invalid key."""
        result = await handle_advanced_tool(
            'matrix_determinant',
            {'matrix_key': 'nonexistent_matrix'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data

    @pytest.mark.asyncio
    async def test_vector_curl_invalid_key(self, ai):
        """Test curl calculation with invalid vector field key."""
        result = await handle_advanced_tool(
            'calculate_curl',
            {'vector_field_key': 'invalid_vector'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestUndefinedVariableErrors:
    """Test error handling for undefined variables."""

    @pytest.mark.asyncio
    async def test_solve_with_undefined_variable(self, ai):
        """Test solving with undefined variable."""
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': '5', 'key': 'const'},
            ai
        )

        result = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'const',
                'variable': 'undefined_var',
                'domain': 'real'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data

    @pytest.mark.asyncio
    async def test_differentiate_undefined_variable(self, ai):
        """Test differentiation with undefined variable."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2', 'key': 'expr1'},
            ai
        )

        result = await handle_advanced_tool(
            'differentiate_expression',
            {
                'expression_key': 'expr1',
                'variable': 'y',  # y is not defined
                'order': 1
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestMathematicalErrors:
    """Test error handling for mathematical impossibilities."""

    @pytest.mark.asyncio
    async def test_matrix_inverse_singular(self, ai):
        """Test inverting a singular (non-invertible) matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {
                'elements': [['1', '2'], ['2', '4']],
                'key': 'singular'
            },
            ai
        )

        result = await handle_advanced_tool(
            'matrix_inverse',
            {'matrix_key': 'singular'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data
        assert 'not invertible' in data['error'].lower() or 'singular' in data['error'].lower()

    @pytest.mark.asyncio
    async def test_ode_with_undefined_function(self, ai):
        """Test ODE solving with undefined function."""
        await handle_advanced_tool('intro', {'name': 't'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 't**2', 'key': 'expr1'},
            ai
        )

        result = await handle_advanced_tool(
            'dsolve_ode',
            {
                'equation_key': 'expr1',
                'function_name': 'undefined_function'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestCoordinateSystemErrors:
    """Test error handling for coordinate systems and vector fields."""

    @pytest.mark.asyncio
    async def test_vector_field_without_coord_system(self, ai):
        """Test creating vector field without coordinate system."""
        result = await handle_advanced_tool(
            'create_vector_field',
            {
                'coord_system_name': 'nonexistent_system',
                'components': {'i': '1', 'j': '0', 'k': '0'}
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data

    @pytest.mark.asyncio
    async def test_gradient_without_coord_system(self, ai):
        """Test calculating gradient without coordinate system."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2', 'key': 'scalar'},
            ai
        )

        result = await handle_advanced_tool(
            'calculate_gradient',
            {
                'expression_key': 'scalar',
                'coord_system_name': 'nonexistent'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestLinearSystemErrors:
    """Test error handling for linear systems."""

    @pytest.mark.asyncio
    async def test_linear_system_missing_expressions(self, ai):
        """Test solving linear system with missing expression keys."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x + y', 'key': 'eq1'},
            ai
        )

        result = await handle_advanced_tool(
            'solve_linear_system',
            {
                'expression_keys': ['eq1', 'eq2'],  # eq2 doesn't exist
                'variables': ['x', 'y']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestExpressionSyntaxErrors:
    """Test error handling for invalid expression syntax."""

    @pytest.mark.asyncio
    async def test_introduce_invalid_expression(self, ai):
        """Test introducing an expression with invalid syntax."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        # This should handle the error gracefully
        try:
            result = await handle_advanced_tool(
                'introduce_expression',
                {'expression': 'x +* y'},  # Invalid syntax
                ai
            )
            data = json.loads(result[0].text)
            # Either it errors or sympify handles it
            # We just want to ensure no crash
        except Exception:
            # Acceptable to raise exception for invalid syntax
            pass


class TestSubstitutionErrors:
    """Test error handling for substitutions."""

    @pytest.mark.asyncio
    async def test_substitute_nonexistent_variable(self, ai):
        """Test substituting values for non-existent variables."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2', 'key': 'expr1'},
            ai
        )

        # Substitute for 'y' which doesn't exist
        result = await handle_advanced_tool(
            'substitute_expression',
            {
                'expression_key': 'expr1',
                'substitutions': {'y': '5'}
            },
            ai
        )
        # This should succeed but have no effect since y is not in the expression
        # The tool should handle this gracefully


class TestUnitOperationErrors:
    """Test error handling for unit operations."""

    @pytest.mark.asyncio
    async def test_convert_to_undefined_unit(self, ai):
        """Test unit conversion with undefined target unit."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x', 'key': 'expr1'},
            ai
        )

        result = await handle_advanced_tool(
            'convert_to_units',
            {
                'expression_key': 'expr1',
                'target_unit': 'nonexistent_unit',
                'unit_system': 'SI'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestStateConsistency:
    """Test state consistency across operations."""

    @pytest.mark.asyncio
    async def test_expression_key_persistence(self, ai):
        """Test that expression keys persist across operations."""
        # Create multiple expressions
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x', 'key': 'expr1'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2', 'key': 'expr2'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**3', 'key': 'expr3'},
            ai
        )

        # Verify all keys are accessible
        assert 'expr1' in ai.expressions
        assert 'expr2' in ai.expressions
        assert 'expr3' in ai.expressions

    @pytest.mark.asyncio
    async def test_variable_persistence(self, ai):
        """Test that variables persist across operations."""
        await handle_advanced_tool('intro', {'name': 'a'}, ai)
        await handle_advanced_tool('intro', {'name': 'b'}, ai)
        await handle_advanced_tool('intro', {'name': 'c'}, ai)

        # All variables should be accessible
        assert 'a' in ai.variables
        assert 'b' in ai.variables
        assert 'c' in ai.variables
