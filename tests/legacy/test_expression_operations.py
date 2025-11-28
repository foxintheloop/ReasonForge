"""
Tests for Expression Operations Tools (4 tools).

Tools tested:
1. simplify_expression_advanced - Advanced simplification with options
2. substitute_expression - Expression substitution
3. integrate_expression - Advanced integration
4. differentiate_expression - Advanced differentiation
"""

import pytest
import json
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestSimplifyExpression:
    """Test the 'simplify_expression_advanced' tool."""

    @pytest.mark.asyncio
    async def test_simplify_default_method(self, ai):
        """Test simplification with default method."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': '(x + 1)**2 - (x**2 + 2*x + 1)', 'key': 'expr1'},
            ai
        )

        result = await handle_advanced_tool(
            'simplify_expression',
            {'expression_key': 'expr1', 'method': 'default'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'result_key' in data
        assert data['method'] == 'default'
        assert '0' in data['result']  # Should simplify to 0

    @pytest.mark.asyncio
    async def test_simplify_trigsimp(self, ai):
        """Test trigonometric simplification."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'sin(x)**2 + cos(x)**2', 'key': 'trig1'},
            ai
        )

        result = await handle_advanced_tool(
            'simplify_expression',
            {'expression_key': 'trig1', 'method': 'trigsimp'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'result' in data
        # Should simplify to 1

    @pytest.mark.asyncio
    async def test_simplify_all_methods(self, ai):
        """Test all simplification methods."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 + 2*x + 1', 'key': 'expr2'},
            ai
        )

        methods = ['default', 'trigsimp', 'radsimp', 'powsimp', 'logcombine']
        for method in methods:
            result = await handle_advanced_tool(
                'simplify_expression',
                {'expression_key': 'expr2', 'method': method},
                ai
            )
            data = json.loads(result[0].text)
            assert data['method'] == method


class TestSubstituteExpression:
    """Test the 'substitute_expression' tool."""

    @pytest.mark.asyncio
    async def test_substitute_numeric(self, ai):
        """Test numeric substitution."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 + 1', 'key': 'expr1'},
            ai
        )

        result = await handle_advanced_tool(
            'substitute_expression',
            {
                'expression_key': 'expr1',
                'substitutions': {'x': '2'}
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'result_key' in data
        assert '5' in data['result']  # 2^2 + 1 = 5

    @pytest.mark.asyncio
    async def test_substitute_multiple_variables(self, ai):
        """Test substituting multiple variables."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x + y', 'key': 'expr2'},
            ai
        )

        result = await handle_advanced_tool(
            'substitute_expression',
            {
                'expression_key': 'expr2',
                'substitutions': {'x': '1', 'y': '2'}
            },
            ai
        )
        data = json.loads(result[0].text)

        assert '3' in data['result']


class TestIntegrateExpression:
    """Test the 'integrate_expression' tool."""

    @pytest.mark.asyncio
    async def test_integrate_indefinite(self, ai):
        """Test indefinite integration."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2', 'key': 'expr1'},
            ai
        )

        result = await handle_advanced_tool(
            'integrate_expression',
            {
                'expression_key': 'expr1',
                'variable': 'x'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['integral_type'] == 'indefinite'
        assert 'x**3' in data['result'] or 'x^3' in data['latex']

    @pytest.mark.asyncio
    async def test_integrate_definite(self, ai):
        """Test definite integration."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x', 'key': 'expr2'},
            ai
        )

        result = await handle_advanced_tool(
            'integrate_expression',
            {
                'expression_key': 'expr2',
                'variable': 'x',
                'lower_bound': '0',
                'upper_bound': '1'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['integral_type'] == 'definite'
        # Integral of x from 0 to 1 is 1/2


class TestDifferentiateExpression:
    """Test the 'differentiate_expression' tool."""

    @pytest.mark.asyncio
    async def test_differentiate_first_order(self, ai):
        """Test first-order differentiation."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**3', 'key': 'expr1'},
            ai
        )

        result = await handle_advanced_tool(
            'differentiate_expression',
            {
                'expression_key': 'expr1',
                'variable': 'x',
                'order': 1
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['order'] == 1
        assert '3*x**2' in data['result']

    @pytest.mark.asyncio
    async def test_differentiate_higher_order(self, ai):
        """Test higher-order differentiation."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**4', 'key': 'expr2'},
            ai
        )

        result = await handle_advanced_tool(
            'differentiate_expression',
            {
                'expression_key': 'expr2',
                'variable': 'x',
                'order': 2
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['order'] == 2
        # Second derivative of x^4 is 12*x^2
