"""
Tests for Variable & Expression Management Tools (4 tools).

Tools tested:
1. intro - Introduce single symbolic variable
2. intro_many - Introduce multiple symbolic variables
3. introduce_expression - Store named expression for reuse
4. print_latex_expression - Format expression as LaTeX
"""

import pytest
import json
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestIntro:
    """Test the 'intro' tool for introducing single variables."""

    @pytest.mark.asyncio
    async def test_intro_basic(self, ai):
        """Test introducing a variable with no assumptions."""
        result = await handle_advanced_tool('intro', {'name': 'x'}, ai)
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['variable'] == 'x'
        assert 'x' in ai.variables
        assert data['assumptions'] == {}

    @pytest.mark.asyncio
    async def test_intro_with_positive_assumption(self, ai):
        """Test introducing a variable with positive assumption."""
        result = await handle_advanced_tool(
            'intro',
            {
                'name': 'a',
                'positive_assumptions': ['positive', 'real']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['variable'] == 'a'
        assert 'a' in ai.variables
        assert data['assumptions']['positive'] is True
        assert data['assumptions']['real'] is True

        # Verify the variable has the assumptions
        var = ai.variables['a']
        assert var.is_positive is True
        assert var.is_real is True

    @pytest.mark.asyncio
    async def test_intro_with_negative_assumptions(self, ai):
        """Test introducing a variable with negative assumptions (explicit False)."""
        result = await handle_advanced_tool(
            'intro',
            {
                'name': 'b',
                'negative_assumptions': ['positive']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['variable'] == 'b'
        assert data['assumptions']['positive'] is False

    @pytest.mark.asyncio
    async def test_intro_with_mixed_assumptions(self, ai):
        """Test variable with both positive and negative assumptions."""
        result = await handle_advanced_tool(
            'intro',
            {
                'name': 'c',
                'positive_assumptions': ['real', 'integer'],
                'negative_assumptions': ['positive']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['assumptions']['real'] is True
        assert data['assumptions']['integer'] is True
        assert data['assumptions']['positive'] is False

    @pytest.mark.asyncio
    async def test_intro_complex_variable_name(self, ai):
        """Test introducing variables with complex names."""
        test_names = ['alpha', 'x_1', 'theta', 'Delta']

        for name in test_names:
            result = await handle_advanced_tool('intro', {'name': name}, ai)
            data = json.loads(result[0].text)

            assert data['status'] == 'success'
            assert data['variable'] == name
            assert name in ai.variables

    @pytest.mark.asyncio
    async def test_intro_overwrites_existing_variable(self, ai):
        """Test that introducing a variable overwrites existing one."""
        # First introduction
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        # Second introduction with different assumptions
        result = await handle_advanced_tool(
            'intro',
            {
                'name': 'x',
                'positive_assumptions': ['positive']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert ai.variables['x'].is_positive is True


class TestIntroMany:
    """Test the 'intro_many' tool for introducing multiple variables."""

    @pytest.mark.asyncio
    async def test_intro_many_basic(self, ai):
        """Test introducing multiple variables with no assumptions."""
        result = await handle_advanced_tool(
            'intro_many',
            {'names': ['x', 'y', 'z']},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['variables'] == ['x', 'y', 'z']
        assert 'x' in ai.variables
        assert 'y' in ai.variables
        assert 'z' in ai.variables
        assert data['assumptions'] == {}

    @pytest.mark.asyncio
    async def test_intro_many_with_assumptions(self, ai):
        """Test introducing multiple variables with shared assumptions."""
        result = await handle_advanced_tool(
            'intro_many',
            {
                'names': ['a', 'b', 'c'],
                'positive_assumptions': ['real', 'positive']
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert len(data['variables']) == 3
        assert data['assumptions']['real'] is True
        assert data['assumptions']['positive'] is True

        # Verify all variables have the assumptions
        for var_name in ['a', 'b', 'c']:
            var = ai.variables[var_name]
            assert var.is_real is True
            assert var.is_positive is True

    @pytest.mark.asyncio
    async def test_intro_many_single_variable(self, ai):
        """Test intro_many with a single variable."""
        result = await handle_advanced_tool(
            'intro_many',
            {'names': ['x']},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['variables'] == ['x']
        assert 'x' in ai.variables

    @pytest.mark.asyncio
    async def test_intro_many_large_set(self, ai):
        """Test introducing a large set of variables."""
        names = [f'x{i}' for i in range(10)]
        result = await handle_advanced_tool(
            'intro_many',
            {'names': names},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert len(data['variables']) == 10
        assert all(name in ai.variables for name in names)


class TestIntroduceExpression:
    """Test the 'introduce_expression' tool for storing named expressions."""

    @pytest.mark.asyncio
    async def test_introduce_expression_simple(self, ai):
        """Test introducing a simple expression."""
        # First introduce variable
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        # Then introduce expression
        result = await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 + 1'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'key' in data
        assert data['key'] in ai.expressions
        assert data['expression'] == 'x**2 + 1'
        assert 'latex' in data
        assert 'pretty' in data

    @pytest.mark.asyncio
    async def test_introduce_expression_with_custom_key(self, ai):
        """Test introducing an expression with a custom key."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        result = await handle_advanced_tool(
            'introduce_expression',
            {
                'expression': 'x**2',
                'key': 'my_expression'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['key'] == 'my_expression'
        assert 'my_expression' in ai.expressions

    @pytest.mark.asyncio
    async def test_introduce_expression_auto_key_generation(self, ai):
        """Test that auto-generated keys are sequential."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        # Introduce multiple expressions
        keys = []
        for i in range(3):
            result = await handle_advanced_tool(
                'introduce_expression',
                {'expression': f'x**{i+1}'},
                ai
            )
            data = json.loads(result[0].text)
            keys.append(data['key'])

        # Keys should be sequential
        assert 'expr_0' in keys
        assert 'expr_1' in keys
        assert 'expr_2' in keys

    @pytest.mark.asyncio
    async def test_introduce_expression_complex(self, ai):
        """Test introducing a complex nested expression."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)

        result = await handle_advanced_tool(
            'introduce_expression',
            {'expression': '(x + y)**2 - (x**2 + 2*x*y + y**2)'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        # This expression should simplify to 0
        assert 'expression' in data

    @pytest.mark.asyncio
    async def test_introduce_expression_using_functions(self, ai):
        """Test introducing an expression using mathematical functions."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        result = await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'sin(x)**2 + cos(x)**2'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'sin' in data['expression'] or '1' in data['expression']  # May be simplified

    @pytest.mark.asyncio
    async def test_introduce_expression_overwrites_key(self, ai):
        """Test that using the same key overwrites the previous expression."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        # First expression
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x', 'key': 'my_expr'},
            ai
        )

        # Second expression with same key
        result = await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2', 'key': 'my_expr'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert str(ai.expressions['my_expr']) == 'x**2'


class TestPrintLatexExpression:
    """Test the 'print_latex_expression' tool for formatting expressions."""

    @pytest.mark.asyncio
    async def test_print_latex_basic(self, ai):
        """Test printing LaTeX for a simple expression."""
        # Set up
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        setup_result = await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2', 'key': 'expr1'},
            ai
        )

        # Test
        result = await handle_advanced_tool(
            'print_latex_expression',
            {'key': 'expr1'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'key' in data
        assert data['key'] == 'expr1'
        assert 'latex' in data
        assert 'pretty' in data
        assert 'expression' in data

    @pytest.mark.asyncio
    async def test_print_latex_complex_expression(self, ai):
        """Test printing LaTeX for a complex expression."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {
                'expression': '(x + y)**3',
                'key': 'complex_expr'
            },
            ai
        )

        result = await handle_advanced_tool(
            'print_latex_expression',
            {'key': 'complex_expr'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['key'] == 'complex_expr'
        assert 'latex' in data
        assert len(data['latex']) > 0  # LaTeX string should not be empty

    @pytest.mark.asyncio
    async def test_print_latex_invalid_key(self, ai):
        """Test printing LaTeX with an invalid expression key."""
        result = await handle_advanced_tool(
            'print_latex_expression',
            {'key': 'nonexistent_key'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data
        assert 'not found' in data['error']
        assert 'available_keys' in data

    @pytest.mark.asyncio
    async def test_print_latex_trigonometric(self, ai):
        """Test LaTeX formatting for trigonometric expressions."""
        await handle_advanced_tool('intro', {'name': 'theta'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'sin(2*theta) + cos(theta)**2', 'key': 'trig'},
            ai
        )

        result = await handle_advanced_tool(
            'print_latex_expression',
            {'key': 'trig'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'latex' in data
        # LaTeX should contain sin/cos formatting
        assert 'sin' in data['latex'] or 'cos' in data['latex']

    @pytest.mark.asyncio
    async def test_print_latex_rational_expression(self, ai):
        """Test LaTeX formatting for rational expressions."""
        await handle_advanced_tool('intro', {'name': 'x'}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': '(x**2 - 1)/(x - 1)', 'key': 'rational'},
            ai
        )

        result = await handle_advanced_tool(
            'print_latex_expression',
            {'key': 'rational'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'latex' in data
        assert len(data['latex']) > 0
