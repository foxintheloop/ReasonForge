"""
Comprehensive tests for reasonforge-expressions MCP server.

Tests all 15 tools:
- Variable Management (4 tools): intro, intro_many, introduce_expression, introduce_function
- Expression Operations (5 tools): simplify_expression, substitute_expression, substitute_values, expand_expression, factor_expression
- Basic Calculus (4 tools): differentiate, integrate, compute_limit, expand_series
- Utilities (2 tools): print_latex_expression, solve_word_problem
"""

import asyncio
import json
import pytest

from reasonforge_expressions.server import server as expressions_server


class TestVariableManagement:
    """Test variable management tools (4 tools)."""

    @pytest.mark.asyncio
    async def test_intro_basic(self):
        """Test introducing a single variable."""
        result = await expressions_server.call_tool_for_test(
            "intro",
            {"name": "x"}
        )
        data = json.loads(result[0].text)

        assert data["name"] == "x"
        assert data["variable"] == "x"
        assert "latex" in data
        assert data["assumptions"] == {}

    @pytest.mark.asyncio
    async def test_intro_with_assumptions(self):
        """Test introducing a variable with assumptions."""
        result = await expressions_server.call_tool_for_test(
            "intro",
            {
                "name": "a",
                "positive_assumptions": ["real", "positive"]
            }
        )
        data = json.loads(result[0].text)

        assert data["name"] == "a"
        assert data["assumptions"]["real"] == True
        assert data["assumptions"]["positive"] == True

    @pytest.mark.asyncio
    async def test_intro_many(self):
        """Test introducing multiple variables."""
        result = await expressions_server.call_tool_for_test(
            "intro_many",
            {
                "names": ["x", "y", "z"],
                "positive_assumptions": ["real"]
            }
        )
        data = json.loads(result[0].text)

        assert data["names"] == ["x", "y", "z"]
        assert data["count"] == 3
        assert len(data["variables"]) == 3
        assert data["assumptions"]["real"] == True

    @pytest.mark.asyncio
    async def test_introduce_expression(self):
        """Test storing an expression."""
        result = await expressions_server.call_tool_for_test(
            "introduce_expression",
            {
                "expression": "x**2 + 2*x + 1",
                "key": "my_expr"
            }
        )
        data = json.loads(result[0].text)

        assert data["key"] == "my_expr"
        assert data["stored"] == True
        assert "x**2" in data["expression"] or "x^2" in data["expression"]
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_introduce_expression_auto_key(self):
        """Test storing an expression with auto-generated key."""
        result = await expressions_server.call_tool_for_test(
            "introduce_expression",
            {"expression": "sin(x) + cos(x)"}
        )
        data = json.loads(result[0].text)

        assert data["stored"] == True
        assert "expr_" in data["key"]  # Auto-generated key
        assert "sin" in data["expression"]

    @pytest.mark.asyncio
    async def test_introduce_function(self):
        """Test defining a function symbol."""
        result = await expressions_server.call_tool_for_test(
            "introduce_function",
            {"name": "f"}
        )
        data = json.loads(result[0].text)

        assert data["name"] == "f"
        assert data["type"] == "undefined_function"
        assert "f(x)" in data["usage"]


class TestExpressionOperations:
    """Test expression operation tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_simplify_expression_default(self):
        """Test basic simplification."""
        result = await expressions_server.call_tool_for_test(
            "simplify_expression",
            {
                "expression_key": "(x + 1)**2 - (x**2 + 2*x + 1)",
                "method": "simplify"
            }
        )
        data = json.loads(result[0].text)

        assert data["method"] == "simplify"
        assert data["simplified"] == "0"
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_simplify_expression_trigsimp(self):
        """Test trigonometric simplification."""
        result = await expressions_server.call_tool_for_test(
            "simplify_expression",
            {
                "expression_key": "sin(x)**2 + cos(x)**2",
                "method": "trigsimp"
            }
        )
        data = json.loads(result[0].text)

        assert data["method"] == "trigsimp"
        assert "1" in data["simplified"]  # Should simplify to 1

    @pytest.mark.asyncio
    async def test_simplify_expression_ratsimp(self):
        """Test rational simplification."""
        result = await expressions_server.call_tool_for_test(
            "simplify_expression",
            {
                "expression_key": "(x**2 - 1)/(x - 1)",
                "method": "ratsimp"
            }
        )
        data = json.loads(result[0].text)

        assert data["method"] == "ratsimp"
        # Should simplify (x^2 - 1)/(x - 1) to (x + 1)

    @pytest.mark.asyncio
    async def test_substitute_expression(self):
        """Test expression substitution."""
        # First, introduce and store an expression
        await expressions_server.call_tool_for_test(
            "introduce_expression",
            {
                "expression": "x**2 + y",
                "key": "test_expr"
            }
        )

        # Then substitute
        result = await expressions_server.call_tool_for_test(
            "substitute_expression",
            {
                "expression_key": "test_expr",
                "substitutions": {"x": "2", "y": "3"}
            }
        )
        data = json.loads(result[0].text)

        assert data["result"] == "7"  # 2^2 + 3 = 7
        assert "stored_key" in data

    @pytest.mark.asyncio
    async def test_substitute_expression_symbolic(self):
        """Test symbolic substitution."""
        result = await expressions_server.call_tool_for_test(
            "substitute_expression",
            {
                "expression_key": "x**2 + 1",
                "substitutions": {"x": "y + 1"}
            }
        )
        data = json.loads(result[0].text)

        assert "y" in data["result"]
        assert "stored_key" in data

    @pytest.mark.asyncio
    async def test_substitute_values(self):
        """Test numeric value substitution."""
        result = await expressions_server.call_tool_for_test(
            "substitute_values",
            {
                "expression": "x**2 + 2*x + 1",
                "substitutions": {"x": "3"}
            }
        )
        data = json.loads(result[0].text)

        assert data["result"] == "16"  # (3)^2 + 2(3) + 1 = 16

    @pytest.mark.asyncio
    async def test_expand_expression(self):
        """Test expression expansion."""
        result = await expressions_server.call_tool_for_test(
            "expand_expression",
            {"expression": "(x + y)**2"}
        )
        data = json.loads(result[0].text)

        expanded = data["expanded"]
        assert "x**2" in expanded or "x^2" in expanded
        assert "y**2" in expanded or "y^2" in expanded
        assert "x*y" in expanded or "xy" in expanded

    @pytest.mark.asyncio
    async def test_expand_expression_complex(self):
        """Test expanding complex expression."""
        result = await expressions_server.call_tool_for_test(
            "expand_expression",
            {"expression": "(x + 1)*(x - 1)*(x + 2)"}
        )
        data = json.loads(result[0].text)

        assert "x**3" in data["expanded"] or "x^3" in data["expanded"]

    @pytest.mark.asyncio
    async def test_factor_expression(self):
        """Test expression factoring."""
        result = await expressions_server.call_tool_for_test(
            "factor_expression",
            {"expression": "x**2 - 1"}
        )
        data = json.loads(result[0].text)

        factored = data["factored"]
        assert "x - 1" in factored or "(x - 1)" in factored
        assert "x + 1" in factored or "(x + 1)" in factored

    @pytest.mark.asyncio
    async def test_factor_expression_quadratic(self):
        """Test factoring quadratic expression."""
        result = await expressions_server.call_tool_for_test(
            "factor_expression",
            {"expression": "x**2 + 5*x + 6"}
        )
        data = json.loads(result[0].text)

        # x^2 + 5x + 6 = (x + 2)(x + 3)
        assert "factored" in data


class TestBasicCalculus:
    """Test calculus tools (4 tools)."""

    @pytest.mark.asyncio
    async def test_differentiate_first_order(self):
        """Test first-order differentiation."""
        result = await expressions_server.call_tool_for_test(
            "differentiate",
            {
                "expression": "x**3 + 2*x**2 + x",
                "variable": "x",
                "order": 1
            }
        )
        data = json.loads(result[0].text)

        assert data["order"] == 1
        assert "3*x**2" in data["result"]
        assert "4*x" in data["result"]

    @pytest.mark.asyncio
    async def test_differentiate_second_order(self):
        """Test second-order differentiation."""
        result = await expressions_server.call_tool_for_test(
            "differentiate",
            {
                "expression": "x**4",
                "variable": "x",
                "order": 2
            }
        )
        data = json.loads(result[0].text)

        assert data["order"] == 2
        assert "12*x**2" in data["result"]  # d²/dx²(x^4) = 12x^2

    @pytest.mark.asyncio
    async def test_differentiate_trig(self):
        """Test differentiation of trigonometric functions."""
        result = await expressions_server.call_tool_for_test(
            "differentiate",
            {
                "expression": "sin(x)",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "cos" in data["result"]

    @pytest.mark.asyncio
    async def test_integrate_polynomial(self):
        """Test integration of polynomial."""
        result = await expressions_server.call_tool_for_test(
            "integrate",
            {
                "expression": "x**2",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "x**3/3" in data["result"] or "x^3/3" in data["result"]

    @pytest.mark.asyncio
    async def test_integrate_trig(self):
        """Test integration of trigonometric function."""
        result = await expressions_server.call_tool_for_test(
            "integrate",
            {
                "expression": "cos(x)",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "sin" in data["result"]

    @pytest.mark.asyncio
    async def test_integrate_exponential(self):
        """Test integration of exponential function."""
        result = await expressions_server.call_tool_for_test(
            "integrate",
            {
                "expression": "exp(x)",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "exp" in data["result"]

    @pytest.mark.asyncio
    async def test_compute_limit_finite(self):
        """Test limit at finite point."""
        result = await expressions_server.call_tool_for_test(
            "compute_limit",
            {
                "expression": "(x**2 - 1)/(x - 1)",
                "variable": "x",
                "point": "1"
            }
        )
        data = json.loads(result[0].text)

        assert data["result"] == "2"  # Limit is 2

    @pytest.mark.asyncio
    async def test_compute_limit_infinity(self):
        """Test limit at infinity."""
        result = await expressions_server.call_tool_for_test(
            "compute_limit",
            {
                "expression": "1/x",
                "variable": "x",
                "point": "inf"
            }
        )
        data = json.loads(result[0].text)

        assert data["result"] == "0"  # Limit is 0

    @pytest.mark.asyncio
    async def test_compute_limit_zero(self):
        """Test limit at zero."""
        result = await expressions_server.call_tool_for_test(
            "compute_limit",
            {
                "expression": "sin(x)/x",
                "variable": "x",
                "point": "zero"
            }
        )
        data = json.loads(result[0].text)

        assert data["result"] == "1"  # Classic limit

    @pytest.mark.asyncio
    async def test_expand_series_sin(self):
        """Test Taylor series for sin(x)."""
        result = await expressions_server.call_tool_for_test(
            "expand_series",
            {
                "expression": "sin(x)",
                "variable": "x",
                "point": 0,
                "order": 6
            }
        )
        data = json.loads(result[0].text)

        assert data["point"] == 0
        assert data["order"] == 6
        assert "x" in data["result"]
        assert "x**3" in data["result"] or "x^3" in data["result"]

    @pytest.mark.asyncio
    async def test_expand_series_exp(self):
        """Test Taylor series for exp(x)."""
        result = await expressions_server.call_tool_for_test(
            "expand_series",
            {
                "expression": "exp(x)",
                "variable": "x",
                "point": 0,
                "order": 5
            }
        )
        data = json.loads(result[0].text)

        result_str = data["result"]
        assert "1" in result_str
        assert "x" in result_str
        assert "x**2" in result_str or "x^2" in result_str

    @pytest.mark.asyncio
    async def test_expand_series_custom_point(self):
        """Test Taylor series around non-zero point."""
        result = await expressions_server.call_tool_for_test(
            "expand_series",
            {
                "expression": "x**2",
                "variable": "x",
                "point": 1,
                "order": 3
            }
        )
        data = json.loads(result[0].text)

        assert data["point"] == 1
        # Expansion around x=1


class TestUtilities:
    """Test utility tools (2 tools)."""

    @pytest.mark.asyncio
    async def test_print_latex_expression(self):
        """Test LaTeX printing for stored expression."""
        # First store an expression
        await expressions_server.call_tool_for_test(
            "introduce_expression",
            {
                "expression": "x**2 + sqrt(y)",
                "key": "latex_test"
            }
        )

        # Then get its LaTeX representation
        result = await expressions_server.call_tool_for_test(
            "print_latex_expression",
            {"key": "latex_test"}
        )
        data = json.loads(result[0].text)

        assert data["key"] == "latex_test"
        assert "latex" in data
        assert "mathjax" in data
        assert "$$" in data["mathjax"]

    @pytest.mark.asyncio
    async def test_print_latex_expression_not_found(self):
        """Test LaTeX printing for non-existent key."""
        result = await expressions_server.call_tool_for_test(
            "print_latex_expression",
            {"key": "nonexistent"}
        )
        data = json.loads(result[0].text)

        assert "error" in data

    @pytest.mark.asyncio
    async def test_solve_word_problem_simple(self):
        """Test solving a simple word problem."""
        result = await expressions_server.call_tool_for_test(
            "solve_word_problem",
            {
                "problem": "Find two numbers whose sum is 10 and difference is 2",
                "equations": ["x + y - 10", "x - y - 2"],
                "unknowns": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["problem"] == "Find two numbers whose sum is 10 and difference is 2"
        assert "solutions" in data
        assert "interpretation" in data
        # Solution should be x=6, y=4

    @pytest.mark.asyncio
    async def test_solve_word_problem_with_interpretation(self):
        """Test word problem solving with interpretation."""
        result = await expressions_server.call_tool_for_test(
            "solve_word_problem",
            {
                "problem": "A rectangle has perimeter 20 and length is twice the width",
                "equations": ["2*l + 2*w - 20", "l - 2*w"],
                "unknowns": ["l", "w"]
            }
        )
        data = json.loads(result[0].text)

        assert "solutions" in data
        assert "equations" in data
        assert len(data["equations"]) == 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
