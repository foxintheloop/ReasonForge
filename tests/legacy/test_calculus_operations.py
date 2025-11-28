"""
Tests for core calculus operation tools.

This module tests the 6 core calculus tools from server.py:
1. solve_equations - Solve systems of equations
2. differentiate - Compute symbolic derivatives
3. integrate - Compute integrals
4. compute_limit - Calculate limits
5. expand_series - Taylor/Maclaurin series expansion
6. optimize_function - Find critical points and extrema

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI


class TestSolveEquations:
    """Test the 'solve_equations' tool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_solve_quadratic_equation(self):
        """Test solving x^2 - 5x + 6 = 0 (from USAGE_EXAMPLES.md)."""
        import sympy as sp

        # Parse equations to sympy expressions
        equations = [sp.sympify("x**2 - 5*x + 6")]
        result = self.ai.solve_equation_system(equations)

        assert result is not None
        assert 'solutions' in result
        assert len(result['solutions']) == 2

        # Solutions should be x=2 and x=3
        x_values = [sol[list(sol.keys())[0]] for sol in result['solutions']]
        assert 2 in x_values or sp.sympify('2') in x_values
        assert 3 in x_values or sp.sympify('3') in x_values
        assert result['verification'][0] is True
        assert result['verification'][1] is True

    def test_solve_linear_system(self):
        """Test solving system: 2x + 3y = 7, x - y = 1."""
        import sympy as sp

        equations = [sp.sympify("2*x + 3*y - 7"), sp.sympify("x - y - 1")]
        x, y = sp.symbols('x y')

        result = self.ai.solve_equation_system(equations, [x, y])

        assert result is not None
        assert 'solutions' in result
        assert len(result['solutions']) > 0

    def test_solve_cubic_equation(self):
        """Test solving x^3 - 6x^2 + 11x - 6 = 0."""
        import sympy as sp

        equations = [sp.sympify("x**3 - 6*x**2 + 11*x - 6")]
        result = self.ai.solve_equation_system(equations)

        assert result is not None
        assert 'solutions' in result
        # Should have 3 solutions: x=1, x=2, x=3
        assert len(result['solutions']) == 3

    def test_solve_no_real_solutions(self):
        """Test equation with no real solutions."""
        import sympy as sp

        equations = [sp.sympify("x**2 + 1")]
        result = self.ai.solve_equation_system(equations)

        assert result is not None
        assert 'solutions' in result
        # Should find complex solutions

    def test_solve_infinite_solutions(self):
        """Test equation with infinite solutions."""
        import sympy as sp

        equations = [sp.sympify("0*x")]
        result = self.ai.solve_equation_system(equations)

        assert result is not None
        # Should handle gracefully


class TestDifferentiate:
    """Test the 'differentiate' tool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_differentiate_product_exponential(self):
        """Test d/dx[sin(x)*cos(x)*e^x] (from USAGE_EXAMPLES.md)."""
        result = self.ai.perform_calculus("sin(x)*cos(x)*exp(x)", "x", "diff")

        assert result is not None
        assert 'result' in result
        # Result should contain both product rule and exponential terms
        result_str = str(result['result'])
        assert 'exp(x)' in result_str or 'e**x' in result_str

    def test_differentiate_polynomial(self):
        """Test d/dx[x^3 + 2x^2 - x + 5]."""
        result = self.ai.perform_calculus("x**3 + 2*x**2 - x + 5", "x", "diff")

        assert result is not None
        assert 'result' in result
        # Should be 3x^2 + 4x - 1
        assert "3*x**2" in str(result['result'])

    def test_differentiate_trigonometric(self):
        """Test d/dx[sin(x)]."""
        result = self.ai.perform_calculus("sin(x)", "x", "diff")

        assert result is not None
        assert 'result' in result
        assert "cos(x)" in str(result['result'])

    def test_differentiate_chain_rule(self):
        """Test d/dx[sin(x^2)]."""
        result = self.ai.perform_calculus("sin(x**2)", "x", "diff")

        assert result is not None
        assert 'result' in result
        # Should use chain rule: 2x*cos(x^2)
        result_str = str(result['result'])
        assert 'cos' in result_str and 'x' in result_str

    def test_differentiate_constant(self):
        """Test d/dx[5]."""
        result = self.ai.perform_calculus("5", "x", "diff")

        assert result is not None
        assert 'result' in result
        assert str(result['result']) == "0"


class TestIntegrate:
    """Test the 'integrate' tool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_integrate_product_logarithm(self):
        """Test ∫x^2*ln(x)dx (from USAGE_EXAMPLES.md)."""
        result = self.ai.perform_calculus("x**2 * log(x)", "x", "integrate")

        assert result is not None
        assert 'result' in result
        # Should use integration by parts
        result_str = str(result['result'])
        assert 'x**3' in result_str or 'x^3' in result_str

    def test_integrate_polynomial(self):
        """Test ∫x^2 dx."""
        result = self.ai.perform_calculus("x**2", "x", "integrate")

        assert result is not None
        assert 'result' in result
        # Should be x^3/3
        assert "x**3/3" in str(result['result'])

    def test_integrate_trigonometric(self):
        """Test ∫sin(x) dx."""
        result = self.ai.perform_calculus("sin(x)", "x", "integrate")

        assert result is not None
        assert 'result' in result
        # Should be -cos(x)
        result_str = str(result['result'])
        assert 'cos' in result_str

    def test_integrate_exponential(self):
        """Test ∫e^x dx."""
        result = self.ai.perform_calculus("exp(x)", "x", "integrate")

        assert result is not None
        assert 'result' in result
        assert "exp(x)" in str(result['result'])

    def test_integrate_rational_function(self):
        """Test ∫1/x dx."""
        result = self.ai.perform_calculus("1/x", "x", "integrate")

        assert result is not None
        assert 'result' in result
        # Should be log(x)
        assert "log" in str(result['result'])


class TestComputeLimit:
    """Test the 'compute_limit' tool (via perform_calculus)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_limit_sinx_over_x_at_zero(self):
        """Test lim(x→0) sin(x)/x = 1 (from USAGE_EXAMPLES.md)."""
        result = self.ai.perform_calculus("sin(x)/x", "x", "limit_zero")

        assert result is not None
        assert 'result' in result
        assert str(result['result']) == "1"

    def test_limit_at_infinity(self):
        """Test lim(x→∞) 1/x = 0."""
        result = self.ai.perform_calculus("1/x", "x", "limit_inf")

        assert result is not None
        assert 'result' in result
        assert str(result['result']) == "0"

    def test_limit_polynomial_at_zero(self):
        """Test lim(x→0) x^2 + 3x."""
        result = self.ai.perform_calculus("x**2 + 3*x", "x", "limit_zero")

        assert result is not None
        assert 'result' in result
        # Should be 0 at x=0
        assert str(result['result']) == "0"

    def test_limit_exponential_at_infinity(self):
        """Test lim(x→∞) e^x."""
        result = self.ai.perform_calculus("exp(x)", "x", "limit_inf")

        assert result is not None
        assert 'result' in result
        # Should be ∞
        assert 'oo' in str(result['result']) or 'Infinity' in str(result['result'])

    def test_limit_with_lhopital(self):
        """Test lim(x→0) (1-cos(x))/x^2."""
        result = self.ai.perform_calculus("(1 - cos(x))/x**2", "x", "limit_zero")

        assert result is not None
        assert 'result' in result
        # Should be 1/2 (using L'Hopital's rule)


class TestExpandSeries:
    """Test the 'expand_series' tool (via perform_calculus)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_expand_exponential_series(self):
        """Test Taylor series of e^x around 0 (from USAGE_EXAMPLES.md)."""
        result = self.ai.perform_calculus("exp(x)", "x", "series")

        assert result is not None
        assert 'result' in result
        series_str = str(result['result'])
        # Should contain 1 + x + x^2/2 + x^3/6 + ...
        assert '1' in series_str
        assert 'x' in series_str
        assert 'O(x**' in series_str or 'O(x^' in series_str

    def test_expand_sine_series(self):
        """Test Taylor series of sin(x) around 0."""
        result = self.ai.perform_calculus("sin(x)", "x", "series")

        assert result is not None
        assert 'result' in result
        # sin(x) = x - x^3/6 + x^5/120 - ...
        series_str = str(result['result'])
        assert 'x' in series_str

    def test_expand_logarithm_series(self):
        """Test Taylor series of log(1+x) around 0."""
        result = self.ai.perform_calculus("log(1 + x)", "x", "series")

        assert result is not None
        assert 'result' in result

    def test_expand_cosine_series(self):
        """Test Taylor series of cos(x) around 0."""
        result = self.ai.perform_calculus("cos(x)", "x", "series")

        assert result is not None
        assert 'result' in result

    def test_expand_rational_series(self):
        """Test series expansion of rational function."""
        result = self.ai.perform_calculus("1/(1 - x)", "x", "series")

        assert result is not None
        assert 'result' in result
        # Should be geometric series 1 + x + x^2 + ...


class TestOptimizeFunction:
    """Test the 'optimize_function' tool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_optimize_quadratic_two_variables(self):
        """Test optimization of x^2 + y^2 - 2x - 4y + 5 (from USAGE_EXAMPLES.md)."""
        result = self.ai.optimize_function("x**2 + y**2 - 2*x - 4*y + 5", ['x', 'y'])

        assert result is not None
        assert 'critical_points' in result
        assert len(result['critical_points']) > 0
        # Critical point should be at (1, 2)
        cp = result['critical_points'][0]
        assert 'x' in cp and 'y' in cp

    def test_optimize_simple_quadratic(self):
        """Test optimization of x^2 - 4x + 3."""
        result = self.ai.optimize_function("x**2 - 4*x + 3")

        assert result is not None
        assert 'critical_points' in result
        assert len(result['critical_points']) > 0
        # Critical point at x=2

    def test_optimize_cubic_function(self):
        """Test optimization of x^3 - 3x^2 + 2."""
        result = self.ai.optimize_function("x**3 - 3*x**2 + 2")

        assert result is not None
        assert 'critical_points' in result
        # Should have 2 critical points

    def test_optimize_trigonometric(self):
        """Test optimization of sin(x) + cos(x)."""
        result = self.ai.optimize_function("sin(x) + cos(x)")

        assert result is not None
        assert 'critical_points' in result

    def test_optimize_multivariate(self):
        """Test optimization of x^2 + 2xy + y^2."""
        result = self.ai.optimize_function("x**2 + 2*x*y + y**2", ['x', 'y'])

        assert result is not None
        assert 'critical_points' in result
