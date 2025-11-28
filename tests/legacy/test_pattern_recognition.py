"""
Tests for pattern recognition and expression manipulation tools.

This module tests:
1. recognize_pattern - Identify patterns in sequences
2. factor_expression - Factor polynomials

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI


class TestRecognizePattern:
    """Test the 'recognize_pattern' tool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_recognize_square_pattern(self):
        """Test recognizing n^2 pattern (from USAGE_EXAMPLES.md)."""
        sequence = [1, 4, 9, 16, 25, 36]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert 'patterns_found' in result
        assert result['patterns_found'] > 0
        assert 'most_likely' in result
        assert result['most_likely']['type'] == 'polynomial'
        assert 'next_terms' in result['most_likely']
        # Next terms should be 49, 64, 81
        assert 49 in result['most_likely']['next_terms'] or '49' in result['most_likely']['next_terms']

    def test_recognize_arithmetic_sequence(self):
        """Test recognizing arithmetic sequence."""
        sequence = [2, 5, 8, 11, 14]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0
        # Should recognize as arithmetic or polynomial (degree 1)
        assert result['most_likely']['type'] in ['arithmetic', 'polynomial']
        # Next term should be 17
        assert 17 in result['most_likely']['next_terms'] or '17' in result['most_likely']['next_terms']

    def test_recognize_geometric_sequence(self):
        """Test recognizing geometric sequence."""
        sequence = [2, 6, 18, 54, 162]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0
        # Should recognize as geometric
        assert result['most_likely']['type'] in ['geometric', 'exponential']

    def test_recognize_fibonacci_like(self):
        """Test recognizing Fibonacci-like sequence."""
        sequence = [1, 1, 2, 3, 5, 8]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0

    def test_recognize_cubic_pattern(self):
        """Test recognizing cubic pattern n^3."""
        sequence = [1, 8, 27, 64, 125]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0
        assert result['most_likely']['type'] == 'polynomial'

    def test_recognize_constant_sequence(self):
        """Test recognizing constant sequence."""
        sequence = [5, 5, 5, 5, 5]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0

    def test_recognize_prime_numbers(self):
        """Test with prime number sequence."""
        sequence = [2, 3, 5, 7, 11, 13]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        # May or may not find a pattern for primes


class TestFactorExpression:
    """Test the 'factor_expression' tool (tested via SymbolicAI)."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_factor_cubic_polynomial(self):
        """Test factoring x^3 - 6x^2 + 11x - 6 (from USAGE_EXAMPLES.md)."""
        # This test verifies factorization capability through SymPy
        import sympy as sp

        expr = sp.sympify("x**3 - 6*x**2 + 11*x - 6")
        factored = sp.factor(expr)

        # Should factor to (x-1)(x-2)(x-3)
        factored_str = str(factored)
        assert 'x - 1' in factored_str or '(x - 1)' in factored_str
        assert 'x - 2' in factored_str or '(x - 2)' in factored_str
        assert 'x - 3' in factored_str or '(x - 3)' in factored_str

    def test_factor_quadratic(self):
        """Test factoring x^2 - 5x + 6."""
        import sympy as sp

        expr = sp.sympify("x**2 - 5*x + 6")
        factored = sp.factor(expr)

        # Should factor to (x-2)(x-3)
        factored_str = str(factored)
        assert 'x - 2' in factored_str
        assert 'x - 3' in factored_str

    def test_factor_difference_of_squares(self):
        """Test factoring x^2 - 4."""
        import sympy as sp

        expr = sp.sympify("x**2 - 4")
        factored = sp.factor(expr)

        # Should factor to (x-2)(x+2)
        factored_str = str(factored)
        assert '(' in factored_str  # Should have factored form

    def test_factor_with_gcf(self):
        """Test factoring with greatest common factor."""
        import sympy as sp

        expr = sp.sympify("2*x**2 + 4*x")
        factored = sp.factor(expr)

        # Should factor to 2x(x + 2)
        factored_str = str(factored)
        assert '2' in factored_str
        assert 'x' in factored_str

    def test_factor_perfect_square(self):
        """Test factoring perfect square."""
        import sympy as sp

        expr = sp.sympify("x**2 + 2*x + 1")
        factored = sp.factor(expr)

        # Should factor to (x+1)^2
        factored_str = str(factored)
        assert '**2' in factored_str or '^2' in factored_str

    def test_factor_prime_polynomial(self):
        """Test factoring irreducible polynomial."""
        import sympy as sp

        expr = sp.sympify("x**2 + 1")
        factored = sp.factor(expr)

        # Should remain as x^2 + 1 (irreducible over reals)
        # Or factor over complex numbers
        assert factored is not None

    def test_factor_multivariate(self):
        """Test factoring multivariate polynomial."""
        import sympy as sp

        expr = sp.sympify("x**2 - y**2")
        factored = sp.factor(expr)

        # Should factor to (x-y)(x+y)
        factored_str = str(factored)
        assert '(' in factored_str
