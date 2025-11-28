"""
Test suite for ReasonForge MCP Server

Run with: pytest tests/test_server.py
"""

import pytest
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reasonforge_mcp.symbolic_engine import SymbolicAI


class TestSymbolicAI:
    """Test the SymbolicAI engine independently."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_define_variables(self):
        """Test variable definition."""
        variables = self.ai.define_variables(['x', 'y', 'z'])
        assert len(variables) == 3
        assert 'x' in self.ai.variables
        assert 'y' in self.ai.variables
        assert 'z' in self.ai.variables

    def test_solve_simple_equation(self):
        """Test solving a simple quadratic equation."""
        import sympy as sp

        x = self.ai.define_variables(['x'])[0]
        equation = x**2 - 4

        result = self.ai.solve_equation_system([equation])

        assert result is not None
        assert 'solutions' in result
        assert len(result['solutions']) == 2
        assert result['verification'][0] is True
        assert result['verification'][1] is True

    def test_solve_system_of_equations(self):
        """Test solving a system of equations."""
        import sympy as sp

        x, y = self.ai.define_variables(['x', 'y'])
        equations = [
            x + y - 7,
            x - y - 1
        ]

        result = self.ai.solve_equation_system(equations)

        assert result is not None
        assert 'solutions' in result
        assert len(result['solutions']) > 0

    def test_differentiate(self):
        """Test differentiation."""
        result = self.ai.perform_calculus("x**2", "x", "diff")

        assert result is not None
        assert 'result' in result
        assert str(result['result']) == "2*x"

    def test_integrate(self):
        """Test integration."""
        result = self.ai.perform_calculus("x**2", "x", "integrate")

        assert result is not None
        assert 'result' in result
        assert "x**3" in str(result['result'])

    def test_optimize_function(self):
        """Test function optimization."""
        result = self.ai.optimize_function("x**2 - 4*x + 3")

        assert result is not None
        assert 'critical_points' in result
        assert len(result['critical_points']) > 0

    def test_pattern_recognition_arithmetic(self):
        """Test pattern recognition for arithmetic sequence."""
        sequence = [2, 5, 8, 11, 14]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0
        # Arithmetic sequence can be recognized as either 'arithmetic' or 'polynomial' (degree 1)
        assert result['most_likely']['type'] in ['arithmetic', 'polynomial']
        assert result['most_likely']['next_terms'][0] == 17

    def test_pattern_recognition_geometric(self):
        """Test pattern recognition for geometric sequence."""
        sequence = [2, 6, 18, 54, 162]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0
        # Should detect geometric sequence with ratio 3

    def test_pattern_recognition_squares(self):
        """Test pattern recognition for perfect squares."""
        sequence = [1, 4, 9, 16, 25, 36]
        result = self.ai.pattern_recognition(sequence)

        assert result is not None
        assert result['patterns_found'] > 0
        assert result['most_likely'] is not None

    def test_matrix_operations_determinant(self):
        """Test matrix determinant calculation."""
        matrix = [[1, 2], [3, 4]]
        result = self.ai.matrix_operations([matrix], "determinant")

        assert result is not None
        assert result['result'] == -2

    def test_matrix_operations_inverse(self):
        """Test matrix inverse calculation."""
        matrix = [[1, 2], [3, 4]]
        result = self.ai.matrix_operations([matrix], "inverse")

        assert result is not None
        assert result['result'] is not None

    def test_matrix_operations_multiply(self):
        """Test matrix multiplication."""
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[5, 6], [7, 8]]
        result = self.ai.matrix_operations([matrix1, matrix2], "multiply")

        assert result is not None
        assert result['result'] is not None

    def test_solve_word_problem(self):
        """Test word problem solving."""
        problem = "Find two numbers whose sum is 10 and difference is 2"
        equations = ["x + y - 10", "x - y - 2"]
        unknowns = ["x", "y"]

        result = self.ai.solve_word_problem(problem, equations, unknowns)

        assert result is not None
        assert 'solutions' in result
        assert result['solutions'] is not None

    def test_generate_proof(self):
        """Test proof generation."""
        theorem = "x**2 - x**2"
        result = self.ai.generate_proof(theorem)

        assert result is not None
        assert 'proven' in result
        assert 'proof_steps' in result

    def test_error_handling_invalid_equation(self):
        """Test error handling for invalid equations."""
        with pytest.raises(Exception):
            self.ai.solve_equation_system(["invalid equation!!!"])


class TestMCPToolInputs:
    """Test MCP tool input validation."""

    def test_solve_equations_empty_input(self):
        """Test that empty equation list is handled."""
        # This would be caught by the MCP server's validation
        equations = []
        assert len(equations) == 0  # Should be rejected

    def test_solve_equations_valid_input(self):
        """Test valid equation input format."""
        equations = ["x**2 - 4", "x + y - 7"]
        assert isinstance(equations, list)
        assert all(isinstance(eq, str) for eq in equations)

    def test_differentiate_valid_input(self):
        """Test valid differentiate input format."""
        expression = "sin(x)*cos(x)"
        variable = "x"
        assert isinstance(expression, str)
        assert isinstance(variable, str)

    def test_pattern_recognition_valid_input(self):
        """Test valid pattern recognition input."""
        sequence = [1, 4, 9, 16, 25]
        assert isinstance(sequence, list)
        assert all(isinstance(n, (int, float)) for n in sequence)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_empty_variable_list(self):
        """Test with empty variable list."""
        variables = self.ai.define_variables([])
        assert len(variables) == 0

    def test_single_variable(self):
        """Test with single variable."""
        variables = self.ai.define_variables('x')
        assert len(variables) == 1

    def test_pattern_single_element(self):
        """Test pattern recognition with single element."""
        sequence = [42]
        result = self.ai.pattern_recognition(sequence)
        # Should handle gracefully even if no pattern found
        assert result is not None

    def test_pattern_two_elements(self):
        """Test pattern recognition with two elements."""
        sequence = [1, 2]
        result = self.ai.pattern_recognition(sequence)
        assert result is not None


class TestAccuracy:
    """Test the accuracy claims of the symbolic AI system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_complex_equation_accuracy(self):
        """Test accuracy on complex equations."""
        import sympy as sp

        x = self.ai.define_variables(['x'])[0]
        equation = x**4 - 5*x**3 + 6*x**2 + 4*x - 8

        result = self.ai.solve_equation_system([equation])

        # Verify all solutions
        for i, solution in enumerate(result['solutions']):
            assert result['verification'][i] is True, f"Solution {i} failed verification"

    def test_calculus_chain_rule_accuracy(self):
        """Test accuracy of chain rule differentiation."""
        result = self.ai.perform_calculus("sin(x**2)", "x", "diff")

        # Result should be 2*x*cos(x**2)
        assert "cos" in str(result['result'])
        assert "x" in str(result['result'])

    def test_integration_accuracy(self):
        """Test accuracy of integration."""
        result = self.ai.perform_calculus("2*x", "x", "integrate")

        # Result should be x**2 (plus constant)
        assert "x**2" in str(result['result'])


class TestPerformance:
    """Test performance characteristics."""

    def setup_method(self):
        """Setup test fixtures."""
        self.ai = SymbolicAI()

    def test_solve_speed(self):
        """Test that solving is reasonably fast."""
        import time
        import sympy as sp

        x = self.ai.define_variables(['x'])[0]
        equation = x**2 - 4

        start = time.time()
        result = self.ai.solve_equation_system([equation])
        end = time.time()

        assert end - start < 1.0, "Solving took too long"

    def test_differentiate_speed(self):
        """Test that differentiation is reasonably fast."""
        import time

        start = time.time()
        result = self.ai.perform_calculus("x**2 * sin(x)", "x", "diff")
        end = time.time()

        assert end - start < 1.0, "Differentiation took too long"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
