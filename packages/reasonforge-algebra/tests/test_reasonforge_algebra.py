"""
Comprehensive tests for reasonforge-algebra MCP server.

Tests all 18 tools:
- Equation Solving (4 tools): solve_equations, solve_algebraically, solve_linear_system, solve_nonlinear_system
- Matrix Operations (5 tools): create_matrix, matrix_determinant, matrix_inverse, matrix_eigenvalues, matrix_eigenvectors
- Optimization (6 tools): optimize_function, lagrange_multipliers, linear_programming, convex_optimization, calculus_of_variations, dynamic_programming
- Other Operations (3 tools): recognize_pattern, differentiate_expression, integrate_expression
"""

import asyncio
import json
import pytest

from reasonforge_algebra.server import server as algebra_server
from reasonforge_expressions.server import server as expressions_server


class TestEquationSolving:
    """Test equation solving tools (4 tools)."""

    @pytest.mark.asyncio
    async def test_solve_equations_single(self):
        """Test solving a single equation."""
        result = await algebra_server.call_tool_for_test(
            "solve_equations",
            {
                "equations": ["x**2 - 4"],
                "variables": ["x"]
            }
        )
        data = json.loads(result[0].text)

        assert "solutions" in data
        assert len(data["solutions"]) == 2  # x = 2 and x = -2
        assert "explanation" in data

    @pytest.mark.asyncio
    async def test_solve_equations_system(self):
        """Test solving a system of equations."""
        result = await algebra_server.call_tool_for_test(
            "solve_equations",
            {
                "equations": ["x + y - 3", "x - y - 1"],
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert "solutions" in data
        assert "verification" in data
        # Solution should be x=2, y=1

    @pytest.mark.asyncio
    async def test_solve_equations_auto_detect_variables(self):
        """Test auto-detection of variables."""
        result = await algebra_server.call_tool_for_test(
            "solve_equations",
            {
                "equations": ["a**2 - 9"]
            }
        )
        data = json.loads(result[0].text)

        assert "solutions" in data

    @pytest.mark.asyncio
    async def test_solve_algebraically_quadratic(self):
        """Test algebraic solving of quadratic equation."""
        result = await algebra_server.call_tool_for_test(
            "solve_algebraically",
            {
                "expression_key": "x**2 + 5*x + 6",
                "variable": "x",
                "domain": "complex"
            }
        )
        data = json.loads(result[0].text)

        assert data["variable"] == "x"
        assert data["domain"] == "complex"
        assert len(data["solutions"]) == 2
        # Solutions should be x=-2 and x=-3

    @pytest.mark.asyncio
    async def test_solve_algebraically_cubic(self):
        """Test algebraic solving of cubic equation."""
        result = await algebra_server.call_tool_for_test(
            "solve_algebraically",
            {
                "expression_key": "x**3 - 1",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert data["count"] == 3  # Three solutions (one real, two complex)

    @pytest.mark.asyncio
    async def test_solve_linear_system_2x2(self):
        """Test solving 2x2 linear system."""
        result = await algebra_server.call_tool_for_test(
            "solve_linear_system",
            {
                "equations": ["2*x + 3*y - 7", "x - y - 1"],
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["method"] == "linsolve"
        assert "solution" in data
        # Solution should be x=2, y=1

    @pytest.mark.asyncio
    async def test_solve_linear_system_3x3(self):
        """Test solving 3x3 linear system."""
        result = await algebra_server.call_tool_for_test(
            "solve_linear_system",
            {
                "equations": ["x + y + z - 6", "2*x - y + z - 3", "x + 2*y - z - 2"],
                "variables": ["x", "y", "z"]
            }
        )
        data = json.loads(result[0].text)

        assert "solution" in data

    @pytest.mark.asyncio
    async def test_solve_nonlinear_system(self):
        """Test solving nonlinear system."""
        result = await algebra_server.call_tool_for_test(
            "solve_nonlinear_system",
            {
                "equations": ["x**2 + y**2 - 25", "x - y - 1"],
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["method"] == "nonlinsolve"
        assert "solutions" in data
        # Circle x^2 + y^2 = 25 intersects line x - y = 1


class TestMatrixOperations:
    """Test matrix operation tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_create_matrix_2x2(self):
        """Test creating a 2x2 matrix."""
        result = await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[1, 2], [3, 4]],
                "key": "mat1"
            }
        )
        data = json.loads(result[0].text)

        assert data["key"] == "mat1"
        assert data["shape"] == [2, 2]
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_create_matrix_auto_key(self):
        """Test creating matrix with auto-generated key."""
        result = await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            }
        )
        data = json.loads(result[0].text)

        assert "matrix_" in data["key"]
        assert data["shape"] == [3, 3]

    @pytest.mark.asyncio
    async def test_matrix_determinant_2x2(self):
        """Test calculating determinant of 2x2 matrix."""
        # First create a matrix
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[1, 2], [3, 4]],
                "key": "det_test"
            }
        )

        # Calculate determinant
        result = await algebra_server.call_tool_for_test(
            "matrix_determinant",
            {"matrix_key": "det_test"}
        )
        data = json.loads(result[0].text)

        assert data["determinant"] == "-2"  # det = 1*4 - 2*3 = -2

    @pytest.mark.asyncio
    async def test_matrix_determinant_3x3(self):
        """Test calculating determinant of 3x3 matrix."""
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "key": "identity"
            }
        )

        result = await algebra_server.call_tool_for_test(
            "matrix_determinant",
            {"matrix_key": "identity"}
        )
        data = json.loads(result[0].text)

        assert data["determinant"] == "1"  # Identity matrix has det = 1

    @pytest.mark.asyncio
    async def test_matrix_determinant_not_found(self):
        """Test error handling for non-existent matrix."""
        result = await algebra_server.call_tool_for_test(
            "matrix_determinant",
            {"matrix_key": "nonexistent"}
        )
        data = json.loads(result[0].text)

        assert "error" in data

    @pytest.mark.asyncio
    async def test_matrix_inverse_2x2(self):
        """Test calculating matrix inverse."""
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[1, 2], [3, 4]],
                "key": "inv_test"
            }
        )

        result = await algebra_server.call_tool_for_test(
            "matrix_inverse",
            {"matrix_key": "inv_test"}
        )
        data = json.loads(result[0].text)

        assert "inverse" in data
        assert "stored_key" in data
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_matrix_inverse_identity(self):
        """Test inverse of identity matrix."""
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[1, 0], [0, 1]],
                "key": "identity2"
            }
        )

        result = await algebra_server.call_tool_for_test(
            "matrix_inverse",
            {"matrix_key": "identity2"}
        )
        data = json.loads(result[0].text)

        # Inverse of identity is identity
        assert "inverse" in data

    @pytest.mark.asyncio
    async def test_matrix_inverse_singular(self):
        """Test error handling for singular matrix."""
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[1, 2], [2, 4]],  # Singular matrix (det = 0)
                "key": "singular"
            }
        )

        result = await algebra_server.call_tool_for_test(
            "matrix_inverse",
            {"matrix_key": "singular"}
        )
        data = json.loads(result[0].text)

        assert "error" in data

    @pytest.mark.asyncio
    async def test_matrix_eigenvalues_2x2(self):
        """Test finding eigenvalues of 2x2 matrix."""
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[2, 1], [1, 2]],
                "key": "eigen_test"
            }
        )

        result = await algebra_server.call_tool_for_test(
            "matrix_eigenvalues",
            {"matrix_key": "eigen_test"}
        )
        data = json.loads(result[0].text)

        assert "eigenvalues" in data
        # Eigenvalues should be 3 and 1

    @pytest.mark.asyncio
    async def test_matrix_eigenvalues_diagonal(self):
        """Test eigenvalues of diagonal matrix."""
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[5, 0], [0, 3]],
                "key": "diagonal"
            }
        )

        result = await algebra_server.call_tool_for_test(
            "matrix_eigenvalues",
            {"matrix_key": "diagonal"}
        )
        data = json.loads(result[0].text)

        # Eigenvalues of diagonal matrix are its diagonal elements
        assert "eigenvalues" in data

    @pytest.mark.asyncio
    async def test_matrix_eigenvectors_2x2(self):
        """Test finding eigenvectors."""
        await algebra_server.call_tool_for_test(
            "create_matrix",
            {
                "elements": [[2, 1], [1, 2]],
                "key": "eigvec_test"
            }
        )

        result = await algebra_server.call_tool_for_test(
            "matrix_eigenvectors",
            {"matrix_key": "eigvec_test"}
        )
        data = json.loads(result[0].text)

        assert "eigenvectors" in data
        assert len(data["eigenvectors"]) > 0
        for eigvec_data in data["eigenvectors"]:
            assert "eigenvalue" in eigvec_data
            assert "multiplicity" in eigvec_data
            assert "eigenvector" in eigvec_data


class TestOptimization:
    """Test optimization tools (6 tools)."""

    @pytest.mark.asyncio
    async def test_optimize_function_simple(self):
        """Test optimizing a simple quadratic function."""
        result = await algebra_server.call_tool_for_test(
            "optimize_function",
            {
                "objective": "x**2 + 2*x + 1",
                "variables": ["x"]
            }
        )
        data = json.loads(result[0].text)

        assert "critical_points" in data
        assert "evaluations" in data
        # Minimum at x = -1

    @pytest.mark.asyncio
    async def test_optimize_function_multivariable(self):
        """Test optimizing multivariable function."""
        result = await algebra_server.call_tool_for_test(
            "optimize_function",
            {
                "objective": "x**2 + y**2",
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert "critical_points" in data
        # Minimum at (0, 0)

    @pytest.mark.asyncio
    async def test_optimize_function_auto_detect(self):
        """Test optimization with auto-detected variables."""
        result = await algebra_server.call_tool_for_test(
            "optimize_function",
            {
                "objective": "a**2 + 4*a + 4"
            }
        )
        data = json.loads(result[0].text)

        assert "variables" in data

    @pytest.mark.asyncio
    async def test_lagrange_multipliers_basic(self):
        """Test Lagrange multipliers optimization."""
        result = await algebra_server.call_tool_for_test(
            "lagrange_multipliers",
            {
                "objective": "x**2 + y**2",
                "constraints": ["x + y - 1"],
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert "lagrangian" in data
        assert "critical_points" in data
        assert "lambda_0" in data["lagrangian"]

    @pytest.mark.asyncio
    async def test_lagrange_multipliers_multiple_constraints(self):
        """Test with multiple constraints."""
        result = await algebra_server.call_tool_for_test(
            "lagrange_multipliers",
            {
                "objective": "x + y + z",
                "constraints": ["x**2 + y**2 - 1", "z - 1"],
                "variables": ["x", "y", "z"]
            }
        )
        data = json.loads(result[0].text)

        assert "lagrangian" in data
        assert "lambda_0" in data["lagrangian"]
        assert "lambda_1" in data["lagrangian"]

    @pytest.mark.asyncio
    async def test_linear_programming(self):
        """Test linear programming setup."""
        result = await algebra_server.call_tool_for_test(
            "linear_programming",
            {
                "objective": "2*x + 3*y",
                "constraints": ["x + y - 10", "x - 5", "y - 3"],
                "variables": ["x", "y"],
                "minimize": True
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "linear_programming"
        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_convex_optimization(self):
        """Test convex optimization setup."""
        result = await algebra_server.call_tool_for_test(
            "convex_optimization",
            {
                "objective": "x**2 + y**2",
                "constraints": ["x + y - 1"],
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "convex_optimization"
        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_calculus_of_variations(self):
        """Test calculus of variations setup."""
        result = await algebra_server.call_tool_for_test(
            "calculus_of_variations",
            {
                "functional": "sqrt(1 + (y')**2)",
                "function_name": "y",
                "independent_var": "x"
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "calculus_of_variations"
        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_dynamic_programming(self):
        """Test dynamic programming setup."""
        result = await algebra_server.call_tool_for_test(
            "dynamic_programming",
            {
                "value_function": "V(s)",
                "state_variables": ["s"],
                "decision_variables": ["a"]
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "dynamic_programming"
        assert data["status"] == "symbolic_setup"


class TestOtherOperations:
    """Test other operation tools (3 tools)."""

    @pytest.mark.asyncio
    async def test_recognize_pattern_arithmetic(self):
        """Test recognizing arithmetic sequence."""
        result = await algebra_server.call_tool_for_test(
            "recognize_pattern",
            {
                "sequence": [2, 4, 6, 8, 10]
            }
        )
        data = json.loads(result[0].text)

        assert "patterns_found" in data
        assert data["patterns_found"] > 0
        assert "patterns" in data

    @pytest.mark.asyncio
    async def test_recognize_pattern_geometric(self):
        """Test recognizing geometric sequence."""
        result = await algebra_server.call_tool_for_test(
            "recognize_pattern",
            {
                "sequence": [2, 4, 8, 16, 32]
            }
        )
        data = json.loads(result[0].text)

        assert data["patterns_found"] > 0

    @pytest.mark.asyncio
    async def test_recognize_pattern_quadratic(self):
        """Test recognizing quadratic pattern."""
        result = await algebra_server.call_tool_for_test(
            "recognize_pattern",
            {
                "sequence": [1, 4, 9, 16, 25]  # n^2
            }
        )
        data = json.loads(result[0].text)

        assert data["patterns_found"] > 0

    @pytest.mark.asyncio
    async def test_differentiate_expression_stored(self):
        """Test differentiating an expression string (treated as not stored)."""
        # Use expression string directly since servers have separate state
        result = await algebra_server.call_tool_for_test(
            "differentiate_expression",
            {
                "expression_key": "x**3 + 2*x**2 + x",
                "variable": "x",
                "order": 1
            }
        )
        data = json.loads(result[0].text)

        assert data["order"] == 1
        assert "3*x**2" in data["derivative"]
        assert "stored_key" in data

    @pytest.mark.asyncio
    async def test_differentiate_expression_direct(self):
        """Test differentiating expression directly (not stored)."""
        result = await algebra_server.call_tool_for_test(
            "differentiate_expression",
            {
                "expression_key": "sin(x)",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "cos" in data["derivative"]

    @pytest.mark.asyncio
    async def test_differentiate_expression_higher_order(self):
        """Test higher-order differentiation."""
        result = await algebra_server.call_tool_for_test(
            "differentiate_expression",
            {
                "expression_key": "x**5",
                "variable": "x",
                "order": 3
            }
        )
        data = json.loads(result[0].text)

        assert data["order"] == 3
        # Third derivative of x^5 is 60*x^2

    @pytest.mark.asyncio
    async def test_integrate_expression_stored(self):
        """Test integrating an expression string (treated as not stored)."""
        # Use expression string directly since servers have separate state
        result = await algebra_server.call_tool_for_test(
            "integrate_expression",
            {
                "expression_key": "x**2",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "x**3/3" in data["integral"]
        assert "stored_key" in data

    @pytest.mark.asyncio
    async def test_integrate_expression_direct(self):
        """Test integrating expression directly."""
        result = await algebra_server.call_tool_for_test(
            "integrate_expression",
            {
                "expression_key": "cos(x)",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "sin" in data["integral"]

    @pytest.mark.asyncio
    async def test_integrate_expression_definite(self):
        """Test definite integration."""
        result = await algebra_server.call_tool_for_test(
            "integrate_expression",
            {
                "expression_key": "x",
                "variable": "x",
                "bounds": [0, 1]
            }
        )
        data = json.loads(result[0].text)

        assert data["bounds"] == [0, 1]
        # Integral of x from 0 to 1 is 1/2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
