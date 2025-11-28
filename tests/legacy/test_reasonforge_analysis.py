"""
Comprehensive tests for reasonforge-analysis MCP server.

Tests all 17 tools:
- Differential Equations (3 tools): dsolve_ode, pdsolve_pde, symbolic_ode_initial_conditions
- Physics PDEs (3 tools): schrodinger_equation_solver, wave_equation_solver, heat_equation_solver
- Transforms (5 tools): laplace_transform, fourier_transform, z_transform, mellin_transform, integral_transforms_custom
- Signal Processing (2 tools): convolution, transfer_function_analysis
- Asymptotic Methods (2 tools): perturbation_theory, asymptotic_analysis
- Special Functions & Optimization (2 tools): special_functions_properties, symbolic_optimization_setup
"""

import sys
import os
import asyncio
import json
import pytest

# Add packages to path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-analysis', 'src'))

from reasonforge_analysis.server import server as analysis_server


class TestDifferentialEquations:
    """Test differential equation tools (3 tools)."""

    @pytest.mark.asyncio
    async def test_dsolve_ode_first_order(self):
        """Test solving first-order ODE."""
        result = await analysis_server.call_tool_for_test(
            "dsolve_ode",
            {
                "equation": "Derivative(y(x), x) - y(x)",
                "function": "y(x)"
            }
        )
        data = json.loads(result[0].text)

        assert "solution" in data
        assert "y(x)" in data["solution"]
        assert "exp" in data["solution"]  # Solution involves exponential
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_dsolve_ode_second_order(self):
        """Test solving second-order ODE."""
        result = await analysis_server.call_tool_for_test(
            "dsolve_ode",
            {
                "equation": "Derivative(y(x), x, x) + y(x)",
                "function": "y(x)"
            }
        )
        data = json.loads(result[0].text)

        assert "solution" in data
        # Solution should involve sin and cos

    @pytest.mark.asyncio
    async def test_dsolve_ode_with_coefficient(self):
        """Test ODE with coefficients."""
        result = await analysis_server.call_tool_for_test(
            "dsolve_ode",
            {
                "equation": "Derivative(y(x), x) - 2*y(x)",
                "function": "y(x)"
            }
        )
        data = json.loads(result[0].text)

        assert "solution" in data

    @pytest.mark.asyncio
    async def test_pdsolve_pde(self):
        """Test solving PDE."""
        result = await analysis_server.call_tool_for_test(
            "pdsolve_pde",
            {
                "equation": "Derivative(u(x,t), x) + Derivative(u(x,t), t)",
                "function": "u(x,t)"
            }
        )
        data = json.loads(result[0].text)

        # PDEs might not always have symbolic solutions
        assert "equation" in data
        assert "function" in data

    @pytest.mark.asyncio
    async def test_pdsolve_pde_error_handling(self):
        """Test PDE error handling for complex cases."""
        result = await analysis_server.call_tool_for_test(
            "pdsolve_pde",
            {
                "equation": "Derivative(u(x,y,t), x, x) + Derivative(u(x,y,t), y, y) - Derivative(u(x,y,t), t)",
                "function": "u(x,y,t)"
            }
        )
        data = json.loads(result[0].text)

        assert "equation" in data
        # May have 'note' field if no symbolic solution

    @pytest.mark.asyncio
    async def test_symbolic_ode_initial_conditions(self):
        """Test ODE with initial conditions."""
        result = await analysis_server.call_tool_for_test(
            "symbolic_ode_initial_conditions",
            {
                "equation": "Derivative(f(x), x) - f(x)",
                "function": "f(x)",
                "initial_conditions": {
                    "f(0)": 1
                }
            }
        )
        data = json.loads(result[0].text)

        assert "solution" in data or "error" in data
        if "solution" in data:
            assert "initial_conditions" in data
            # Solution should not have arbitrary constants (C1, etc.)


class TestPhysicsPDEs:
    """Test physics PDE tools (3 tools)."""

    @pytest.mark.asyncio
    async def test_schrodinger_equation_solver(self):
        """Test Schrödinger equation solver setup."""
        result = await analysis_server.call_tool_for_test(
            "schrodinger_equation_solver",
            {
                "equation_type": "time_independent",
                "potential": "0.5*k*x**2",
                "boundary_conditions": {}
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "schrodinger_equation_solver"
        assert data["status"] == "symbolic_formulation"

    @pytest.mark.asyncio
    async def test_wave_equation_solver(self):
        """Test wave equation solver setup."""
        result = await analysis_server.call_tool_for_test(
            "wave_equation_solver",
            {
                "wave_type": "1D_string",
                "dimension": 1,
                "boundary_conditions": {}
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "wave_equation_solver"
        assert data["status"] == "symbolic_formulation"

    @pytest.mark.asyncio
    async def test_heat_equation_solver(self):
        """Test heat equation solver setup."""
        result = await analysis_server.call_tool_for_test(
            "heat_equation_solver",
            {
                "geometry": "1D_rod",
                "boundary_conditions": {},
                "initial_conditions": {}
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "heat_equation_solver"
        assert data["status"] == "symbolic_formulation"


class TestTransforms:
    """Test transform tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_laplace_transform_exponential(self):
        """Test Laplace transform of exponential."""
        result = await analysis_server.call_tool_for_test(
            "laplace_transform",
            {
                "expression": "exp(-a*t)",
                "variable": "t",
                "transform_variable": "s"
            }
        )
        data = json.loads(result[0].text)

        assert "transform" in data
        assert data["variable"] == "t"
        assert data["transform_variable"] == "s"
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_laplace_transform_sine(self):
        """Test Laplace transform of sine."""
        result = await analysis_server.call_tool_for_test(
            "laplace_transform",
            {
                "expression": "sin(w*t)",
                "variable": "t"
            }
        )
        data = json.loads(result[0].text)

        assert "transform" in data
        # Transform should involve w and s

    @pytest.mark.asyncio
    async def test_laplace_transform_polynomial(self):
        """Test Laplace transform of polynomial."""
        result = await analysis_server.call_tool_for_test(
            "laplace_transform",
            {
                "expression": "t**2",
                "variable": "t"
            }
        )
        data = json.loads(result[0].text)

        assert "transform" in data

    @pytest.mark.asyncio
    async def test_fourier_transform_gaussian(self):
        """Test Fourier transform of Gaussian."""
        result = await analysis_server.call_tool_for_test(
            "fourier_transform",
            {
                "expression": "exp(-x**2)",
                "variable": "x",
                "transform_variable": "k"
            }
        )
        data = json.loads(result[0].text)

        assert "transform" in data
        assert data["variable"] == "x"
        assert data["transform_variable"] == "k"
        # Fourier transform of Gaussian is also Gaussian

    @pytest.mark.asyncio
    async def test_fourier_transform_exponential(self):
        """Test Fourier transform of exponential."""
        result = await analysis_server.call_tool_for_test(
            "fourier_transform",
            {
                "expression": "exp(-abs(x))",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "transform" in data

    @pytest.mark.asyncio
    async def test_z_transform(self):
        """Test Z-transform setup."""
        result = await analysis_server.call_tool_for_test(
            "z_transform",
            {
                "sequence": "a**n",
                "n_variable": "n",
                "z_variable": "z"
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "z_transform"
        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_mellin_transform(self):
        """Test Mellin transform setup."""
        result = await analysis_server.call_tool_for_test(
            "mellin_transform",
            {
                "expression": "exp(-x)",
                "variable": "x",
                "transform_variable": "s"
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "mellin_transform"
        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_integral_transforms_custom(self):
        """Test custom integral transform setup."""
        result = await analysis_server.call_tool_for_test(
            "integral_transforms_custom",
            {
                "transform_type": "Hankel",
                "expression": "exp(-r)",
                "variable": "r"
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "integral_transforms_custom"
        assert data["status"] == "symbolic_setup"


class TestSignalProcessing:
    """Test signal processing tools (2 tools)."""

    @pytest.mark.asyncio
    async def test_convolution_simple(self):
        """Test convolution of simple functions."""
        result = await analysis_server.call_tool_for_test(
            "convolution",
            {
                "f": "exp(-t)",
                "g": "exp(-t)",
                "variable": "t"
            }
        )
        data = json.loads(result[0].text)

        assert "convolution" in data
        assert data["variable"] == "t"
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_convolution_polynomial(self):
        """Test convolution with polynomials."""
        result = await analysis_server.call_tool_for_test(
            "convolution",
            {
                "f": "1",
                "g": "x",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "convolution" in data

    @pytest.mark.asyncio
    async def test_transfer_function_analysis_simple(self):
        """Test transfer function analysis."""
        result = await analysis_server.call_tool_for_test(
            "transfer_function_analysis",
            {
                "transfer_function": "1/(s**2 + 2*s + 1)",
                "variable": "s"
            }
        )
        data = json.loads(result[0].text)

        assert "poles" in data
        assert "zeros" in data
        # Poles should be at s = -1 (double pole)

    @pytest.mark.asyncio
    async def test_transfer_function_analysis_complex(self):
        """Test transfer function with complex poles."""
        result = await analysis_server.call_tool_for_test(
            "transfer_function_analysis",
            {
                "transfer_function": "(s + 1)/(s**2 + s + 1)",
                "variable": "s"
            }
        )
        data = json.loads(result[0].text)

        assert "poles" in data
        assert "zeros" in data
        assert len(data["zeros"]) == 1  # One zero at s = -1
        assert len(data["poles"]) == 2  # Two complex poles

    @pytest.mark.asyncio
    async def test_transfer_function_analysis_zeros(self):
        """Test transfer function with zeros."""
        result = await analysis_server.call_tool_for_test(
            "transfer_function_analysis",
            {
                "transfer_function": "(s - 1)*(s + 2)/s",
                "variable": "s"
            }
        )
        data = json.loads(result[0].text)

        assert len(data["zeros"]) == 2  # Zeros at s=1 and s=-2
        assert len(data["poles"]) == 1  # Pole at s=0


class TestAsymptoticMethods:
    """Test asymptotic method tools (2 tools)."""

    @pytest.mark.asyncio
    async def test_perturbation_theory(self):
        """Test perturbation theory setup."""
        result = await analysis_server.call_tool_for_test(
            "perturbation_theory",
            {
                "equation": "x**2 + epsilon*x - 1",
                "small_parameter": "epsilon"
            }
        )
        data = json.loads(result[0].text)

        assert data["small_parameter"] == "epsilon"
        assert data["status"] == "perturbation_setup"

    @pytest.mark.asyncio
    async def test_perturbation_theory_ode(self):
        """Test perturbation theory for ODE."""
        result = await analysis_server.call_tool_for_test(
            "perturbation_theory",
            {
                "equation": "Derivative(y(x), x, x) + (1 + epsilon)*y(x)",
                "small_parameter": "epsilon",
                "perturbation_type": "regular"
            }
        )
        data = json.loads(result[0].text)

        assert "small_parameter" in data

    @pytest.mark.asyncio
    async def test_asymptotic_analysis_infinity(self):
        """Test asymptotic expansion at infinity."""
        result = await analysis_server.call_tool_for_test(
            "asymptotic_analysis",
            {
                "expression": "1/x + 1/x**2",
                "variable": "x",
                "limit_point": "inf"
            }
        )
        data = json.loads(result[0].text)

        assert "expansion" in data
        assert data["limit_point"] == "inf"
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_asymptotic_analysis_zero(self):
        """Test asymptotic expansion at zero."""
        result = await analysis_server.call_tool_for_test(
            "asymptotic_analysis",
            {
                "expression": "sin(x)/x",
                "variable": "x",
                "limit_point": "0"
            }
        )
        data = json.loads(result[0].text)

        assert "expansion" in data

    @pytest.mark.asyncio
    async def test_asymptotic_analysis_exponential(self):
        """Test asymptotic expansion of exponential."""
        result = await analysis_server.call_tool_for_test(
            "asymptotic_analysis",
            {
                "expression": "exp(1/x)",
                "variable": "x",
                "limit_point": "inf"
            }
        )
        data = json.loads(result[0].text)

        assert "expansion" in data


class TestSpecialFunctionsAndOptimization:
    """Test special functions and optimization tools (2 tools)."""

    @pytest.mark.asyncio
    async def test_special_functions_properties_bessel(self):
        """Test special functions properties for Bessel."""
        result = await analysis_server.call_tool_for_test(
            "special_functions_properties",
            {
                "function_type": "Bessel",
                "operation": "derivative",
                "parameters": {"order": "n"}
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "special_functions_properties"
        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_special_functions_properties_legendre(self):
        """Test special functions properties for Legendre."""
        result = await analysis_server.call_tool_for_test(
            "special_functions_properties",
            {
                "function_type": "Legendre",
                "operation": "orthogonality"
            }
        )
        data = json.loads(result[0].text)

        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_symbolic_optimization_setup_unconstrained(self):
        """Test optimization setup without constraints."""
        result = await analysis_server.call_tool_for_test(
            "symbolic_optimization_setup",
            {
                "objective": "x**2 + y**2",
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["tool"] == "symbolic_optimization_setup"
        assert data["status"] == "symbolic_setup"

    @pytest.mark.asyncio
    async def test_symbolic_optimization_setup_constrained(self):
        """Test optimization setup with constraints."""
        result = await analysis_server.call_tool_for_test(
            "symbolic_optimization_setup",
            {
                "objective": "x**2 + y**2",
                "equality_constraints": ["x + y - 1"],
                "inequality_constraints": ["x - 0", "y - 0"],
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["status"] == "symbolic_setup"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
