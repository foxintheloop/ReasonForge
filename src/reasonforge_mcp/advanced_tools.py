"""
Advanced SymPy Tools for ReasonForge MCP Server

This module contains 52 advanced tools for:
- Variable & Expression Management
- Advanced Solvers (ODE/PDE, linear/nonlinear systems)
- Tensor Calculus & General Relativity
- Vector Calculus
- Unit Operations
- Enhanced Matrix Operations
- Probability & Statistics (8 tools)
- Transform Theory (6 tools)
- Optimization Extensions (5 tools)
"""

import json
import sympy as sp
from sympy import (
    symbols, Symbol, Function, Eq, dsolve, pdsolve,
    solveset, linsolve, nonlinsolve, solve,
    simplify, trigsimp, radsimp, powsimp, logcombine,
    factor, expand, latex, pretty, Matrix, sympify
)
from sympy.vector import CoordSys3D, curl, divergence, gradient
from sympy.physics.units import convert_to
from sympy.stats import (
    Normal, Binomial, Poisson, Exponential, Uniform, Geometric,
    density, cdf, E, variance, moment, skewness, kurtosis, P,
    Die, FiniteRV
)
from sympy import sqrt, pi, exp as sp_exp, Sum, Product, Integral
from sympy.integrals.transforms import (
    laplace_transform, inverse_laplace_transform,
    fourier_transform, inverse_fourier_transform,
    mellin_transform, inverse_mellin_transform
)
from sympy import oo, DiracDelta, Heaviside, diff, hessian

# Try to import quantity_simplify from different locations based on SymPy version
try:
    from sympy.physics.units.util import quantity_simplify
except ImportError:
    try:
        from sympy.physics.units import quantity_simplify
    except ImportError:
        # Fallback: create a simple wrapper using convert_to
        def quantity_simplify(expr, unit_system=None):
            """Fallback for older SymPy versions without quantity_simplify."""
            return simplify(expr)

from mcp.types import Tool, TextContent

# Try to import EinsteinPy for GR calculations
try:
    from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols
    from einsteinpy.symbolic import RicciTensor, RicciScalar
    from einsteinpy.symbolic import EinsteinTensor, WeylTensor
    from einsteinpy.symbolic.predefined import (
        Schwarzschild, Kerr, KerrNewman, Minkowski,
        DeSitter, AntiDeSitter, Godel, BesselGravitationalWave
    )
    EINSTEINPY_AVAILABLE = True
except ImportError:
    EINSTEINPY_AVAILABLE = False


def get_advanced_tool_definitions() -> list[Tool]:
    """
    Return tool definitions for all 33 advanced tools.

    Returns:
        list[Tool]: List of MCP Tool objects
    """
    tools = []

    # ========================================================================
    # CATEGORY A: Variable & Expression Management (4 tools)
    # ========================================================================

    tools.append(Tool(
        name="intro",
        description="Introduce a variable with specified assumptions (real, positive, etc.). Stores variable for later use.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Variable name (e.g., 'x', 'theta', 'alpha')"
                },
                "positive_assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Assumptions to set as True: real, positive, negative, integer, rational, irrational, finite, infinite, complex, imaginary, even, odd, prime, composite, zero, nonzero, commutative"
                },
                "negative_assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Assumptions to set as False"
                }
            },
            "required": ["name"]
        }
    ))

    tools.append(Tool(
        name="intro_many",
        description="Introduce multiple variables with the same assumptions simultaneously.",
        inputSchema={
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of variable names (e.g., ['x', 'y', 'z'])"
                },
                "positive_assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Assumptions to apply to all variables"
                }
            },
            "required": ["names"]
        }
    ))

    tools.append(Tool(
        name="introduce_expression",
        description="Parse an expression string using available variables/functions and store it with a key for later reference.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression (e.g., 'x**2 + 2*x + 1', 'sin(x)*cos(y)')"
                },
                "key": {
                    "type": "string",
                    "description": "Optional custom key (defaults to auto-generated expr_0, expr_1, etc.)"
                }
            },
            "required": ["expression"]
        }
    ))

    tools.append(Tool(
        name="print_latex_expression",
        description="Print a stored expression in LaTeX format along with variable assumptions.",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Expression key (e.g., 'expr_0')"
                }
            },
            "required": ["key"]
        }
    ))

    # ========================================================================
    # CATEGORY B: Advanced Solvers (6 tools)
    # ========================================================================

    tools.append(Tool(
        name="solve_algebraically",
        description="Solve an equation algebraically over a specified domain (complex, real, integers, naturals).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to stored expression"
                },
                "variable": {
                    "type": "string",
                    "description": "Variable to solve for"
                },
                "domain": {
                    "type": "string",
                    "enum": ["complex", "real", "integers", "naturals"],
                    "description": "Solution domain (default: complex)"
                }
            },
            "required": ["expression_key", "variable"]
        }
    ))

    tools.append(Tool(
        name="solve_linear_system",
        description="Solve a system of linear equations using linsolve.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to stored equations"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variables to solve for"
                }
            },
            "required": ["expression_keys", "variables"]
        }
    ))

    tools.append(Tool(
        name="solve_nonlinear_system",
        description="Solve a system of nonlinear equations using nonlinsolve.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to stored equations"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variables to solve for"
                }
            },
            "required": ["expression_keys", "variables"]
        }
    ))

    tools.append(Tool(
        name="introduce_function",
        description="Introduce a function variable for use in differential equations (ODEs/PDEs).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Function name (e.g., 'f', 'u', 'psi')"
                }
            },
            "required": ["name"]
        }
    ))

    tools.append(Tool(
        name="dsolve_ode",
        description="Solve an ordinary differential equation (ODE) with optional initial conditions.",
        inputSchema={
            "type": "object",
            "properties": {
                "equation_key": {
                    "type": "string",
                    "description": "Key to ODE equation"
                },
                "function_name": {
                    "type": "string",
                    "description": "Function to solve for"
                },
                "hint": {
                    "type": "string",
                    "description": "Optional solution method hint (separable, 1st_linear, etc.)"
                },
                "ics": {
                    "type": "object",
                    "description": "Initial conditions (e.g., {'f(0)': 1})"
                }
            },
            "required": ["equation_key", "function_name"]
        }
    ))

    tools.append(Tool(
        name="pdsolve_pde",
        description="Solve a partial differential equation (PDE).",
        inputSchema={
            "type": "object",
            "properties": {
                "equation_key": {
                    "type": "string",
                    "description": "Key to PDE equation"
                },
                "function_name": {
                    "type": "string",
                    "description": "Function to solve for"
                },
                "hint": {
                    "type": "string",
                    "description": "Optional solution method hint"
                }
            },
            "required": ["equation_key", "function_name"]
        }
    ))

    # ========================================================================
    # CATEGORY C: Tensor & General Relativity (5 tools - need EinsteinPy)
    # ========================================================================

    tools.append(Tool(
        name="create_predefined_metric",
        description="Create a predefined spacetime metric (Schwarzschild, Kerr, Minkowski, DeSitter, AntiDeSitter, Godel, etc.). Requires einsteinpy.",
        inputSchema={
            "type": "object",
            "properties": {
                "metric_type": {
                    "type": "string",
                    "enum": ["Schwarzschild", "Kerr", "KerrNewman", "Minkowski", "DeSitter", "AntiDeSitter", "Godel", "BesselGravitationalWave"],
                    "description": "Type of metric to create"
                },
                "parameters": {
                    "type": "object",
                    "description": "Metric parameters (e.g., {'M': 'M', 'a': 'a'} for Kerr)"
                },
                "key": {
                    "type": "string",
                    "description": "Optional storage key"
                }
            },
            "required": ["metric_type"]
        }
    ))

    tools.append(Tool(
        name="search_predefined_metrics",
        description="Search and list available predefined spacetime metrics with descriptions.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search term to filter metrics"
                }
            }
        }
    ))

    tools.append(Tool(
        name="calculate_tensor",
        description="Calculate tensors from a metric (Ricci, Einstein, Weyl, Christoffel, RicciScalar). Requires einsteinpy.",
        inputSchema={
            "type": "object",
            "properties": {
                "metric_key": {
                    "type": "string",
                    "description": "Key to stored metric"
                },
                "tensor_type": {
                    "type": "string",
                    "enum": ["Christoffel", "Ricci", "RicciScalar", "Einstein", "Weyl"],
                    "description": "Type of tensor to calculate"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["metric_key", "tensor_type"]
        }
    ))

    tools.append(Tool(
        name="create_custom_metric",
        description="Create a custom metric tensor from provided components. Requires einsteinpy.",
        inputSchema={
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": "4x4 matrix of component expressions"
                },
                "coordinates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Coordinate names (e.g., ['t', 'r', 'theta', 'phi'])"
                },
                "key": {
                    "type": "string",
                    "description": "Optional storage key"
                }
            },
            "required": ["components", "coordinates"]
        }
    ))

    tools.append(Tool(
        name="print_latex_tensor",
        description="Print a stored tensor or metric in LaTeX format.",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Tensor or metric key"
                }
            },
            "required": ["key"]
        }
    ))

    # ========================================================================
    # CATEGORY D: Expression Operations (4 tools)
    # ========================================================================

    tools.append(Tool(
        name="simplify_expression",
        description="Simplify an expression using various methods (default, trigsimp, radsimp, powsimp, logcombine).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to expression to simplify"
                },
                "method": {
                    "type": "string",
                    "enum": ["default", "trigsimp", "radsimp", "powsimp", "logcombine"],
                    "description": "Simplification method (default: default)"
                }
            },
            "required": ["expression_key"]
        }
    ))

    tools.append(Tool(
        name="substitute_expression",
        description="Substitute variables in an expression with values or other expressions.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to expression"
                },
                "substitutions": {
                    "type": "object",
                    "description": "Variable: value/expression_key pairs"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["expression_key", "substitutions"]
        }
    ))

    tools.append(Tool(
        name="integrate_expression",
        description="Integrate an expression (definite or indefinite).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to expression to integrate"
                },
                "variable": {
                    "type": "string",
                    "description": "Integration variable"
                },
                "lower_bound": {
                    "type": "string",
                    "description": "Lower bound for definite integral (optional)"
                },
                "upper_bound": {
                    "type": "string",
                    "description": "Upper bound for definite integral (optional)"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["expression_key", "variable"]
        }
    ))

    tools.append(Tool(
        name="differentiate_expression",
        description="Differentiate an expression with respect to a variable (arbitrary order).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to expression to differentiate"
                },
                "variable": {
                    "type": "string",
                    "description": "Differentiation variable"
                },
                "order": {
                    "type": "integer",
                    "description": "Derivative order (default: 1)"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["expression_key", "variable"]
        }
    ))

    # ========================================================================
    # CATEGORY E: Vector Calculus (5 tools)
    # ========================================================================

    tools.append(Tool(
        name="create_coordinate_system",
        description="Create a 3D coordinate system (cartesian, cylindrical, or spherical).",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Coordinate system name"
                },
                "coord_type": {
                    "type": "string",
                    "enum": ["cartesian", "cylindrical", "spherical"],
                    "description": "Coordinate system type (default: cartesian)"
                }
            },
            "required": ["name"]
        }
    ))

    tools.append(Tool(
        name="create_vector_field",
        description="Create a vector field in a specified coordinate system.",
        inputSchema={
            "type": "object",
            "properties": {
                "coord_system_name": {
                    "type": "string",
                    "description": "Name of coordinate system"
                },
                "components": {
                    "type": "object",
                    "description": "Component expressions ({'i': 'x*y', 'j': 'y*z', 'k': 'z*x'})"
                },
                "key": {
                    "type": "string",
                    "description": "Optional storage key"
                }
            },
            "required": ["coord_system_name", "components"]
        }
    ))

    tools.append(Tool(
        name="calculate_curl",
        description="Calculate the curl of a vector field.",
        inputSchema={
            "type": "object",
            "properties": {
                "vector_field_key": {
                    "type": "string",
                    "description": "Key to stored vector field"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["vector_field_key"]
        }
    ))

    tools.append(Tool(
        name="calculate_divergence",
        description="Calculate the divergence of a vector field.",
        inputSchema={
            "type": "object",
            "properties": {
                "vector_field_key": {
                    "type": "string",
                    "description": "Key to stored vector field"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["vector_field_key"]
        }
    ))

    tools.append(Tool(
        name="calculate_gradient",
        description="Calculate the gradient of a scalar field.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to scalar expression"
                },
                "coord_system_name": {
                    "type": "string",
                    "description": "Coordinate system name"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["expression_key", "coord_system_name"]
        }
    ))

    # ========================================================================
    # CATEGORY F: Unit Operations (2 tools)
    # ========================================================================

    tools.append(Tool(
        name="convert_to_units",
        description="Convert an expression to target units in a specified unit system.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to expression with units"
                },
                "target_unit": {
                    "type": "string",
                    "description": "Target unit name (e.g., 'kilogram', 'joule')"
                },
                "unit_system": {
                    "type": "string",
                    "enum": ["SI", "MKS", "MKSA", "natural", "CGS"],
                    "description": "Unit system (default: SI)"
                }
            },
            "required": ["expression_key", "target_unit"]
        }
    ))

    tools.append(Tool(
        name="quantity_simplify_units",
        description="Simplify units in an expression within a unit system.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression_key": {
                    "type": "string",
                    "description": "Key to expression with units"
                },
                "unit_system": {
                    "type": "string",
                    "enum": ["SI", "MKS", "MKSA"],
                    "description": "Unit system (default: SI)"
                }
            },
            "required": ["expression_key"]
        }
    ))

    # ========================================================================
    # CATEGORY G: Enhanced Matrix Operations (5 tools)
    # ========================================================================

    tools.append(Tool(
        name="create_matrix",
        description="Create a matrix from expression strings.",
        inputSchema={
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": "Matrix elements as string expressions"
                },
                "key": {
                    "type": "string",
                    "description": "Optional storage key"
                }
            },
            "required": ["elements"]
        }
    ))

    tools.append(Tool(
        name="matrix_determinant",
        description="Calculate the determinant of a matrix.",
        inputSchema={
            "type": "object",
            "properties": {
                "matrix_key": {
                    "type": "string",
                    "description": "Key to stored matrix"
                }
            },
            "required": ["matrix_key"]
        }
    ))

    tools.append(Tool(
        name="matrix_inverse",
        description="Calculate the inverse of a matrix.",
        inputSchema={
            "type": "object",
            "properties": {
                "matrix_key": {
                    "type": "string",
                    "description": "Key to stored matrix"
                },
                "store_key": {
                    "type": "string",
                    "description": "Optional key to store result"
                }
            },
            "required": ["matrix_key"]
        }
    ))

    tools.append(Tool(
        name="matrix_eigenvalues",
        description="Calculate the eigenvalues of a matrix.",
        inputSchema={
            "type": "object",
            "properties": {
                "matrix_key": {
                    "type": "string",
                    "description": "Key to stored matrix"
                }
            },
            "required": ["matrix_key"]
        }
    ))

    tools.append(Tool(
        name="matrix_eigenvectors",
        description="Calculate the eigenvectors of a matrix.",
        inputSchema={
            "type": "object",
            "properties": {
                "matrix_key": {
                    "type": "string",
                    "description": "Key to stored matrix"
                }
            },
            "required": ["matrix_key"]
        }
    ))

    # ========================================================================
    # CATEGORY G: Probability & Statistics (8 tools)
    # ========================================================================

    tools.append(Tool(
        name="calculate_probability",
        description="Calculate probabilities for discrete and continuous probability distributions symbolically.",
        inputSchema={
            "type": "object",
            "properties": {
                "distribution": {
                    "type": "string",
                    "enum": ["normal", "binomial", "poisson", "exponential", "uniform", "geometric"],
                    "description": "Type of probability distribution"
                },
                "parameters": {
                    "type": "object",
                    "description": "Distribution parameters (e.g., mu, sigma for normal; n, p for binomial)"
                },
                "calculation": {
                    "type": "string",
                    "enum": ["pdf", "cdf", "expectation", "variance", "moment"],
                    "description": "Type of calculation to perform"
                },
                "at_value": {
                    "type": "string",
                    "description": "Optional: Evaluate at specific value (symbolic expression)"
                }
            },
            "required": ["distribution", "parameters", "calculation"]
        }
    ))

    tools.append(Tool(
        name="bayesian_inference",
        description="Perform Bayesian inference calculations using Bayes' theorem symbolically. P(H|E) = P(E|H) * P(H) / P(E)",
        inputSchema={
            "type": "object",
            "properties": {
                "prior": {
                    "type": "string",
                    "description": "Prior probability P(H) as symbolic expression"
                },
                "likelihood": {
                    "type": "string",
                    "description": "Likelihood P(E|H) as symbolic expression"
                },
                "evidence": {
                    "type": "string",
                    "description": "Evidence probability P(E) as symbolic expression (optional, can be computed)"
                },
                "simplify": {
                    "type": "boolean",
                    "description": "Simplify the posterior probability (default: true)",
                    "default": True
                }
            },
            "required": ["prior", "likelihood"]
        }
    ))

    tools.append(Tool(
        name="statistical_test",
        description="Perform statistical hypothesis tests symbolically (t-test, chi-square, etc.).",
        inputSchema={
            "type": "object",
            "properties": {
                "test_type": {
                    "type": "string",
                    "enum": ["t_test", "chi_square", "z_test", "f_test"],
                    "description": "Type of statistical test"
                },
                "sample_statistics": {
                    "type": "object",
                    "description": "Sample statistics (mean, variance, n, etc.) as symbolic expressions"
                },
                "null_hypothesis": {
                    "type": "string",
                    "description": "Null hypothesis value (e.g., 'mu = 0')"
                },
                "alpha": {
                    "type": "string",
                    "description": "Significance level (default: '0.05')",
                    "default": "0.05"
                }
            },
            "required": ["test_type", "sample_statistics"]
        }
    ))

    tools.append(Tool(
        name="distribution_properties",
        description="Calculate moments, variance, skewness, kurtosis, and other properties of probability distributions symbolically.",
        inputSchema={
            "type": "object",
            "properties": {
                "distribution": {
                    "type": "string",
                    "enum": ["normal", "binomial", "poisson", "exponential", "uniform", "geometric", "custom"],
                    "description": "Type of distribution"
                },
                "parameters": {
                    "type": "object",
                    "description": "Distribution parameters"
                },
                "pdf_expression": {
                    "type": "string",
                    "description": "For custom distributions: PDF as symbolic expression"
                },
                "properties": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["mean", "variance", "std", "skewness", "kurtosis", "moment"]
                    },
                    "description": "Properties to calculate"
                },
                "moment_order": {
                    "type": "integer",
                    "description": "For moment calculation: order of moment (default: 1)",
                    "default": 1
                }
            },
            "required": ["distribution", "properties"]
        }
    ))

    tools.append(Tool(
        name="correlation_analysis",
        description="Calculate correlation, covariance, and other dependency measures symbolically between random variables.",
        inputSchema={
            "type": "object",
            "properties": {
                "variable_x": {
                    "type": "string",
                    "description": "First random variable (symbolic)"
                },
                "variable_y": {
                    "type": "string",
                    "description": "Second random variable (symbolic)"
                },
                "joint_pdf": {
                    "type": "string",
                    "description": "Joint probability density function"
                },
                "calculate": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["covariance", "correlation", "independence_test"]
                    },
                    "description": "What to calculate"
                }
            },
            "required": ["variable_x", "variable_y", "calculate"]
        }
    ))

    tools.append(Tool(
        name="regression_symbolic",
        description="Perform symbolic regression to derive regression equations and coefficients.",
        inputSchema={
            "type": "object",
            "properties": {
                "independent_vars": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Independent variable names"
                },
                "dependent_var": {
                    "type": "string",
                    "description": "Dependent variable name"
                },
                "model_type": {
                    "type": "string",
                    "enum": ["linear", "polynomial", "multiple_linear"],
                    "description": "Type of regression model"
                },
                "polynomial_degree": {
                    "type": "integer",
                    "description": "For polynomial regression: degree (default: 2)",
                    "default": 2
                },
                "derive_ols": {
                    "type": "boolean",
                    "description": "Derive ordinary least squares solution symbolically (default: true)",
                    "default": True
                }
            },
            "required": ["independent_vars", "dependent_var", "model_type"]
        }
    ))

    tools.append(Tool(
        name="confidence_intervals",
        description="Calculate confidence intervals symbolically for various parameters (mean, proportion, difference, etc.).",
        inputSchema={
            "type": "object",
            "properties": {
                "parameter": {
                    "type": "string",
                    "enum": ["mean", "proportion", "variance", "difference_means"],
                    "description": "Parameter to estimate"
                },
                "sample_stats": {
                    "type": "object",
                    "description": "Sample statistics (e.g., x_bar, s, n)"
                },
                "confidence_level": {
                    "type": "string",
                    "description": "Confidence level as symbolic expression (default: '0.95')",
                    "default": "0.95"
                },
                "distribution": {
                    "type": "string",
                    "enum": ["t", "z", "chi_square"],
                    "description": "Distribution for critical value (default: 't')",
                    "default": "t"
                }
            },
            "required": ["parameter", "sample_stats"]
        }
    ))

    tools.append(Tool(
        name="probability_distributions",
        description="Work with probability distributions: create, combine, transform, and analyze distributions symbolically.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "sum", "product", "transform", "convolution"],
                    "description": "Operation to perform"
                },
                "distribution_type": {
                    "type": "string",
                    "enum": ["normal", "binomial", "poisson", "exponential", "uniform", "custom"],
                    "description": "Type of distribution (for create operation)"
                },
                "parameters": {
                    "type": "object",
                    "description": "Distribution parameters"
                },
                "distributions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Distribution names for combination operations"
                },
                "transformation": {
                    "type": "string",
                    "description": "Transformation function (e.g., '2*X + 3')"
                },
                "store_key": {
                    "type": "string",
                    "description": "Key to store the resulting distribution"
                }
            },
            "required": ["operation"]
        }
    ))

    # ========================================================================
    # CATEGORY H: Transform Theory (6 tools)
    # ========================================================================

    tools.append(Tool(
        name="laplace_transform",
        description="Compute Laplace transform of expressions symbolically. L{f(t)}(s).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression in time domain (use 't' as variable)"
                },
                "inverse": {
                    "type": "boolean",
                    "description": "Compute inverse Laplace transform (default: false)",
                    "default": False
                },
                "store_key": {
                    "type": "string",
                    "description": "Key to store result"
                }
            },
            "required": ["expression"]
        }
    ))

    tools.append(Tool(
        name="fourier_transform",
        description="Compute Fourier transform symbolically. F{f(t)}(ω).",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression in time domain"
                },
                "inverse": {
                    "type": "boolean",
                    "description": "Compute inverse Fourier transform (default: false)",
                    "default": False
                },
                "transform_type": {
                    "type": "string",
                    "enum": ["standard", "discrete", "fast"],
                    "description": "Type of Fourier transform (default: standard)",
                    "default": "standard"
                },
                "store_key": {
                    "type": "string",
                    "description": "Key to store result"
                }
            },
            "required": ["expression"]
        }
    ))

    tools.append(Tool(
        name="z_transform",
        description="Compute Z-transform for discrete-time signals. Z{x[n]}(z).",
        inputSchema={
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "description": "Discrete sequence expression (use 'n' as variable)"
                },
                "inverse": {
                    "type": "boolean",
                    "description": "Compute inverse Z-transform (default: false)",
                    "default": False
                },
                "store_key": {
                    "type": "string",
                    "description": "Key to store result"
                }
            },
            "required": ["sequence"]
        }
    ))

    tools.append(Tool(
        name="convolution",
        description="Compute convolution of two functions or sequences symbolically.",
        inputSchema={
            "type": "object",
            "properties": {
                "function1": {
                    "type": "string",
                    "description": "First function f(t) or sequence"
                },
                "function2": {
                    "type": "string",
                    "description": "Second function g(t) or sequence"
                },
                "convolution_type": {
                    "type": "string",
                    "enum": ["continuous", "discrete", "circular"],
                    "description": "Type of convolution (default: continuous)",
                    "default": "continuous"
                },
                "limits": {
                    "type": "object",
                    "description": "Integration limits (for continuous)"
                }
            },
            "required": ["function1", "function2"]
        }
    ))

    tools.append(Tool(
        name="transfer_function_analysis",
        description="Analyze transfer functions for control systems: poles, zeros, stability, frequency response.",
        inputSchema={
            "type": "object",
            "properties": {
                "numerator": {
                    "type": "string",
                    "description": "Numerator polynomial"
                },
                "denominator": {
                    "type": "string",
                    "description": "Denominator polynomial"
                },
                "analysis": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["poles", "zeros", "stability", "step_response", "impulse_response", "bode"]
                    },
                    "description": "Types of analysis to perform"
                },
                "variable": {
                    "type": "string",
                    "description": "Transfer function variable (default: 's')",
                    "default": "s"
                }
            },
            "required": ["numerator", "denominator", "analysis"]
        }
    ))

    tools.append(Tool(
        name="mellin_transform",
        description="Compute Mellin transform and its inverse symbolically.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expression to transform"
                },
                "inverse": {
                    "type": "boolean",
                    "description": "Compute inverse Mellin transform (default: false)",
                    "default": False
                },
                "variable": {
                    "type": "string",
                    "description": "Variable name (default: 'x')",
                    "default": "x"
                }
            },
            "required": ["expression"]
        }
    ))

    # ========================================================================
    # CATEGORY I: Optimization Extensions (5 tools)
    # ========================================================================

    tools.append(Tool(
        name="lagrange_multipliers",
        description="Solve constrained optimization using Lagrange multipliers. Find extrema of f subject to g=0.",
        inputSchema={
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Objective function to optimize"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Equality constraints (each should equal 0)"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optimization variables"
                }
            },
            "required": ["objective", "constraints", "variables"]
        }
    ))

    tools.append(Tool(
        name="linear_programming",
        description="Solve linear programming problems: maximize/minimize linear objective subject to linear constraints.",
        inputSchema={
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Linear objective function"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Linear constraints (inequalities)"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Decision variables"
                },
                "maximize": {
                    "type": "boolean",
                    "description": "Maximize (true) or minimize (false) (default: false)",
                    "default": False
                },
                "method": {
                    "type": "string",
                    "enum": ["simplex", "interior_point", "symbolic"],
                    "description": "Solution method (default: symbolic)",
                    "default": "symbolic"
                }
            },
            "required": ["objective", "constraints", "variables"]
        }
    ))

    tools.append(Tool(
        name="convex_optimization",
        description="Verify convexity and solve convex optimization problems symbolically.",
        inputSchema={
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Objective function"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optimization variables"
                },
                "operation": {
                    "type": "string",
                    "enum": ["verify_convex", "verify_concave", "find_minimum", "hessian"],
                    "description": "Operation to perform"
                },
                "domain": {
                    "type": "object",
                    "description": "Domain constraints"
                }
            },
            "required": ["objective", "variables", "operation"]
        }
    ))

    tools.append(Tool(
        name="calculus_of_variations",
        description="Solve variational problems using Euler-Lagrange equations. Optimize functionals.",
        inputSchema={
            "type": "object",
            "properties": {
                "functional": {
                    "type": "string",
                    "description": "Integrand of functional L(x, y, y')"
                },
                "dependent_var": {
                    "type": "string",
                    "description": "Dependent variable (e.g., 'y')"
                },
                "independent_var": {
                    "type": "string",
                    "description": "Independent variable (e.g., 'x')"
                },
                "boundary_conditions": {
                    "type": "object",
                    "description": "Boundary conditions"
                }
            },
            "required": ["functional", "dependent_var", "independent_var"]
        }
    ))

    tools.append(Tool(
        name="dynamic_programming",
        description="Solve dynamic programming problems symbolically using Bellman equations.",
        inputSchema={
            "type": "object",
            "properties": {
                "value_function": {
                    "type": "string",
                    "description": "Value function or cost function"
                },
                "state_variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "State variables"
                },
                "control_variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Control/decision variables"
                },
                "transition": {
                    "type": "string",
                    "description": "State transition function"
                },
                "horizon": {
                    "type": "string",
                    "enum": ["finite", "infinite"],
                    "description": "Time horizon (default: finite)",
                    "default": "finite"
                }
            },
            "required": ["value_function", "state_variables"]
        }
    ))

    return tools


async def handle_advanced_tool(name: str, arguments: dict, ai) -> list[TextContent]:
    """
    Handle execution of advanced tools.

    Args:
        name: Tool name
        arguments: Tool arguments
        ai: SymbolicAI instance

    Returns:
        list[TextContent]: Result messages
    """
    try:
        # ================================================================
        # CATEGORY A: Variable & Expression Management
        # ================================================================

        if name == "intro":
            var_name = arguments["name"]
            positive_assumptions = arguments.get("positive_assumptions", [])
            negative_assumptions = arguments.get("negative_assumptions", [])

            # Build kwargs for Symbol()
            kwargs = {}
            for assumption in positive_assumptions:
                kwargs[assumption] = True
            for assumption in negative_assumptions:
                kwargs[assumption] = False

            # Create and store
            var = Symbol(var_name, **kwargs)
            ai.variables[var_name] = var

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "variable": var_name,
                    "assumptions": kwargs,
                    "message": f"Variable '{var_name}' created with assumptions: {kwargs}"
                }, indent=2)
            )]

        elif name == "intro_many":
            names = arguments["names"]
            positive_assumptions = arguments.get("positive_assumptions", [])

            kwargs = {assumption: True for assumption in positive_assumptions}
            vars_created = []

            for var_name in names:
                var = Symbol(var_name, **kwargs)
                ai.variables[var_name] = var
                vars_created.append(var_name)

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "variables": vars_created,
                    "assumptions": kwargs,
                    "message": f"Created variables: {', '.join(vars_created)}"
                }, indent=2)
            )]

        elif name == "introduce_expression":
            expression = arguments["expression"]
            key = arguments.get("key")

            # Parse using available variables and functions
            local_dict = {**ai.variables, **ai.functions}
            expr = sp.sympify(expression, locals=local_dict)

            # Generate or use provided key
            if key is None:
                key = ai._get_next_key("expr")

            ai.expressions[key] = expr

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "key": key,
                    "expression": str(expr),
                    "latex": latex(expr),
                    "pretty": pretty(expr)
                }, indent=2)
            )]

        elif name == "print_latex_expression":
            key = arguments["key"]

            if key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Expression '{key}' not found",
                        "available_keys": list(ai.expressions.keys())
                    }, indent=2)
                )]

            expr = ai.expressions[key]

            return [TextContent(
                type="text",
                text=json.dumps({
                    "key": key,
                    "expression": str(expr),
                    "latex": latex(expr),
                    "pretty": pretty(expr)
                }, indent=2)
            )]

        # ================================================================
        # CATEGORY B: Advanced Solvers
        # ================================================================

        elif name == "solve_algebraically":
            expression_key = arguments["expression_key"]
            variable = arguments["variable"]
            domain = arguments.get("domain", "complex")

            if expression_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Expression '{expression_key}' not found"}, indent=2)
                )]

            if variable not in ai.variables:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Variable '{variable}' not introduced"}, indent=2)
                )]

            expr = ai.expressions[expression_key]
            var = ai.variables[variable]

            domain_map = {
                "complex": sp.S.Complexes,
                "real": sp.S.Reals,
                "integers": sp.S.Integers,
                "naturals": sp.S.Naturals
            }

            solutions = solveset(expr, var, domain=domain_map[domain])

            return [TextContent(
                type="text",
                text=json.dumps({
                    "equation": str(expr),
                    "variable": variable,
                    "domain": domain,
                    "solutions": str(solutions),
                    "latex": latex(solutions)
                }, indent=2)
            )]

        elif name == "solve_linear_system":
            expression_keys = arguments["expression_keys"]
            variables = arguments["variables"]

            equations = [ai.expressions[key] for key in expression_keys if key in ai.expressions]
            vars_list = [ai.variables[var] for var in variables if var in ai.variables]

            if len(equations) != len(expression_keys):
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Some expression keys not found"}, indent=2)
                )]

            solutions = linsolve(equations, vars_list)

            result_key = ai._get_next_key("solution")
            ai.solutions[result_key] = solutions

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "result_key": result_key,
                    "solutions": str(solutions),
                    "latex": latex(solutions)
                }, indent=2)
            )]

        elif name == "solve_nonlinear_system":
            expression_keys = arguments["expression_keys"]
            variables = arguments["variables"]

            equations = [ai.expressions[key] for key in expression_keys if key in ai.expressions]
            vars_list = [ai.variables[var] for var in variables if var in ai.variables]

            solutions = nonlinsolve(equations, vars_list)

            result_key = ai._get_next_key("solution")
            ai.solutions[result_key] = solutions

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "result_key": result_key,
                    "solutions": str(solutions),
                    "latex": latex(solutions)
                }, indent=2)
            )]

        elif name == "introduce_function":
            func_name = arguments["name"]

            func = Function(func_name)
            ai.functions[func_name] = func

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "function": func_name,
                    "message": f"Function '{func_name}' created for differential equations"
                }, indent=2)
            )]

        elif name == "dsolve_ode":
            equation_key = arguments["equation_key"]
            function_name = arguments["function_name"]
            hint = arguments.get("hint")
            ics = arguments.get("ics")

            if equation_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Equation '{equation_key}' not found"}, indent=2)
                )]

            if function_name not in ai.functions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Function '{function_name}' not introduced"}, indent=2)
                )]

            equation = ai.expressions[equation_key]
            func = ai.functions[function_name]

            # Extract the free symbols from the equation to find the independent variable(s)
            # For ODE, we need func(var) format
            free_syms = equation.free_symbols

            # Find which variable the function depends on
            # This is a heuristic - look for function calls in the equation
            func_call = None
            for atom in equation.atoms(sp.core.function.AppliedUndef):
                if atom.func == func:
                    func_call = atom
                    break

            if func_call is None:
                # Try to construct it from the equation's free symbols
                # Assume single independent variable
                indep_vars = [s for s in free_syms if str(s) in ai.variables]
                if indep_vars:
                    func_call = func(indep_vars[0])
                else:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": "Could not determine function variable from equation"}, indent=2)
                    )]

            kwargs = {}
            if hint:
                kwargs['hint'] = hint
            if ics:
                # Parse initial conditions
                parsed_ics = {}
                for k, v in ics.items():
                    key_expr = sp.sympify(k, locals={**ai.variables, **ai.functions})
                    parsed_ics[key_expr] = sp.sympify(v)
                kwargs['ics'] = parsed_ics

            solution = dsolve(equation, func_call, **kwargs)

            result_key = ai._get_next_key("ode_solution")
            ai.expressions[result_key] = solution

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "result_key": result_key,
                    "solution": str(solution),
                    "latex": latex(solution)
                }, indent=2)
            )]

        elif name == "pdsolve_pde":
            equation_key = arguments["equation_key"]
            function_name = arguments["function_name"]
            hint = arguments.get("hint")

            if equation_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Equation '{equation_key}' not found"}, indent=2)
                )]

            if function_name not in ai.functions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Function '{function_name}' not introduced"}, indent=2)
                )]

            equation = ai.expressions[equation_key]
            func = ai.functions[function_name]

            # Find function call in equation (similar to ODE)
            func_call = None
            for atom in equation.atoms(sp.core.function.AppliedUndef):
                if atom.func == func:
                    func_call = atom
                    break

            if func_call is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Could not find function call in equation"}, indent=2)
                )]

            kwargs = {}
            if hint:
                kwargs['hint'] = hint

            solution = pdsolve(equation, func_call, **kwargs)

            result_key = ai._get_next_key("pde_solution")
            ai.expressions[result_key] = solution

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "result_key": result_key,
                    "solution": str(solution),
                    "latex": latex(solution)
                }, indent=2)
            )]

        # ================================================================
        # CATEGORY C: Tensor & General Relativity (Requires EinsteinPy)
        # ================================================================

        elif name == "create_predefined_metric":
            if not EINSTEINPY_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "EinsteinPy not installed",
                        "message": "Install with: pip install einsteinpy"
                    }, indent=2)
                )]

            metric_type = arguments["metric_type"]
            parameters = arguments.get("parameters", {})
            key = arguments.get("key")

            metric_map = {
                "Schwarzschild": Schwarzschild,
                "Kerr": Kerr,
                "KerrNewman": KerrNewman,
                "Minkowski": Minkowski,
                "DeSitter": DeSitter,
                "AntiDeSitter": AntiDeSitter,
                "Godel": Godel,
                "BesselGravitationalWave": BesselGravitationalWave
            }

            if metric_type not in metric_map:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Unknown metric: {metric_type}",
                        "available": list(metric_map.keys())
                    }, indent=2)
                )]

            metric_constructor = metric_map[metric_type]

            # Parse parameters
            parsed_params = {}
            for k, v in parameters.items():
                if isinstance(v, str) and v in ai.variables:
                    parsed_params[k] = ai.variables[v]
                elif isinstance(v, str):
                    # Convert string to sympy symbol
                    parsed_params[k] = sp.Symbol(v)
                else:
                    parsed_params[k] = v

            # Try to construct the metric
            try:
                if parsed_params:
                    metric = metric_constructor(**parsed_params)
                else:
                    metric = metric_constructor()
            except TypeError:
                # Some metrics don't accept parameters in the constructor
                # Try without parameters and note the parameters separately
                metric = metric_constructor()
                # Store parameters as metadata
                if not hasattr(metric, 'parameters'):
                    metric.parameters = parsed_params

            if key is None:
                key = f"metric_{metric_type}"

            ai.metrics[key] = metric

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "key": key,
                    "metric_type": metric_type,
                    "latex": latex(metric.tensor())
                }, indent=2)
            )]

        elif name == "search_predefined_metrics":
            query = arguments.get("query", "")

            metrics_info = {
                "Schwarzschild": "Spherically symmetric vacuum solution (black hole)",
                "Kerr": "Rotating black hole in Boyer-Lindquist coordinates",
                "KerrNewman": "Rotating charged black hole",
                "Minkowski": "Flat spacetime in polar coordinates",
                "DeSitter": "Positive cosmological constant spacetime",
                "AntiDeSitter": "Negative cosmological constant spacetime",
                "Godel": "Rotating universe solution",
                "BesselGravitationalWave": "Cylindrically symmetric gravitational wave"
            }

            if query:
                filtered = {k: v for k, v in metrics_info.items()
                           if query.lower() in k.lower() or query.lower() in v.lower()}
            else:
                filtered = metrics_info

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "available_metrics": filtered,
                    "einsteinpy_installed": EINSTEINPY_AVAILABLE
                }, indent=2)
            )]

        elif name == "calculate_tensor":
            if not EINSTEINPY_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "EinsteinPy not installed",
                        "message": "Install with: pip install einsteinpy"
                    }, indent=2)
                )]

            metric_key = arguments["metric_key"]
            tensor_type = arguments["tensor_type"]
            store_key = arguments.get("store_key")

            if metric_key not in ai.metrics:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Metric '{metric_key}' not found",
                        "available": list(ai.metrics.keys())
                    }, indent=2)
                )]

            metric = ai.metrics[metric_key]

            tensor_map = {
                "Christoffel": lambda m: ChristoffelSymbols.from_metric(m),
                "Ricci": lambda m: RicciTensor.from_metric(m),
                "RicciScalar": lambda m: RicciScalar.from_riccitensor(RicciTensor.from_metric(m)),
                "Einstein": lambda m: EinsteinTensor.from_metric(m),
                "Weyl": lambda m: WeylTensor.from_metric(m)
            }

            if tensor_type not in tensor_map:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Unknown tensor: {tensor_type}",
                        "available": list(tensor_map.keys())
                    }, indent=2)
                )]

            tensor = tensor_map[tensor_type](metric)

            if store_key is None:
                store_key = ai._get_next_key(f"tensor_{tensor_type}")

            ai.tensor_objects[store_key] = tensor

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "key": store_key,
                    "tensor_type": tensor_type,
                    "latex": latex(tensor.tensor())
                }, indent=2)
            )]

        elif name == "create_custom_metric":
            if not EINSTEINPY_AVAILABLE:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "EinsteinPy not installed",
                        "message": "Install with: pip install einsteinpy"
                    }, indent=2)
                )]

            components = arguments["components"]
            coordinates = arguments["coordinates"]
            key = arguments.get("key")

            # Parse components
            local_dict = {**ai.variables, **ai.functions}
            parsed_components = []
            for row in components:
                parsed_row = []
                for expr_str in row:
                    expr = sp.sympify(expr_str, locals=local_dict)
                    parsed_row.append(expr)
                parsed_components.append(parsed_row)

            # Create coordinate symbols
            coord_symbols = symbols(coordinates)
            for name_str, sym in zip(coordinates, coord_symbols):
                ai.variables[name_str] = sym

            # Create metric
            metric = MetricTensor(
                arr=parsed_components,
                syms=coord_symbols,
                config='ll',
                name=key or "custom_metric"
            )

            if key is None:
                key = ai._get_next_key("metric_custom")

            ai.metrics[key] = metric

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "key": key,
                    "coordinates": coordinates,
                    "latex": latex(metric.tensor())
                }, indent=2)
            )]

        elif name == "print_latex_tensor":
            key = arguments["key"]

            if key in ai.tensor_objects:
                tensor = ai.tensor_objects[key]
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "key": key,
                        "type": "tensor",
                        "latex": latex(tensor.tensor())
                    }, indent=2)
                )]
            elif key in ai.metrics:
                metric = ai.metrics[key]
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "key": key,
                        "type": "metric",
                        "latex": latex(metric.tensor())
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"Tensor or metric '{key}' not found",
                        "available_tensors": list(ai.tensor_objects.keys()),
                        "available_metrics": list(ai.metrics.keys())
                    }, indent=2)
                )]

        # ================================================================
        # CATEGORY D: Expression Operations
        # ================================================================

        elif name == "simplify_expression":
            expression_key = arguments["expression_key"]
            method = arguments.get("method", "default")

            if expression_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Expression '{expression_key}' not found"}, indent=2)
                )]

            expr = ai.expressions[expression_key]

            method_map = {
                "default": simplify,
                "trigsimp": trigsimp,
                "radsimp": radsimp,
                "powsimp": powsimp,
                "logcombine": logcombine
            }

            simplified = method_map[method](expr)

            result_key = f"simplified_{expression_key}"
            ai.expressions[result_key] = simplified

            return [TextContent(
                type="text",
                text=json.dumps({
                    "original_key": expression_key,
                    "result_key": result_key,
                    "method": method,
                    "result": str(simplified),
                    "latex": latex(simplified)
                }, indent=2)
            )]

        elif name == "substitute_expression":
            expression_key = arguments["expression_key"]
            substitutions = arguments["substitutions"]
            store_key = arguments.get("store_key")

            if expression_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Expression '{expression_key}' not found"}, indent=2)
                )]

            expr = ai.expressions[expression_key]

            # Parse substitutions
            subs_dict = {}
            for var_name, value in substitutions.items():
                if var_name not in ai.variables:
                    continue
                var = ai.variables[var_name]

                # Value can be number, variable name, or expression key
                if isinstance(value, str) and value in ai.variables:
                    subs_dict[var] = ai.variables[value]
                elif isinstance(value, str) and value in ai.expressions:
                    subs_dict[var] = ai.expressions[value]
                else:
                    subs_dict[var] = sp.sympify(value)

            result = expr.subs(subs_dict)

            if store_key is None:
                store_key = f"subst_{expression_key}"

            ai.expressions[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "original_key": expression_key,
                    "result_key": store_key,
                    "substitutions": str(subs_dict),
                    "result": str(result),
                    "latex": latex(result)
                }, indent=2)
            )]

        elif name == "integrate_expression":
            expression_key = arguments["expression_key"]
            variable = arguments["variable"]
            lower_bound = arguments.get("lower_bound")
            upper_bound = arguments.get("upper_bound")
            store_key = arguments.get("store_key")

            if expression_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Expression '{expression_key}' not found"}, indent=2)
                )]

            if variable not in ai.variables:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Variable '{variable}' not introduced"}, indent=2)
                )]

            expr = ai.expressions[expression_key]
            var = ai.variables[variable]

            if lower_bound and upper_bound:
                # Definite integral
                lb = sp.sympify(lower_bound, locals=ai.variables)
                ub = sp.sympify(upper_bound, locals=ai.variables)
                result = sp.integrate(expr, (var, lb, ub))
                integral_type = "definite"
            else:
                # Indefinite integral
                result = sp.integrate(expr, var)
                integral_type = "indefinite"

            if store_key is None:
                store_key = f"integral_{expression_key}"

            ai.expressions[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "original_key": expression_key,
                    "result_key": store_key,
                    "integral_type": integral_type,
                    "variable": variable,
                    "result": str(result),
                    "latex": latex(result)
                }, indent=2)
            )]

        elif name == "differentiate_expression":
            expression_key = arguments["expression_key"]
            variable = arguments["variable"]
            order = arguments.get("order", 1)
            store_key = arguments.get("store_key")

            if expression_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Expression '{expression_key}' not found"}, indent=2)
                )]

            if variable not in ai.variables:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Variable '{variable}' not introduced"}, indent=2)
                )]

            expr = ai.expressions[expression_key]
            var = ai.variables[variable]

            result = sp.diff(expr, var, order)

            if store_key is None:
                store_key = f"derivative_{expression_key}"

            ai.expressions[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "original_key": expression_key,
                    "result_key": store_key,
                    "variable": variable,
                    "order": order,
                    "result": str(result),
                    "latex": latex(result)
                }, indent=2)
            )]

        # ================================================================
        # CATEGORY E: Vector Calculus
        # ================================================================

        elif name == "create_coordinate_system":
            sys_name = arguments["name"]
            coord_type = arguments.get("coord_type", "cartesian")

            if coord_type == "cartesian":
                coords = CoordSys3D(sys_name)
            elif coord_type == "cylindrical":
                coords = CoordSys3D(sys_name, transformation='cylindrical')
            elif coord_type == "spherical":
                coords = CoordSys3D(sys_name, transformation='spherical')
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Invalid coordinate type: {coord_type}"}, indent=2)
                )]

            ai.coordinate_systems[sys_name] = coords

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "name": sys_name,
                    "type": coord_type,
                    "basis": f"{coords.i}, {coords.j}, {coords.k}"
                }, indent=2)
            )]

        elif name == "create_vector_field":
            coord_system_name = arguments["coord_system_name"]
            components = arguments["components"]
            key = arguments.get("key")

            if coord_system_name not in ai.coordinate_systems:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Coordinate system '{coord_system_name}' not found"}, indent=2)
                )]

            coords = ai.coordinate_systems[coord_system_name]

            # Parse components
            local_dict = {**ai.variables, **ai.functions}
            local_dict.update({
                'x': coords.x,
                'y': coords.y,
                'z': coords.z
            })

            # Build vector from components
            i_expr = sp.sympify(components.get('i', '0'), locals=local_dict)
            j_expr = sp.sympify(components.get('j', '0'), locals=local_dict)
            k_expr = sp.sympify(components.get('k', '0'), locals=local_dict)

            vector = i_expr * coords.i + j_expr * coords.j + k_expr * coords.k

            if key is None:
                key = ai._get_next_key("vector")

            ai.vector_fields[key] = vector

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "key": key,
                    "vector": str(vector)
                }, indent=2)
            )]

        elif name == "calculate_curl":
            vector_field_key = arguments["vector_field_key"]
            store_key = arguments.get("store_key")

            if vector_field_key not in ai.vector_fields:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Vector field '{vector_field_key}' not found"}, indent=2)
                )]

            vector = ai.vector_fields[vector_field_key]
            result = curl(vector)

            if store_key is None:
                store_key = f"curl_{vector_field_key}"

            ai.vector_fields[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "original_key": vector_field_key,
                    "result_key": store_key,
                    "curl": str(result)
                }, indent=2)
            )]

        elif name == "calculate_divergence":
            vector_field_key = arguments["vector_field_key"]
            store_key = arguments.get("store_key")

            if vector_field_key not in ai.vector_fields:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Vector field '{vector_field_key}' not found"}, indent=2)
                )]

            vector = ai.vector_fields[vector_field_key]
            result = divergence(vector)

            if store_key is None:
                store_key = f"div_{vector_field_key}"

            ai.expressions[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "original_key": vector_field_key,
                    "result_key": store_key,
                    "divergence": str(result),
                    "latex": latex(result)
                }, indent=2)
            )]

        elif name == "calculate_gradient":
            expression_key = arguments["expression_key"]
            coord_system_name = arguments["coord_system_name"]
            store_key = arguments.get("store_key")

            if expression_key not in ai.expressions:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Expression '{expression_key}' not found"}, indent=2)
                )]

            if coord_system_name not in ai.coordinate_systems:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Coordinate system '{coord_system_name}' not found"}, indent=2)
                )]

            expr = ai.expressions[expression_key]
            coords = ai.coordinate_systems[coord_system_name]

            result = gradient(expr, coords)

            if store_key is None:
                store_key = f"grad_{expression_key}"

            ai.vector_fields[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "original_key": expression_key,
                    "result_key": store_key,
                    "gradient": str(result)
                }, indent=2)
            )]

        # ================================================================
        # CATEGORY F: Unit Operations
        # ================================================================

        elif name == "convert_to_units":
            # Support both APIs: expression (raw) or expression_key (lookup)
            # Support both APIs: target_units (raw) or target_unit (lookup)
            unit_system = arguments.get("unit_system", "SI")

            from sympy.physics.units.systems import SI, MKS, MKSA, natural, cgs
            from sympy.physics.units import (meter, second, kilogram, kelvin, newton, kilometer, hour, convert_to,
                                             gram, joule, watt, pascal, ampere, volt, ohm)

            # Create locals dict for sympify to recognize unit names
            unit_locals = {
                'meter': meter, 'second': second, 'kilogram': kilogram, 'kelvin': kelvin,
                'newton': newton, 'kilometer': kilometer, 'hour': hour, 'gram': gram,
                'joule': joule, 'watt': watt, 'pascal': pascal, 'ampere': ampere,
                'volt': volt, 'ohm': ohm
            }

            # Get expression - support both 'expression' (raw string) and 'expression_key' (lookup)
            if "expression" in arguments:
                # Parse raw expression string with unit objects
                expression_str = arguments["expression"]
                try:
                    expr = sympify(expression_str, locals=unit_locals)
                    original_ref = expression_str
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"status": "error", "error": f"Failed to parse expression: {str(e)}"}, indent=2)
                    )]
            elif "expression_key" in arguments:
                # Lookup expression by key
                expression_key = arguments["expression_key"]
                if expression_key not in ai.expressions:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"status": "error", "error": f"Expression '{expression_key}' not found"}, indent=2)
                    )]
                expr = ai.expressions[expression_key]
                original_ref = expression_key
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "error": "Must provide either 'expression' or 'expression_key'"}, indent=2)
                )]

            # Get target units - support both 'target_units' (raw string) and 'target_unit' (lookup)
            if "target_units" in arguments:
                # Parse raw target units string with unit objects
                target_units_str = arguments["target_units"]
                try:
                    target = sympify(target_units_str, locals=unit_locals)
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"status": "error", "error": f"Failed to parse target units: {str(e)}"}, indent=2)
                    )]
            elif "target_unit" in arguments:
                # Lookup target unit by key
                target_unit = arguments["target_unit"]
                if target_unit not in ai.variables:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"status": "error", "error": f"Unit '{target_unit}' not found in variables"}, indent=2)
                    )]
                target = ai.variables[target_unit]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "error": "Must provide either 'target_units' or 'target_unit'"}, indent=2)
                )]

            unit_system_map = {
                "SI": SI,
                "MKS": MKS,
                "MKSA": MKSA,
                "natural": natural,
                "CGS": cgs
            }

            try:
                result = convert_to(expr, target, unit_system=unit_system_map.get(unit_system, SI))
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "error": f"Conversion failed: {str(e)}"}, indent=2)
                )]

            store_key = f"converted_{hash(str(expr))}"
            ai.expressions[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "expression": str(expr),
                    "target_units": str(target),
                    "result": str(result),
                    "latex": latex(result)
                }, indent=2)
            )]

        elif name == "quantity_simplify_units":
            # Support both APIs: expression (raw) or expression_key (lookup)
            unit_system = arguments.get("unit_system", "SI")

            from sympy.physics.units.systems import SI, MKS, MKSA
            from sympy.physics.units import meter, second, kilogram, newton

            # Get expression - support both 'expression' (raw string) and 'expression_key' (lookup)
            if "expression" in arguments:
                # Parse raw expression string
                expression_str = arguments["expression"]
                try:
                    expr = sympify(expression_str)
                    original_ref = expression_str
                except Exception as e:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"status": "error", "error": f"Failed to parse expression: {str(e)}"}, indent=2)
                    )]
            elif "expression_key" in arguments:
                # Lookup expression by key
                expression_key = arguments["expression_key"]
                if expression_key not in ai.expressions:
                    return [TextContent(
                        type="text",
                        text=json.dumps({"status": "error", "error": f"Expression '{expression_key}' not found"}, indent=2)
                    )]
                expr = ai.expressions[expression_key]
                original_ref = expression_key
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "error": "Must provide either 'expression' or 'expression_key'"}, indent=2)
                )]

            unit_system_map = {
                "SI": SI,
                "MKS": MKS,
                "MKSA": MKSA
            }

            try:
                result = quantity_simplify(expr, unit_system=unit_system_map.get(unit_system, SI))
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "error": f"Simplification failed: {str(e)}"}, indent=2)
                )]

            store_key = f"simplified_units_{hash(str(expr))}"
            ai.expressions[store_key] = result

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "expression": str(expr),
                    "simplified": str(result),
                    "result": str(result),
                    "latex": latex(result)
                }, indent=2)
            )]

        # ================================================================
        # CATEGORY G: Enhanced Matrix Operations
        # ================================================================

        elif name == "create_matrix":
            elements = arguments["elements"]
            key = arguments.get("key")

            # Parse elements
            local_dict = {**ai.variables, **ai.functions}
            parsed_elements = []
            for row in elements:
                parsed_row = []
                for expr_str in row:
                    expr = sp.sympify(expr_str, locals=local_dict)
                    parsed_row.append(expr)
                parsed_elements.append(parsed_row)

            matrix = Matrix(parsed_elements)

            if key is None:
                key = ai._get_next_key("matrix")

            ai.matrices[key] = matrix

            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "key": key,
                    "matrix": str(matrix),
                    "latex": latex(matrix)
                }, indent=2)
            )]

        elif name == "matrix_determinant":
            matrix_key = arguments["matrix_key"]

            if matrix_key not in ai.matrices:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Matrix '{matrix_key}' not found"}, indent=2)
                )]

            matrix = ai.matrices[matrix_key]
            det = matrix.det()

            store_key = f"det_{matrix_key}"
            ai.expressions[store_key] = det

            return [TextContent(
                type="text",
                text=json.dumps({
                    "matrix_key": matrix_key,
                    "result_key": store_key,
                    "determinant": str(det),
                    "latex": latex(det)
                }, indent=2)
            )]

        elif name == "matrix_inverse":
            matrix_key = arguments["matrix_key"]
            store_key = arguments.get("store_key")

            if matrix_key not in ai.matrices:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Matrix '{matrix_key}' not found"}, indent=2)
                )]

            matrix = ai.matrices[matrix_key]

            try:
                inverse = matrix.inv()
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Matrix not invertible: {str(e)}"}, indent=2)
                )]

            if store_key is None:
                store_key = f"inv_{matrix_key}"

            ai.matrices[store_key] = inverse

            return [TextContent(
                type="text",
                text=json.dumps({
                    "matrix_key": matrix_key,
                    "result_key": store_key,
                    "inverse": str(inverse),
                    "latex": latex(inverse)
                }, indent=2)
            )]

        elif name == "matrix_eigenvalues":
            matrix_key = arguments["matrix_key"]

            if matrix_key not in ai.matrices:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Matrix '{matrix_key}' not found"}, indent=2)
                )]

            matrix = ai.matrices[matrix_key]
            eigenvals = matrix.eigenvals()

            result_text = {}
            for eigenval, mult in eigenvals.items():
                result_text[str(eigenval)] = {
                    "value": str(eigenval),
                    "multiplicity": mult,
                    "latex": latex(eigenval)
                }

            return [TextContent(
                type="text",
                text=json.dumps({
                    "matrix_key": matrix_key,
                    "eigenvalues": result_text
                }, indent=2)
            )]

        elif name == "matrix_eigenvectors":
            matrix_key = arguments["matrix_key"]

            if matrix_key not in ai.matrices:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Matrix '{matrix_key}' not found"}, indent=2)
                )]

            matrix = ai.matrices[matrix_key]
            eigenvects = matrix.eigenvects()

            result_list = []
            for eigenval, mult, vects in eigenvects:
                vectors_data = []
                for i, vect in enumerate(vects):
                    vectors_data.append({
                        "vector": str(vect),
                        "latex": latex(vect)
                    })

                result_list.append({
                    "eigenvalue": str(eigenval),
                    "multiplicity": mult,
                    "eigenvectors": vectors_data
                })

            return [TextContent(
                type="text",
                text=json.dumps({
                    "matrix_key": matrix_key,
                    "eigenvectors": result_list
                }, indent=2)
            )]

        # ================================================================
        # CATEGORY G: Probability & Statistics
        # ================================================================

        elif name == "calculate_probability":
            distribution_type = arguments["distribution"]
            parameters = arguments["parameters"]
            calculation = arguments["calculation"]
            at_value = arguments.get("at_value")

            x = symbols('x', real=True)

            # Create distribution
            if distribution_type == "normal":
                mu = sp.sympify(parameters.get("mu", "0"))
                sigma = sp.sympify(parameters.get("sigma", "1"))
                dist = Normal('X', mu, sigma)
            elif distribution_type == "binomial":
                n = int(parameters.get("n", 10))
                p = sp.sympify(parameters.get("p", "0.5"))
                dist = Binomial('X', n, p)
            elif distribution_type == "poisson":
                lam = sp.sympify(parameters.get("lambda", "1"))
                dist = Poisson('X', lam)
            elif distribution_type == "exponential":
                rate = sp.sympify(parameters.get("rate", "1"))
                dist = Exponential('X', rate)
            elif distribution_type == "uniform":
                a = sp.sympify(parameters.get("a", "0"))
                b = sp.sympify(parameters.get("b", "1"))
                dist = Uniform('X', a, b)
            elif distribution_type == "geometric":
                p = sp.sympify(parameters.get("p", "0.5"))
                dist = Geometric('X', p)
            else:
                return [TextContent(type="text", text=json.dumps({"error": f"Unknown distribution: {distribution_type}"}, indent=2))]

            # Perform calculation
            result = {}
            if calculation == "pdf":
                pdf_expr = density(dist)(x)
                result["pdf"] = str(pdf_expr)
                result["latex"] = latex(pdf_expr)
                if at_value:
                    val = sp.sympify(at_value)
                    result["pdf_at_value"] = str(pdf_expr.subs(x, val))
            elif calculation == "cdf":
                cdf_expr = cdf(dist)(x)
                result["cdf"] = str(cdf_expr)
                result["latex"] = latex(cdf_expr)
                if at_value:
                    val = sp.sympify(at_value)
                    result["cdf_at_value"] = str(cdf_expr.subs(x, val))
            elif calculation == "expectation":
                exp_val = E(dist)
                result["expectation"] = str(exp_val)
                result["latex"] = latex(exp_val)
            elif calculation == "variance":
                var_val = variance(dist)
                result["variance"] = str(var_val)
                result["std"] = str(sqrt(var_val))
                result["latex"] = latex(var_val)
            elif calculation == "moment":
                order = int(parameters.get("order", 1))
                mom = moment(dist, order)
                result["moment"] = str(mom)
                result["order"] = order
                result["latex"] = latex(mom)

            result["distribution"] = distribution_type
            result["parameters"] = parameters
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "bayesian_inference":
            prior_str = arguments["prior"]
            likelihood_str = arguments["likelihood"]
            evidence_str = arguments.get("evidence")
            do_simplify = arguments.get("simplify", True)

            prior = sp.sympify(prior_str)
            likelihood = sp.sympify(likelihood_str)

            if evidence_str:
                evidence = sp.sympify(evidence_str)
            else:
                # If evidence not provided, it's just prior * likelihood (unnormalized)
                evidence = sp.sympify("1")

            # Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
            posterior = (likelihood * prior) / evidence

            if do_simplify:
                posterior = simplify(posterior)

            result = {
                "prior": str(prior),
                "likelihood": str(likelihood),
                "evidence": str(evidence),
                "posterior": str(posterior),
                "posterior_latex": latex(posterior),
                "formula": "P(H|E) = P(E|H) * P(H) / P(E)"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "statistical_test":
            test_type = arguments["test_type"]
            sample_stats = arguments["sample_statistics"]
            null_hypothesis = arguments.get("null_hypothesis", "")
            alpha = sp.sympify(arguments.get("alpha", "0.05"))

            result = {
                "test_type": test_type,
                "null_hypothesis": null_hypothesis,
                "alpha": str(alpha)
            }

            if test_type == "t_test":
                # t = (x_bar - mu_0) / (s / sqrt(n))
                x_bar = sp.sympify(sample_stats.get("mean", "x_bar"))
                mu_0 = sp.sympify(sample_stats.get("mu_0", "0"))
                s = sp.sympify(sample_stats.get("std", "s"))
                n = sp.sympify(sample_stats.get("n", "n"))

                t_statistic = (x_bar - mu_0) / (s / sqrt(n))
                t_statistic = simplify(t_statistic)

                result["t_statistic"] = str(t_statistic)
                result["t_statistic_latex"] = latex(t_statistic)
                result["degrees_of_freedom"] = str(n - 1)
                result["formula"] = "t = (x_bar - mu_0) / (s / sqrt(n))"

            elif test_type == "z_test":
                # z = (x_bar - mu_0) / (sigma / sqrt(n))
                x_bar = sp.sympify(sample_stats.get("mean", "x_bar"))
                mu_0 = sp.sympify(sample_stats.get("mu_0", "0"))
                sigma = sp.sympify(sample_stats.get("sigma", "sigma"))
                n = sp.sympify(sample_stats.get("n", "n"))

                z_statistic = (x_bar - mu_0) / (sigma / sqrt(n))
                z_statistic = simplify(z_statistic)

                result["z_statistic"] = str(z_statistic)
                result["z_statistic_latex"] = latex(z_statistic)
                result["formula"] = "z = (x_bar - mu_0) / (sigma / sqrt(n))"

            elif test_type == "chi_square":
                # chi^2 = sum((observed - expected)^2 / expected)
                observed = sp.sympify(sample_stats.get("observed", "O"))
                expected = sp.sympify(sample_stats.get("expected", "E"))

                chi_square = (observed - expected)**2 / expected
                chi_square = simplify(chi_square)

                result["chi_square_statistic"] = str(chi_square)
                result["chi_square_latex"] = latex(chi_square)
                result["formula"] = "chi^2 = (O - E)^2 / E"

            elif test_type == "f_test":
                # F = s1^2 / s2^2
                s1_sq = sp.sympify(sample_stats.get("var1", "s1**2"))
                s2_sq = sp.sympify(sample_stats.get("var2", "s2**2"))

                f_statistic = s1_sq / s2_sq
                f_statistic = simplify(f_statistic)

                result["f_statistic"] = str(f_statistic)
                result["f_statistic_latex"] = latex(f_statistic)
                result["formula"] = "F = s1^2 / s2^2"

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "distribution_properties":
            distribution_type = arguments["distribution"]
            parameters = arguments.get("parameters", {})
            pdf_expression = arguments.get("pdf_expression")
            properties_list = arguments["properties"]
            moment_order = arguments.get("moment_order", 1)

            x = symbols('x', real=True)

            # Create distribution
            if distribution_type == "normal":
                mu = sp.sympify(parameters.get("mu", "0"))
                sigma = sp.sympify(parameters.get("sigma", "1"))
                dist = Normal('X', mu, sigma)
            elif distribution_type == "binomial":
                n = int(parameters.get("n", 10))
                p = sp.sympify(parameters.get("p", "0.5"))
                dist = Binomial('X', n, p)
            elif distribution_type == "poisson":
                lam = sp.sympify(parameters.get("lambda", "1"))
                dist = Poisson('X', lam)
            elif distribution_type == "exponential":
                rate = sp.sympify(parameters.get("rate", "1"))
                dist = Exponential('X', rate)
            elif distribution_type == "uniform":
                a = sp.sympify(parameters.get("a", "0"))
                b = sp.sympify(parameters.get("b", "1"))
                dist = Uniform('X', a, b)
            elif distribution_type == "geometric":
                p = sp.sympify(parameters.get("p", "0.5"))
                dist = Geometric('X', p)
            elif distribution_type == "custom":
                return [TextContent(type="text", text=json.dumps({"error": "Custom distributions not yet fully implemented"}, indent=2))]
            else:
                return [TextContent(type="text", text=json.dumps({"error": f"Unknown distribution: {distribution_type}"}, indent=2))]

            result = {"distribution": distribution_type, "properties": {}}

            for prop in properties_list:
                if prop == "mean":
                    result["properties"]["mean"] = str(E(dist))
                elif prop == "variance":
                    result["properties"]["variance"] = str(variance(dist))
                elif prop == "std":
                    result["properties"]["std"] = str(sqrt(variance(dist)))
                elif prop == "skewness":
                    result["properties"]["skewness"] = str(skewness(dist))
                elif prop == "kurtosis":
                    result["properties"]["kurtosis"] = str(kurtosis(dist))
                elif prop == "moment":
                    result["properties"][f"moment_{moment_order}"] = str(moment(dist, moment_order))

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "correlation_analysis":
            var_x = sp.sympify(arguments["variable_x"])
            var_y = sp.sympify(arguments["variable_y"])
            joint_pdf_str = arguments.get("joint_pdf")
            calculate_list = arguments["calculate"]

            result = {
                "variable_x": str(var_x),
                "variable_y": str(var_y),
                "analysis": {}
            }

            # This is a simplified implementation - full correlation analysis
            # would require numerical integration or specific joint distributions
            for calc in calculate_list:
                if calc == "covariance":
                    # Cov(X,Y) = E[XY] - E[X]E[Y]
                    # Symbolic placeholder
                    cov_formula = "E[X*Y] - E[X]*E[Y]"
                    result["analysis"]["covariance_formula"] = cov_formula
                elif calc == "correlation":
                    # Cor(X,Y) = Cov(X,Y) / (std(X) * std(Y))
                    cor_formula = "Cov(X,Y) / (std(X) * std(Y))"
                    result["analysis"]["correlation_formula"] = cor_formula
                elif calc == "independence_test":
                    result["analysis"]["independence_test"] = "X and Y are independent if P(X,Y) = P(X)*P(Y)"

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "regression_symbolic":
            independent_vars = arguments["independent_vars"]
            dependent_var = arguments["dependent_var"]
            model_type = arguments["model_type"]
            poly_degree = arguments.get("polynomial_degree", 2)
            derive_ols = arguments.get("derive_ols", True)

            y = symbols(dependent_var)
            x_vars = symbols(' '.join(independent_vars))
            if not isinstance(x_vars, tuple):
                x_vars = (x_vars,)

            result = {
                "dependent_variable": dependent_var,
                "independent_variables": independent_vars,
                "model_type": model_type
            }

            if model_type == "linear":
                # y = beta_0 + beta_1*x
                beta_0, beta_1 = symbols('beta_0 beta_1')
                if len(x_vars) == 1:
                    model = beta_0 + beta_1 * x_vars[0]
                    result["model"] = str(model)
                    result["model_latex"] = latex(model)

                    if derive_ols:
                        # OLS: beta_1 = Cov(X,Y)/Var(X), beta_0 = mean(Y) - beta_1*mean(X)
                        result["ols_formula_beta_1"] = "Cov(X,Y) / Var(X)"
                        result["ols_formula_beta_0"] = "mean(Y) - beta_1 * mean(X)"

            elif model_type == "polynomial":
                # y = beta_0 + beta_1*x + beta_2*x^2 + ...
                betas = symbols(f'beta_0:{poly_degree + 1}')
                if len(x_vars) == 1:
                    model = sum(betas[i] * x_vars[0]**i for i in range(poly_degree + 1))
                    result["model"] = str(model)
                    result["model_latex"] = latex(model)
                    result["degree"] = poly_degree

            elif model_type == "multiple_linear":
                # y = beta_0 + beta_1*x1 + beta_2*x2 + ...
                betas = symbols(f'beta_0:{len(x_vars) + 1}')
                model = betas[0] + sum(betas[i+1] * x_vars[i] for i in range(len(x_vars)))
                result["model"] = str(model)
                result["model_latex"] = latex(model)

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "confidence_intervals":
            parameter = arguments["parameter"]
            sample_stats = arguments["sample_stats"]
            confidence_level_str = arguments.get("confidence_level", "0.95")
            confidence_level = sp.sympify(confidence_level_str)
            distribution = arguments.get("distribution", "t")

            result = {
                "parameter": parameter,
                "confidence_level": confidence_level_str,  # Preserve original string
                "distribution": distribution
            }

            # Critical value placeholder (would need stats tables for exact values)
            alpha = sp.Integer(1) - confidence_level
            if distribution == "t":
                crit_sym = symbols('t_alpha_2')
                n_val = sample_stats.get('n', 'n')
                df = sp.sympify(n_val) - 1 if n_val != 'n' else 'n-1'
                result["critical_value"] = f"t_(alpha/2) with df={df}"
            elif distribution == "z":
                crit_sym = symbols('z_alpha_2')
                result["critical_value"] = "z_(alpha/2)"
            elif distribution == "chi_square":
                crit_sym = symbols('chi_alpha_2')
                result["critical_value"] = "chi^2_(alpha/2)"

            if parameter == "mean":
                x_bar = sp.sympify(sample_stats.get("x_bar", "x_bar"))
                s = sp.sympify(sample_stats.get("s", "s"))
                n = sp.sympify(sample_stats.get("n", "n"))

                margin_of_error = crit_sym * s / sqrt(n)
                lower = x_bar - margin_of_error
                upper = x_bar + margin_of_error

                result["margin_of_error"] = str(margin_of_error)
                result["lower_bound"] = str(lower)
                result["upper_bound"] = str(upper)
                result["interval"] = f"({str(lower)}, {str(upper)})"

            elif parameter == "proportion":
                p_hat = sp.sympify(sample_stats.get("p_hat", "p_hat"))
                n = sp.sympify(sample_stats.get("n", "n"))

                margin_of_error = crit_sym * sqrt(p_hat * (1 - p_hat) / n)
                lower = p_hat - margin_of_error
                upper = p_hat + margin_of_error

                result["margin_of_error"] = str(margin_of_error)
                result["lower_bound"] = str(lower)
                result["upper_bound"] = str(upper)
                result["interval"] = f"({str(lower)}, {str(upper)})"

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "probability_distributions":
            operation = arguments["operation"]
            dist_type = arguments.get("distribution_type")
            parameters = arguments.get("parameters", {})
            store_key = arguments.get("store_key")

            result = {"operation": operation}

            if operation == "create":
                # Create a new distribution
                if dist_type == "normal":
                    mu = sp.sympify(parameters.get("mu", "0"))
                    sigma = sp.sympify(parameters.get("sigma", "1"))
                    dist = Normal('X', mu, sigma)
                elif dist_type == "binomial":
                    n = int(parameters.get("n", 10))
                    p = sp.sympify(parameters.get("p", "0.5"))
                    dist = Binomial('X', n, p)
                elif dist_type == "poisson":
                    lam = sp.sympify(parameters.get("lambda", "1"))
                    dist = Poisson('X', lam)
                elif dist_type == "exponential":
                    rate = sp.sympify(parameters.get("rate", "1"))
                    dist = Exponential('X', rate)
                elif dist_type == "uniform":
                    a = sp.sympify(parameters.get("a", "0"))
                    b = sp.sympify(parameters.get("b", "1"))
                    dist = Uniform('X', a, b)
                else:
                    return [TextContent(type="text", text=json.dumps({"error": f"Unknown distribution type: {dist_type}"}, indent=2))]

                result["distribution_type"] = dist_type
                result["parameters"] = parameters
                result["mean"] = str(E(dist))
                result["variance"] = str(variance(dist))

                if store_key:
                    # Store distribution info (simplified - full implementation would need state management)
                    result["stored_as"] = store_key

            elif operation == "sum":
                # For sum of independent RVs
                result["note"] = "Sum of distributions: E[X+Y] = E[X] + E[Y], Var[X+Y] = Var[X] + Var[Y] (if independent)"

            elif operation == "product":
                result["note"] = "Product of distributions requires joint distribution analysis"

            elif operation == "transform":
                transformation = arguments.get("transformation", "X")
                result["transformation"] = transformation
                result["note"] = "For Y = g(X), use transformation of variables theorem"

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # ================================================================
        # CATEGORY H: Transform Theory
        # ================================================================

        elif name == "laplace_transform":
            expression_str = arguments["expression"]
            is_inverse = arguments.get("inverse", False)
            store_key = arguments.get("store_key")

            t = symbols('t', real=True, positive=True)
            s = symbols('s')
            expr = sp.sympify(expression_str)

            if is_inverse:
                result_expr = inverse_laplace_transform(expr, s, t)
                transform_type = "Inverse Laplace"
            else:
                result_expr = laplace_transform(expr, t, s)
                if isinstance(result_expr, tuple):
                    result_expr = result_expr[0]  # Get main result
                transform_type = "Laplace"

            result = {
                "transform_type": transform_type,
                "input": expression_str,
                "output": str(result_expr),
                "latex": latex(result_expr),
                "result": str(result_expr),
                "status": "success"
            }

            if store_key:
                ai.expressions[store_key] = result_expr
                result["stored_as"] = store_key

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "fourier_transform":
            expression_str = arguments["expression"]
            is_inverse = arguments.get("inverse", False)
            store_key = arguments.get("store_key")

            # Respect variable names from arguments
            var_name = arguments.get("variable", "t")
            transform_var_name = arguments.get("transform_variable", "omega")

            var = symbols(var_name, real=True)
            transform_var = symbols(transform_var_name, real=True)
            # Sympify with locals to ensure symbol matching
            expr = sp.sympify(expression_str, locals={var_name: var})

            if is_inverse:
                result_expr = inverse_fourier_transform(expr, transform_var, var)
                transform_type = "Inverse Fourier"
            else:
                result_expr = fourier_transform(expr, var, transform_var)
                if isinstance(result_expr, tuple):
                    result_expr = result_expr[0]
                transform_type = "Fourier"

            result = {
                "transform_type": transform_type,
                "input": expression_str,
                "output": str(result_expr),
                "latex": latex(result_expr),
                "result": str(result_expr),
                "status": "success"
            }

            if store_key:
                ai.expressions[store_key] = result_expr
                result["stored_as"] = store_key

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "z_transform":
            sequence_str = arguments["sequence"]
            is_inverse = arguments.get("inverse", False)

            n = symbols('n', integer=True)
            z = symbols('z')
            expr = sp.sympify(sequence_str)

            # Z-transform: X(z) = sum(x[n] * z^(-n), n=0..inf)
            if not is_inverse:
                # Forward Z-transform
                result_expr = Sum(expr * z**(-n), (n, 0, oo))
                transform_type = "Z-Transform"
            else:
                # Inverse Z-transform (simplified)
                result_expr = expr  # Placeholder - full inverse Z needs residue theorem
                transform_type = "Inverse Z-Transform (symbolic representation)"

            result = {
                "transform_type": transform_type,
                "input": sequence_str,
                "output": str(result_expr),
                "latex": latex(result_expr),
                "result": str(result_expr),
                "status": "success"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "convolution":
            # Support both 'f'/'g' and 'function1'/'function2' parameter names
            func1_str = arguments.get("function1") or arguments.get("f")
            func2_str = arguments.get("function2") or arguments.get("g")
            conv_type = arguments.get("convolution_type", "continuous")

            t = symbols('t', real=True)
            tau = symbols('tau', real=True)

            func1 = sp.sympify(func1_str)
            func2 = sp.sympify(func2_str)

            if conv_type == "continuous":
                # (f * g)(t) = integral(f(tau) * g(t-tau), tau, -oo, oo)
                integrand = func1.subs(t, tau) * func2.subs(t, t - tau)
                result_expr = Integral(integrand, (tau, -oo, oo))
                conv_formula = "∫ f(τ)g(t-τ) dτ from -∞ to ∞"
            elif conv_type == "discrete":
                n = symbols('n', integer=True)
                k = symbols('k', integer=True)
                result_expr = Sum(func1.subs(t, k) * func2.subs(t, n-k), (k, -oo, oo))
                conv_formula = "Σ f[k]g[n-k] from k=-∞ to ∞"
            else:
                result_expr = "Circular convolution - specialized implementation"
                conv_formula = "Circular convolution in frequency domain"

            result = {
                "convolution_type": conv_type,
                "function1": func1_str,
                "function2": func2_str,
                "result": str(result_expr),
                "formula": conv_formula,
                "status": "success"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "transfer_function_analysis":
            var_name = arguments.get("variable", "s")
            var = symbols(var_name)

            # Support both transfer_function (string) and numerator/denominator
            if "transfer_function" in arguments:
                tf_str = arguments["transfer_function"]
                tf_expr = sp.sympify(tf_str)
                # Extract numerator and denominator from fraction
                numer, denom = sp.fraction(tf_expr)
            else:
                numerator_str = arguments["numerator"]
                denominator_str = arguments["denominator"]
                numer = sp.sympify(numerator_str)
                denom = sp.sympify(denominator_str)

            # Support both 'analyze' and 'analysis' parameter names
            analysis_types = arguments.get("analysis") or arguments.get("analyze", [])

            transfer_function = numer / denom

            result = {
                "transfer_function": str(transfer_function),
                "transfer_function_latex": latex(transfer_function)
            }

            for analysis_type in analysis_types:
                if analysis_type == "poles":
                    poles = solve(denom, var)
                    result["poles"] = [str(p) for p in poles]
                elif analysis_type == "zeros":
                    zeros = solve(numer, var)
                    result["zeros"] = [str(z) for z in zeros]
                elif analysis_type == "stability":
                    poles = solve(denom, var)
                    # For continuous systems (s-domain), stable if all poles have negative real part
                    # For discrete systems (z-domain), stable if all poles have magnitude < 1
                    result["stability"] = {
                        "poles": [str(p) for p in poles],
                        "criterion": "Continuous: Re(poles) < 0; Discrete: |poles| < 1"
                    }

            result["status"] = "success"
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "mellin_transform":
            expression_str = arguments["expression"]
            is_inverse = arguments.get("inverse", False)
            var_name = arguments.get("variable", "x")
            transform_var_name = arguments.get("transform_variable", "s")

            var = symbols(var_name, positive=True, real=True)
            transform_var = symbols(transform_var_name)
            # Sympify with locals to ensure symbol matching
            expr = sp.sympify(expression_str, locals={var_name: var, transform_var_name: transform_var})

            if is_inverse:
                # inverse_mellin_transform requires a 'strip' parameter (convergence strip)
                result_expr = inverse_mellin_transform(expr, var, transform_var, (0, oo))
                transform_type = "Inverse Mellin"
            else:
                result_expr = mellin_transform(expr, var, transform_var)
                if isinstance(result_expr, tuple):
                    result_expr = result_expr[0]
                transform_type = "Mellin"

            result = {
                "transform_type": transform_type,
                "input": expression_str,
                "output": str(result_expr),
                "latex": latex(result_expr),
                "result": str(result_expr),
                "status": "success"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # ================================================================
        # CATEGORY I: Optimization Extensions
        # ================================================================

        elif name == "lagrange_multipliers":
            objective_str = arguments["objective"]
            constraints_strs = arguments["constraints"]
            var_names = arguments["variables"]

            # Create symbols
            vars_syms = symbols(' '.join(var_names))
            if not isinstance(vars_syms, tuple):
                vars_syms = (vars_syms,)

            # Parse objective and constraints
            objective = sp.sympify(objective_str)
            constraints = [sp.sympify(c) for c in constraints_strs]

            # Create Lagrange multipliers
            lambdas = symbols(f'lambda_0:{len(constraints)}')
            if not isinstance(lambdas, tuple):
                lambdas = (lambdas,)

            # Build Lagrangian: L = f - sum(lambda_i * g_i)
            lagrangian = objective - sum(lam * g for lam, g in zip(lambdas, constraints))

            # Compute gradients
            grad_equations = []
            for var in vars_syms:
                grad_equations.append(diff(lagrangian, var))
            for lam, constraint in zip(lambdas, constraints):
                grad_equations.append(constraint)

            # Try to solve for critical points
            critical_points = []
            try:
                all_vars = list(vars_syms) + list(lambdas)
                solutions = solve(grad_equations, all_vars, dict=True)
                if solutions:
                    critical_points = [
                        {str(k): str(v) for k, v in sol.items() if k in vars_syms}
                        for sol in solutions
                    ]
            except Exception:
                critical_points = []

            result = {
                "status": "success",
                "objective": objective_str,
                "constraints": constraints_strs,
                "variables": var_names,
                "lagrangian": str(lagrangian),
                "lagrangian_latex": latex(lagrangian),
                "gradient_equations": [str(eq) for eq in grad_equations],
                "num_equations": len(grad_equations),
                "critical_points": critical_points if critical_points else "Unable to solve symbolically"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "linear_programming":
            objective_str = arguments["objective"]
            constraints_strs = arguments["constraints"]
            var_names = arguments["variables"]
            maximize = arguments.get("maximize", False)

            vars_syms = symbols(' '.join(var_names))
            if not isinstance(vars_syms, tuple):
                vars_syms = (vars_syms,)

            objective = sp.sympify(objective_str)
            constraints = [sp.sympify(c) for c in constraints_strs]

            result = {
                "status": "success",
                "objective": objective_str,
                "constraints": constraints_strs,
                "variables": var_names,
                "optimization_type": "maximize" if maximize else "minimize",
                "standard_form": {
                    "objective": str(objective if not maximize else -objective),
                    "note": "Convert to standard form: min c^T x subject to Ax <= b, x >= 0"
                },
                "solution_method": "Symbolic LP - use simplex or interior point for numerical solution",
                "solution": "Linear programming requires numerical methods (simplex/interior point) for general solution",
                "symbolic_solution": "LP problems are typically solved numerically"
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "convex_optimization":
            objective_str = arguments["objective"]
            var_names = arguments["variables"]

            # Support both parameter formats
            operation = arguments.get("operation")
            check_convexity = arguments.get("check_convexity", False)

            # If check_convexity is True, set operation to verify_convex
            if check_convexity and not operation:
                operation = "verify_convex"

            vars_syms = symbols(' '.join(var_names))
            if not isinstance(vars_syms, tuple):
                vars_syms = (vars_syms,)

            objective = sp.sympify(objective_str)

            result = {
                "status": "success",
                "objective": objective_str,
                "variables": var_names
            }
            if operation:
                result["operation"] = operation

            if operation == "hessian":
                # Compute Hessian matrix
                hess = hessian(objective, vars_syms)
                result["hessian"] = str(hess)
                result["hessian_latex"] = latex(hess)

            elif operation == "verify_convex":
                # A function is convex if its Hessian is positive semidefinite
                hess = hessian(objective, vars_syms)
                result["hessian"] = str(hess)

                # Try to verify convexity for simple cases
                is_convex = None
                try:
                    # For quadratic forms, check if all diagonal elements are non-negative
                    # and the matrix is symmetric
                    hess_matrix = sp.Matrix(hess)
                    # Check if constant (all elements are numbers)
                    if all(h.is_number for h in hess_matrix):
                        eigenvals = hess_matrix.eigenvals()
                        # Check if all eigenvalues are non-negative
                        is_convex = all(sp.simplify(ev) >= 0 for ev in eigenvals.keys())
                    else:
                        # For simple quadratic forms like x^2, check second derivative
                        if len(vars_syms) == 1:
                            second_deriv = diff(objective, vars_syms[0], 2)
                            if second_deriv.is_number:
                                is_convex = second_deriv >= 0
                            else:
                                is_convex = None
                except:
                    is_convex = None

                if is_convex is not None:
                    result["is_convex"] = is_convex
                else:
                    result["is_convex"] = "Unable to determine symbolically"

                result["note"] = "Function is convex if Hessian is positive semidefinite (eigenvalues >= 0)"
                result["verification_method"] = "Check eigenvalues of Hessian"

            elif operation == "verify_concave":
                hess = hessian(objective, vars_syms)
                result["hessian"] = str(hess)
                result["is_concave"] = "Check if all eigenvalues of Hessian are <= 0"
                result["note"] = "Function is concave if Hessian is negative semidefinite (eigenvalues <= 0)"

            elif operation == "find_minimum":
                # Find critical points by setting gradient to zero
                grad = [diff(objective, var) for var in vars_syms]
                result["gradient"] = [str(g) for g in grad]
                result["critical_point_equations"] = [f"{str(g)} = 0" for g in grad]

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "calculus_of_variations":
            functional_str = arguments["functional"]
            # Support both parameter formats: dependent_var and function_name
            dependent_var_name = arguments.get("dependent_var", arguments.get("function_name", "y"))
            independent_var_name = arguments.get("independent_var", arguments.get("independent_variable", "x"))

            x = symbols(independent_var_name)

            # Try to parse the functional first to see if it already contains function definitions
            try:
                # If functional already contains Derivative expressions, parse directly
                functional = sp.sympify(functional_str)
                # Extract the function from the functional if possible
                y = Function(dependent_var_name)(x)
                y_prime = diff(y, x)
            except:
                # Create function symbols
                y = Function(dependent_var_name)(x)
                y_prime = diff(y, x)

                # Parse functional L(x, y, y')
                functional = sp.sympify(functional_str, locals={
                    dependent_var_name: y,
                    f"{dependent_var_name}'": y_prime,
                    independent_var_name: x
                })

            # Euler-Lagrange equation: d/dx(dL/dy') - dL/dy = 0
            dL_dy = diff(functional, y)
            dL_dy_prime = diff(functional, y_prime)
            d_dx_dL_dy_prime = diff(dL_dy_prime, x)

            euler_lagrange = d_dx_dL_dy_prime - dL_dy

            result = {
                "status": "success",
                "functional": functional_str,
                "dependent_variable": dependent_var_name,
                "independent_variable": independent_var_name,
                "euler_lagrange_equation": str(euler_lagrange) + " = 0",
                "euler_lagrange": str(euler_lagrange) + " = 0",  # Alias for compatibility
                "euler_lagrange_latex": latex(euler_lagrange) + " = 0",
                "dL_dy": str(dL_dy),
                "dL_dy_prime": str(dL_dy_prime)
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "dynamic_programming":
            value_func_str = arguments["value_function"]
            state_vars = arguments["state_variables"]
            control_vars = arguments.get("control_variables", [])
            horizon = arguments.get("horizon", "finite")

            result = {
                "status": "success",
                "value_function": value_func_str,
                "state_variables": state_vars,
                "control_variables": control_vars,
                "horizon": horizon,
                "bellman_equation": ""
            }

            if horizon == "finite":
                # V_t(s) = max_a [r(s,a) + γ V_{t+1}(s')]
                result["bellman_equation"] = "V_t(s) = max_a [r(s,a) + γ V_{t+1}(s')]"
                result["note"] = "Finite horizon: backward induction from terminal state"
            else:
                # V(s) = max_a [r(s,a) + γ V(s')]
                result["bellman_equation"] = "V(s) = max_a [r(s,a) + γ V(s')]"
                result["note"] = "Infinite horizon: solve fixed-point equation or use value/policy iteration"

            result["solution_methods"] = ["Value Iteration", "Policy Iteration", "Linear Programming"]
            result["optimal_policy"] = "π*(s) = arg max_a [r(s,a) + γ V(s')]"

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown advanced tool: {name}"}, indent=2)
            )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": f"Tool execution failed: {str(e)}",
                "tool": name
            }, indent=2)
        )]
