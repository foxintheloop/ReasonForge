"""
Visualization Tools for ReasonForge MCP Server

This module provides 6 visualization tools for plotting symbolic functions,
generating LaTeX representations, and creating ASCII/text-based visualizations.
"""

import json
from typing import Any
from mcp.types import Tool, TextContent
import sympy as sp
from sympy import symbols, Symbol, lambdify, latex
from sympy.plotting import plot, plot3d
from sympy.vector import CoordSys3D
import numpy as np


def get_visualization_tool_definitions() -> list[Tool]:
    """Return list of visualization tool definitions."""
    return [
        Tool(
            name="plot_symbolic_function",
            description="Generate LaTeX and ASCII representations of symbolic function plots. Create visual representations for 1D and 2D functions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Symbolic expression to plot (e.g., 'sin(x)', 'x**2 + 2*x + 1')"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Independent variable (default: 'x')"
                    },
                    "range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        },
                        "description": "Plot range (default: -10 to 10)"
                    },
                    "output_format": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["latex", "ascii", "properties", "critical_points"],
                        "description": "Output formats to generate"
                    }
                },
                "required": ["expression"]
            }
        ),

        Tool(
            name="contour_plot_symbolic",
            description="Generate contour plot representations for 2D functions f(x,y). Create level curves and analyze function topology.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "2D function (e.g., 'x**2 + y**2', 'sin(x)*cos(y)')"
                    },
                    "x_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        }
                    },
                    "y_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        }
                    },
                    "level_curves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific level curve values (e.g., ['0', '1', '4'])"
                    },
                    "compute": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to compute: 'gradient', 'critical_points', 'level_equations'"
                    }
                },
                "required": ["expression"]
            }
        ),

        Tool(
            name="vector_field_plot",
            description="Visualize symbolic vector fields F(x,y) or F(x,y,z). Generate field line equations and flow descriptions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "field_components": {
                        "type": "object",
                        "description": "Vector field components (e.g., {'x': 'y', 'y': '-x'} for rotation)"
                    },
                    "dimension": {
                        "type": "integer",
                        "enum": [2, 3],
                        "description": "2D or 3D vector field"
                    },
                    "analyze": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Analysis: 'divergence', 'curl', 'critical_points', 'flow_lines', 'jacobian'"
                    }
                },
                "required": ["field_components"]
            }
        ),

        Tool(
            name="phase_portrait",
            description="Generate phase portraits for dynamical systems dx/dt = f(x,y), dy/dt = g(x,y). Analyze equilibria and stability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "system": {
                        "type": "object",
                        "properties": {
                            "dx_dt": {"type": "string"},
                            "dy_dt": {"type": "string"}
                        },
                        "description": "System of differential equations"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "State variables (default: ['x', 'y'])"
                    },
                    "analyze": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Analysis: 'equilibria', 'stability', 'nullclines', 'jacobian', 'eigenvectors'"
                    }
                },
                "required": ["system"]
            }
        ),

        Tool(
            name="bifurcation_diagram",
            description="Analyze bifurcations in parametric dynamical systems. Find bifurcation points and classify bifurcation types.",
            inputSchema={
                "type": "object",
                "properties": {
                    "system": {
                        "type": "string",
                        "description": "Dynamical system with parameter (e.g., 'r*x - x**3')"
                    },
                    "state_variable": {
                        "type": "string",
                        "description": "State variable (default: 'x')"
                    },
                    "parameter": {
                        "type": "string",
                        "description": "Bifurcation parameter (e.g., 'r')"
                    },
                    "analyze": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Analysis: 'equilibria', 'bifurcation_points', 'stability_change', 'classification'"
                    }
                },
                "required": ["system", "parameter"]
            }
        ),

        Tool(
            name="3d_surface_plot",
            description="Generate 3D surface representations for functions f(x,y). Analyze surface properties, curvature, and critical points.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "3D surface function z = f(x,y)"
                    },
                    "x_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        }
                    },
                    "y_range": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "string"},
                            "end": {"type": "string"}
                        }
                    },
                    "compute": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to compute: 'gradient', 'hessian', 'curvature', 'critical_points', 'saddle_points'"
                    }
                },
                "required": ["expression"]
            }
        )
    ]


async def handle_visualization_tool(name: str, arguments: dict[str, Any], ai) -> list[TextContent]:
    """Handle visualization tool calls."""

    if name == "plot_symbolic_function":
        return await _plot_symbolic_function(arguments, ai)
    elif name == "contour_plot_symbolic":
        return await _contour_plot_symbolic(arguments, ai)
    elif name == "vector_field_plot":
        return await _vector_field_plot(arguments, ai)
    elif name == "phase_portrait":
        return await _phase_portrait(arguments, ai)
    elif name == "bifurcation_diagram":
        return await _bifurcation_diagram(arguments, ai)
    elif name == "3d_surface_plot":
        return await _3d_surface_plot(arguments, ai)
    else:
        raise ValueError(f"Unknown visualization tool: {name}")


# Implementation functions

async def _plot_symbolic_function(args: dict, ai) -> list[TextContent]:
    """Generate function plot representations."""
    expr_str = args["expression"]
    var_str = args.get("variable", "x")
    output_formats = args.get("output_format", ["latex", "properties"])

    # Parse expression
    var = sp.Symbol(var_str)
    expr = sp.sympify(expr_str)

    result = {
        "expression": expr_str,
        "variable": var_str,
        "output_formats": output_formats
    }

    if "latex" in output_formats:
        result["latex"] = sp.latex(expr)
        result["latex_full"] = f"y = {sp.latex(expr)}"

    if "properties" in output_formats:
        # Analyze function properties
        props = {}

        # Domain
        try:
            domain = sp.calculus.util.continuous_domain(expr, var, sp.S.Reals)
            props["domain"] = str(domain)
        except:
            props["domain"] = "Real numbers (check for specific restrictions)"

        # Derivative
        derivative = sp.diff(expr, var)
        props["derivative"] = str(derivative)
        props["derivative_latex"] = sp.latex(derivative)

        # Second derivative
        second_deriv = sp.diff(derivative, var)
        props["second_derivative"] = str(second_deriv)

        result["properties"] = props

    if "critical_points" in output_formats:
        # Find critical points
        derivative = sp.diff(expr, var)
        critical_pts = sp.solve(derivative, var)

        critical_info = {
            "derivative": str(derivative),
            "critical_points": [str(pt) for pt in critical_pts]
        }

        # Classify critical points using second derivative test
        second_deriv = sp.diff(derivative, var)
        classifications = []

        for pt in critical_pts:
            try:
                pt_value = complex(pt)
                if not np.isreal(pt_value):
                    continue
                pt = sp.N(pt)
            except:
                pass

            second_at_pt = second_deriv.subs(var, pt)

            classification = {
                "point": str(pt),
                "value": str(expr.subs(var, pt))
            }

            try:
                if second_at_pt > 0:
                    classification["type"] = "Local minimum"
                elif second_at_pt < 0:
                    classification["type"] = "Local maximum"
                else:
                    classification["type"] = "Inconclusive (second derivative test fails)"
            except:
                classification["type"] = "Requires further analysis"

            classifications.append(classification)

        critical_info["classifications"] = classifications
        result["critical_points_analysis"] = critical_info
        result["critical_points"] = critical_info  # Alias for compatibility

    if "ascii" in output_formats:
        # Generate simple ASCII representation
        result["ascii_note"] = "Simple character-based representation"
        result["ascii_example"] = _generate_ascii_plot(expr, var)

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


def _generate_ascii_plot(expr, var, width=60, height=20):
    """Generate simple ASCII plot."""
    try:
        # Evaluate function at points
        x_vals = np.linspace(-10, 10, width)
        f = lambdify(var, expr, "numpy")
        y_vals = f(x_vals)

        # Check for invalid values
        if not np.all(np.isfinite(y_vals)):
            return "Function has discontinuities or undefined values in range"

        # Normalize to plot area
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)

        if y_min == y_max:
            return "Constant function"

        # Create plot grid
        plot_lines = []
        for row in range(height):
            threshold = y_min + (y_max - y_min) * (height - row - 1) / height
            line = ""
            for i, y in enumerate(y_vals):
                if abs(y - threshold) < (y_max - y_min) / height:
                    line += "*"
                else:
                    line += " "
            plot_lines.append(line)

        return "\n".join(plot_lines)

    except Exception as e:
        return f"ASCII plot generation requires numerical evaluation: {str(e)}"


async def _contour_plot_symbolic(args: dict, ai) -> list[TextContent]:
    """Generate contour plot representation."""
    expr_str = args["expression"]
    level_curves = args.get("level_curves", [])
    compute = args.get("compute", ["gradient"])

    # Parse expression
    x, y = sp.symbols('x y')
    expr = sp.sympify(expr_str)

    result = {
        "expression": expr_str,
        "latex": sp.latex(expr),
        "function": f"f(x,y) = {expr_str}"
    }

    if "gradient" in compute:
        grad_x = sp.diff(expr, x)
        grad_y = sp.diff(expr, y)

        result["gradient"] = {
            "∇f": f"({grad_x}, {grad_y})",
            "∂f/∂x": str(grad_x),
            "∂f/∂y": str(grad_y),
            "gradient_latex": f"\\nabla f = ({sp.latex(grad_x)}, {sp.latex(grad_y)})"
        }

    if "level_equations" in compute:
        level_eqs = {}
        for level in level_curves:
            level_val = sp.sympify(level)
            equation = sp.Eq(expr, level_val)
            level_eqs[f"level_{level}"] = {
                "equation": str(equation),
                "latex": sp.latex(equation),
                "implicit_form": f"{expr} = {level}"
            }
        result["level_curves"] = level_eqs

    if "critical_points" in compute:
        grad_x = sp.diff(expr, x)
        grad_y = sp.diff(expr, y)

        # Solve ∇f = 0
        critical_pts = sp.solve([grad_x, grad_y], [x, y])

        result["critical_points"] = {
            "system": ["∂f/∂x = 0", "∂f/∂y = 0"],
            "points": [{"x": str(pt[0]), "y": str(pt[1])} for pt in critical_pts] if isinstance(critical_pts, list) else str(critical_pts)
        }

        # Hessian for classification
        f_xx = sp.diff(grad_x, x)
        f_yy = sp.diff(grad_y, y)
        f_xy = sp.diff(grad_x, y)

        result["hessian_matrix"] = {
            "H": f"[[{f_xx}, {f_xy}], [{f_xy}, {f_yy}]]",
            "determinant": str(f_xx * f_yy - f_xy**2),
            "classification_method": "D = f_xx·f_yy - (f_xy)² at critical points"
        }

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _vector_field_plot(args: dict, ai) -> list[TextContent]:
    """Visualize vector field."""
    components = args["field_components"]
    dimension = args.get("dimension", 2)
    analyze = args.get("analyze", ["divergence", "curl"])

    result = {
        "dimension": dimension,
        "field_components": components
    }

    if dimension == 2:
        x, y = sp.symbols('x y')
        F_x = sp.sympify(components.get("x", "0"))
        F_y = sp.sympify(components.get("y", "0"))

        result["field"] = f"F(x,y) = ({F_x}, {F_y})"
        result["field_latex"] = f"\\mathbf{{F}}(x,y) = ({sp.latex(F_x)}, {sp.latex(F_y)})"

        if "divergence" in analyze:
            div = sp.diff(F_x, x) + sp.diff(F_y, y)
            result["divergence"] = {
                "∇·F": str(div),
                "formula": "∂Fₓ/∂x + ∂Fᵧ/∂y",
                "divergence_latex": sp.latex(div)
            }

            if div == 0:
                result["divergence"]["property"] = "Incompressible field (∇·F = 0)"

        if "curl" in analyze:
            curl_z = sp.diff(F_y, x) - sp.diff(F_x, y)
            result["curl"] = {
                "(∇×F)_z": str(curl_z),
                "formula": "∂Fᵧ/∂x - ∂Fₓ/∂y",
                "curl_latex": sp.latex(curl_z)
            }

            if curl_z == 0:
                result["curl"]["property"] = "Irrotational field (∇×F = 0) - Conservative"

        if "critical_points" in analyze:
            # Points where F = 0
            critical_pts = sp.solve([F_x, F_y], [x, y])
            result["critical_points"] = {
                "definition": "Points where F(x,y) = 0",
                "points": str(critical_pts)
            }

        if "jacobian" in analyze:
            J = sp.Matrix([[sp.diff(F_x, x), sp.diff(F_x, y)],
                          [sp.diff(F_y, x), sp.diff(F_y, y)]])
            result["jacobian"] = {
                "matrix": str(J),
                "latex": sp.latex(J),
                "trace": str(J.trace()),
                "determinant": str(J.det())
            }

    elif dimension == 3:
        x, y, z = sp.symbols('x y z')
        F_x = sp.sympify(components.get("x", "0"))
        F_y = sp.sympify(components.get("y", "0"))
        F_z = sp.sympify(components.get("z", "0"))

        result["field"] = f"F(x,y,z) = ({F_x}, {F_y}, {F_z})"

        if "divergence" in analyze:
            div = sp.diff(F_x, x) + sp.diff(F_y, y) + sp.diff(F_z, z)
            result["divergence"] = {
                "∇·F": str(div),
                "formula": "∂Fₓ/∂x + ∂Fᵧ/∂y + ∂Fz/∂z"
            }

        if "curl" in analyze:
            curl_x = sp.diff(F_z, y) - sp.diff(F_y, z)
            curl_y = sp.diff(F_x, z) - sp.diff(F_z, x)
            curl_z = sp.diff(F_y, x) - sp.diff(F_x, y)

            result["curl"] = {
                "∇×F": f"({curl_x}, {curl_y}, {curl_z})",
                "components": {
                    "x": str(curl_x),
                    "y": str(curl_y),
                    "z": str(curl_z)
                }
            }

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _phase_portrait(args: dict, ai) -> list[TextContent]:
    """Generate phase portrait analysis."""
    system = args["system"]
    variables = args.get("variables", ["x", "y"])
    analyze = args.get("analyze", ["equilibria", "stability"])

    # Parse system
    x, y = sp.symbols(variables[0] + ' ' + variables[1])
    dx_dt = sp.sympify(system["dx_dt"])
    dy_dt = sp.sympify(system["dy_dt"])

    result = {
        "system": {
            f"d{variables[0]}/dt": str(dx_dt),
            f"d{variables[1]}/dt": str(dy_dt)
        },
        "variables": variables
    }

    if "equilibria" in analyze:
        # Find equilibrium points where dx/dt = dy/dt = 0
        equilibria = sp.solve([dx_dt, dy_dt], [x, y])

        result["equilibria"] = {
            "definition": "Points where dx/dt = dy/dt = 0",
            "points": str(equilibria)
        }

    if "nullclines" in analyze:
        result["nullclines"] = {
            f"{variables[0]}-nullcline": f"{dx_dt} = 0",
            f"{variables[1]}-nullcline": f"{dy_dt} = 0",
            "interpretation": "Curves where derivatives are zero"
        }

    if "jacobian" in analyze or "stability" in analyze:
        # Compute Jacobian matrix
        J = sp.Matrix([[sp.diff(dx_dt, x), sp.diff(dx_dt, y)],
                      [sp.diff(dy_dt, x), sp.diff(dy_dt, y)]])

        result["jacobian"] = {
            "matrix": str(J),
            "latex": sp.latex(J),
            "∂f/∂x": str(sp.diff(dx_dt, x)),
            "∂f/∂y": str(sp.diff(dx_dt, y)),
            "∂g/∂x": str(sp.diff(dy_dt, x)),
            "∂g/∂y": str(sp.diff(dy_dt, y))
        }

        if "stability" in analyze:
            trace = J.trace()
            det = J.det()

            result["stability_analysis"] = {
                "trace": str(trace),
                "determinant": str(det),
                "characteristic_polynomial": str(J.charpoly(sp.Symbol('λ'))),
                "classification": {
                    "method": "Linearization at equilibria",
                    "trace_determinant_criteria": {
                        "det < 0": "Saddle point (unstable)",
                        "det > 0, trace < 0": "Stable node or spiral",
                        "det > 0, trace > 0": "Unstable node or spiral",
                        "det > 0, trace = 0": "Center (neutrally stable)"
                    }
                }
            }

        if "eigenvectors" in analyze:
            eigenvals = J.eigenvals()
            eigenvects = J.eigenvects()

            result["eigenanalysis"] = {
                "eigenvalues": str(eigenvals),
                "eigenvectors": str(eigenvects),
                "interpretation": "Eigenvectors show directions of principal axes"
            }

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _bifurcation_diagram(args: dict, ai) -> list[TextContent]:
    """Analyze bifurcations."""
    system_str = args["system"]
    param_str = args["parameter"]
    state_var = args.get("state_variable", "x")
    analyze = args.get("analyze", ["equilibria", "bifurcation_points"])

    # Parse
    x = sp.Symbol(state_var)
    r = sp.Symbol(param_str)

    # Handle reserved keywords like 'lambda' by temporarily replacing them
    temp_system_str = system_str
    if param_str == 'lambda':
        # Replace 'lambda' with temporary symbol to avoid Python keyword issues
        temp_system_str = system_str.replace('lambda', '_lambda_temp')
        temp_r = sp.Symbol('_lambda_temp')
        f = sp.sympify(temp_system_str)
        # Substitute back to the actual symbol
        f = f.subs(temp_r, r)
    else:
        locals_dict = {state_var: x, param_str: r}
        f = sp.sympify(system_str, locals=locals_dict)

    result = {
        "system": f"d{state_var}/dt = {system_str}",
        "parameter": param_str,
        "state_variable": state_var
    }

    if "equilibria" in analyze:
        # Find equilibria: f(x, r) = 0
        equilibria = sp.solve(f, x)

        result["equilibria"] = {
            "equation": f"{system_str} = 0",
            "solutions": [str(eq) for eq in equilibria],
            "parametric_dependence": f"Equilibria as functions of {param_str}"
        }

    if "bifurcation_points" in analyze:
        # Bifurcations occur when ∂f/∂x = 0 at equilibrium
        df_dx = sp.diff(f, x)

        result["bifurcation_condition"] = {
            "stability_change": "∂f/∂x = 0",
            "derivative": str(df_dx),
            "note": "Solve simultaneously: f = 0 and ∂f/∂x = 0"
        }

        # Try to find bifurcation points
        try:
            bif_pts = sp.solve([f, df_dx], [x, r])
            result["bifurcation_points"] = str(bif_pts)
        except:
            result["bifurcation_points"] = "Requires numerical methods"

    if "stability_change" in analyze:
        df_dx = sp.diff(f, x)

        result["stability_criterion"] = {
            "stable": "∂f/∂x < 0 at equilibrium",
            "unstable": "∂f/∂x > 0 at equilibrium",
            "derivative": str(df_dx)
        }

    if "classification" in analyze:
        result["bifurcation_types"] = {
            "saddle_node": "Two equilibria collide and annihilate",
            "transcritical": "Equilibria exchange stability",
            "pitchfork": "One equilibrium becomes three (or vice versa)",
            "hopf": "Equilibrium changes stability and limit cycle appears"
        }

        # Common example
        if "r" in system_str and "x**2" in system_str:
            result["likely_type"] = "Saddle-node bifurcation (r - x²)"
        elif "r" in system_str and "x**3" in system_str:
            result["likely_type"] = "Pitchfork bifurcation (rx - x³)"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _3d_surface_plot(args: dict, ai) -> list[TextContent]:
    """Generate 3D surface representation."""
    expr_str = args["expression"]
    compute = args.get("compute", ["gradient", "critical_points"])

    # Parse
    x, y = sp.symbols('x y')
    f = sp.sympify(expr_str)

    result = {
        "expression": expr_str,
        "latex": sp.latex(f),
        "surface": f"z = {expr_str}"
    }

    if "gradient" in compute:
        grad_x = sp.diff(f, x)
        grad_y = sp.diff(f, y)

        result["gradient"] = {
            "∇f": f"({grad_x}, {grad_y})",
            "∂f/∂x": str(grad_x),
            "∂f/∂y": str(grad_y),
            "gradient_latex": f"\\nabla f = \\left({sp.latex(grad_x)}, {sp.latex(grad_y)}\\right)",
            "normal_vector": f"(-∂f/∂x, -∂f/∂y, 1) = ({-grad_x}, {-grad_y}, 1)"
        }

    if "hessian" in compute:
        f_xx = sp.diff(f, x, x)
        f_yy = sp.diff(f, y, y)
        f_xy = sp.diff(f, x, y)

        H = sp.Matrix([[f_xx, f_xy], [f_xy, f_yy]])

        result["hessian"] = {
            "matrix": str(H),
            "latex": sp.latex(H),
            "f_xx": str(f_xx),
            "f_yy": str(f_yy),
            "f_xy": str(f_xy),
            "determinant": str(H.det())
        }

    if "critical_points" in compute or "saddle_points" in compute:
        grad_x = sp.diff(f, x)
        grad_y = sp.diff(f, y)

        # Solve ∇f = 0
        critical_pts = sp.solve([grad_x, grad_y], [x, y])

        critical_info = {
            "system": ["∂f/∂x = 0", "∂f/∂y = 0"],
            "points": str(critical_pts)
        }

        # Classify using Hessian
        if critical_pts:
            f_xx = sp.diff(grad_x, x)
            f_yy = sp.diff(grad_y, y)
            f_xy = sp.diff(grad_x, y)

            D = f_xx * f_yy - f_xy**2  # Hessian determinant

            critical_info["second_derivative_test"] = {
                "discriminant": "D = f_xx·f_yy - (f_xy)²",
                "D_formula": str(D),
                "classification": {
                    "D > 0, f_xx > 0": "Local minimum",
                    "D > 0, f_xx < 0": "Local maximum",
                    "D < 0": "Saddle point",
                    "D = 0": "Test inconclusive"
                }
            }

        result["critical_points"] = critical_info

    if "curvature" in compute:
        result["curvature_info"] = {
            "gaussian_curvature": "K = (f_xx·f_yy - f_xy²) / (1 + f_x² + f_y²)²",
            "mean_curvature": "H = (1 + f_y²)f_xx - 2f_x·f_y·f_xy + (1 + f_x²)f_yy / [2(1 + f_x² + f_y²)^(3/2)]",
            "principal_curvatures": "Eigenvalues of shape operator"
        }

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]
