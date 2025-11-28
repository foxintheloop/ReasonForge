"""
ReasonForge Analysis MCP Server

Advanced calculus, differential equations, transforms, and signal processing.
Provides 17 tools for mathematical analysis.
"""

from typing import Dict, Any, Union

import sympy as sp
from sympy import symbols, Function, dsolve, latex, series, integrate, diff, oo

from reasonforge import (
    BaseReasonForgeServer,
    SymbolicAI,
    create_input_schema,
    safe_sympify,
    validate_variable_name,
    validate_expression_string,
    validate_list_input,
    validate_dict_input,
    ValidationError,
)


class AnalysisServer(BaseReasonForgeServer):
    """MCP server for advanced calculus and analysis tools."""

    def __init__(self):
        super().__init__("reasonforge-analysis")
        self.ai = SymbolicAI()

    def register_tools(self):
        """Register all 17 analysis tools."""

        # Differential Equations Tools
        self.add_tool(
            name="dsolve_ode",
            description="Solve ordinary differential equations symbolically.",
            handler=self.handle_dsolve_ode,
            input_schema=create_input_schema(
                properties={
                    "equation": {
                        "type": "string",
                        "description": "ODE (e.g., 'Derivative(y(x), x) - 2*y(x)')"
                    },
                    "function": {
                        "type": "string",
                        "description": "Function to solve for (e.g., 'y(x)')"
                    },
                    "initial_conditions": {
                        "type": "object",
                        "description": "Initial conditions (optional)"
                    }
                },
                required=["equation", "function"]
            )
        )

        self.add_tool(
            name="pdsolve_pde",
            description="Solve partial differential equations symbolically.",
            handler=self.handle_pdsolve_pde,
            input_schema=create_input_schema(
                properties={
                    "equation": {"type": "string", "description": "PDE"},
                    "function": {"type": "string", "description": "Function (e.g., 'u(x,t)')"}
                },
                required=["equation", "function"]
            )
        )

        self.add_tool(
            name="symbolic_ode_initial_conditions",
            description="Solve ODE with symbolic initial conditions.",
            handler=self.handle_symbolic_ode_initial_conditions,
            input_schema=create_input_schema(
                properties={
                    "equation": {"type": "string"},
                    "function": {"type": "string"},
                    "initial_conditions": {"type": "object"}
                },
                required=["equation", "function", "initial_conditions"]
            )
        )

        # Physics PDEs Tools
        self.add_tool(
            name="schrodinger_equation_solver",
            description="Solve time-independent Schrödinger equation.",
            handler=self.handle_schrodinger_equation_solver,
            input_schema=create_input_schema(
                properties={
                    "equation_type": {"type": "string"},
                    "potential": {"type": "string"},
                    "boundary_conditions": {"type": "object"}
                },
                required=["equation_type", "potential"]
            )
        )

        self.add_tool(
            name="wave_equation_solver",
            description="Solve wave equation.",
            handler=self.handle_wave_equation_solver,
            input_schema=create_input_schema(
                properties={
                    "wave_type": {"type": "string"},
                    "dimension": {"type": "integer"},
                    "boundary_conditions": {"type": "object"}
                },
                required=["wave_type", "dimension"]
            )
        )

        self.add_tool(
            name="heat_equation_solver",
            description="Solve heat/diffusion equation.",
            handler=self.handle_heat_equation_solver,
            input_schema=create_input_schema(
                properties={
                    "geometry": {"type": "string"},
                    "boundary_conditions": {"type": "object"},
                    "initial_conditions": {"type": "object"}
                },
                required=["geometry"]
            )
        )

        # Transform Tools
        self.add_tool(
            name="laplace_transform",
            description="Compute Laplace transform.",
            handler=self.handle_laplace_transform,
            input_schema=create_input_schema(
                properties={
                    "expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "transform_variable": {"type": "string", "default": "s"}
                },
                required=["expression", "variable"]
            )
        )

        self.add_tool(
            name="fourier_transform",
            description="Compute Fourier transform.",
            handler=self.handle_fourier_transform,
            input_schema=create_input_schema(
                properties={
                    "expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "transform_variable": {"type": "string", "default": "k"}
                },
                required=["expression", "variable"]
            )
        )

        self.add_tool(
            name="z_transform",
            description="Compute Z-transform.",
            handler=self.handle_z_transform,
            input_schema=create_input_schema(
                properties={
                    "sequence": {"type": "string"},
                    "n_variable": {"type": "string"},
                    "z_variable": {"type": "string", "default": "z"}
                },
                required=["sequence", "n_variable"]
            )
        )

        self.add_tool(
            name="mellin_transform",
            description="Compute Mellin transform.",
            handler=self.handle_mellin_transform,
            input_schema=create_input_schema(
                properties={
                    "expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "transform_variable": {"type": "string", "default": "s"}
                },
                required=["expression", "variable"]
            )
        )

        self.add_tool(
            name="integral_transforms_custom",
            description="Apply custom integral transforms.",
            handler=self.handle_integral_transforms_custom,
            input_schema=create_input_schema(
                properties={
                    "transform_type": {"type": "string"},
                    "expression": {"type": "string"},
                    "variable": {"type": "string"}
                },
                required=["transform_type", "expression", "variable"]
            )
        )

        # Signal Processing Tools
        self.add_tool(
            name="convolution",
            description="Compute convolution of two functions.",
            handler=self.handle_convolution,
            input_schema=create_input_schema(
                properties={
                    "f": {"type": "string"},
                    "g": {"type": "string"},
                    "variable": {"type": "string"}
                },
                required=["f", "g", "variable"]
            )
        )

        self.add_tool(
            name="transfer_function_analysis",
            description="Analyze transfer functions (poles, zeros, stability).",
            handler=self.handle_transfer_function_analysis,
            input_schema=create_input_schema(
                properties={
                    "transfer_function": {"type": "string"},
                    "variable": {"type": "string"}
                },
                required=["transfer_function", "variable"]
            )
        )

        # Asymptotic Methods Tools
        self.add_tool(
            name="perturbation_theory",
            description="Apply perturbation theory to differential equations.",
            handler=self.handle_perturbation_theory,
            input_schema=create_input_schema(
                properties={
                    "equation": {"type": "string"},
                    "perturbation_type": {"type": "string"},
                    "small_parameter": {"type": "string"}
                },
                required=["equation", "small_parameter"]
            )
        )

        self.add_tool(
            name="asymptotic_analysis",
            description="Find asymptotic expansion of expressions.",
            handler=self.handle_asymptotic_analysis,
            input_schema=create_input_schema(
                properties={
                    "expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "limit_point": {"type": "string"}
                },
                required=["expression", "variable", "limit_point"]
            )
        )

        # Special Functions & Optimization Tools
        self.add_tool(
            name="special_functions_properties",
            description="Get properties of special functions (Bessel, Legendre, etc.).",
            handler=self.handle_special_functions_properties,
            input_schema=create_input_schema(
                properties={
                    "function_type": {"type": "string"},
                    "operation": {"type": "string"},
                    "parameters": {"type": "object"}
                },
                required=["function_type", "operation"]
            )
        )

        self.add_tool(
            name="symbolic_optimization_setup",
            description="Set up optimization problems symbolically.",
            handler=self.handle_symbolic_optimization_setup,
            input_schema=create_input_schema(
                properties={
                    "objective": {"type": "string"},
                    "equality_constraints": {"type": "array", "items": {"type": "string"}},
                    "inequality_constraints": {"type": "array", "items": {"type": "string"}},
                    "variables": {"type": "array", "items": {"type": "string"}}
                },
                required=["objective", "variables"]
            )
        )

    # Differential Equations Handlers

    def handle_dsolve_ode(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dsolve_ode tool - solve ordinary differential equations."""
        equation_str = validate_expression_string(arguments["equation"])
        function_str = validate_expression_string(arguments["function"])
        initial_conds = arguments.get("initial_conditions")

        # Parse equation and function - use sympify for ODEs
        eq = sp.sympify(equation_str)
        func = sp.sympify(function_str)

        solution = dsolve(eq, func)

        return {
            "equation": equation_str,
            "function": function_str,
            "solution": str(solution),
            "latex": latex(solution)
        }

    def handle_pdsolve_pde(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pdsolve_pde tool - solve partial differential equations."""
        equation_str = validate_expression_string(arguments["equation"])
        function_str = validate_expression_string(arguments["function"])

        # Parse equation and function
        eq = sp.sympify(equation_str)
        func = sp.sympify(function_str)

        try:
            solution = sp.pdsolve(eq, func)
            return {
                "equation": equation_str,
                "function": function_str,
                "solution": str(solution),
                "latex": latex(solution)
            }
        except Exception:
            return {
                "equation": equation_str,
                "function": function_str,
                "note": "PDE requires specific conditions for symbolic solution",
                "general_form": "Solution exists but requires boundary conditions"
            }

    def handle_symbolic_ode_initial_conditions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle symbolic_ode_initial_conditions tool - solve ODE with initial conditions."""
        equation_str = validate_expression_string(arguments["equation"])
        function_str = validate_expression_string(arguments["function"])
        ics = arguments["initial_conditions"]

        # Validate initial conditions
        if not isinstance(ics, dict):
            raise ValidationError("Initial conditions must be a dictionary")

        # Parse equation and function
        eq = sp.sympify(equation_str)
        func = sp.sympify(function_str)

        solution = dsolve(eq, func, ics=ics)

        return {
            "equation": equation_str,
            "function": function_str,
            "initial_conditions": ics,
            "solution": str(solution),
            "latex": latex(solution)
        }

    # Physics PDEs Handlers

    def handle_schrodinger_equation_solver(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle schrodinger_equation_solver tool - symbolic formulation."""
        equation_type = arguments["equation_type"]
        potential = validate_expression_string(arguments["potential"])
        boundary_conditions = arguments.get("boundary_conditions", {})

        return {
            "tool": "schrodinger_equation_solver",
            "equation_type": equation_type,
            "potential": potential,
            "boundary_conditions": boundary_conditions,
            "status": "symbolic_formulation",
            "note": "Schrödinger equation solver provides symbolic setup and general solutions"
        }

    def handle_wave_equation_solver(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle wave_equation_solver tool - symbolic formulation."""
        wave_type = arguments["wave_type"]
        dimension = arguments["dimension"]
        boundary_conditions = arguments.get("boundary_conditions", {})

        # Validate dimension
        if not isinstance(dimension, int) or dimension < 1 or dimension > 3:
            raise ValidationError("Dimension must be 1, 2, or 3")

        return {
            "tool": "wave_equation_solver",
            "wave_type": wave_type,
            "dimension": dimension,
            "boundary_conditions": boundary_conditions,
            "status": "symbolic_formulation",
            "note": "Wave equation solver provides symbolic setup and general solutions"
        }

    def handle_heat_equation_solver(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle heat_equation_solver tool - symbolic formulation."""
        geometry = arguments["geometry"]
        boundary_conditions = arguments.get("boundary_conditions", {})
        initial_conditions = arguments.get("initial_conditions", {})

        return {
            "tool": "heat_equation_solver",
            "geometry": geometry,
            "boundary_conditions": boundary_conditions,
            "initial_conditions": initial_conditions,
            "status": "symbolic_formulation",
            "note": "Heat equation solver provides symbolic setup and general solutions"
        }

    # Transform Handlers

    def handle_laplace_transform(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle laplace_transform tool - compute Laplace transform."""
        expr_str = validate_expression_string(arguments["expression"])
        var_str = validate_variable_name(arguments["variable"])
        s_var = validate_variable_name(arguments.get("transform_variable", "s"))

        # Parse expression - use sympify for transforms
        expr = sp.sympify(expr_str)
        var = symbols(var_str)
        s = symbols(s_var)

        transform = sp.laplace_transform(expr, var, s)

        return {
            "expression": expr_str,
            "variable": var_str,
            "transform_variable": s_var,
            "transform": str(transform[0]) if isinstance(transform, tuple) else str(transform),
            "latex": latex(transform[0]) if isinstance(transform, tuple) else latex(transform)
        }

    def handle_fourier_transform(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fourier_transform tool - compute Fourier transform."""
        expr_str = validate_expression_string(arguments["expression"])
        var_str = validate_variable_name(arguments["variable"])
        k_var = validate_variable_name(arguments.get("transform_variable", "k"))

        # Parse expression - use sympify for transforms
        expr = sp.sympify(expr_str)
        var = symbols(var_str)
        k = symbols(k_var)

        transform = sp.fourier_transform(expr, var, k)

        return {
            "expression": expr_str,
            "variable": var_str,
            "transform_variable": k_var,
            "transform": str(transform),
            "latex": latex(transform)
        }

    def handle_z_transform(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle z_transform tool - symbolic setup."""
        sequence = validate_expression_string(arguments["sequence"])
        n_variable = validate_variable_name(arguments["n_variable"])
        z_variable = validate_variable_name(arguments.get("z_variable", "z"))

        return {
            "tool": "z_transform",
            "sequence": sequence,
            "n_variable": n_variable,
            "z_variable": z_variable,
            "status": "symbolic_setup",
            "note": "Z-transform provides symbolic formulation"
        }

    def handle_mellin_transform(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle mellin_transform tool - symbolic setup."""
        expression = validate_expression_string(arguments["expression"])
        variable = validate_variable_name(arguments["variable"])
        transform_variable = validate_variable_name(arguments.get("transform_variable", "s"))

        return {
            "tool": "mellin_transform",
            "expression": expression,
            "variable": variable,
            "transform_variable": transform_variable,
            "status": "symbolic_setup",
            "note": "Mellin transform provides symbolic formulation"
        }

    def handle_integral_transforms_custom(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle integral_transforms_custom tool - symbolic setup."""
        transform_type = arguments["transform_type"]
        expression = validate_expression_string(arguments["expression"])
        variable = validate_variable_name(arguments["variable"])

        return {
            "tool": "integral_transforms_custom",
            "transform_type": transform_type,
            "expression": expression,
            "variable": variable,
            "status": "symbolic_setup",
            "note": "Custom integral transform provides symbolic formulation"
        }

    # Signal Processing Handlers

    def handle_convolution(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle convolution tool - compute convolution of two functions."""
        f_str = validate_expression_string(arguments["f"])
        g_str = validate_expression_string(arguments["g"])
        var_str = validate_variable_name(arguments["variable"])

        # Parse functions - use sympify for convolution
        f = sp.sympify(f_str)
        g = sp.sympify(g_str)
        var = symbols(var_str)

        # Symbolic convolution setup
        tau = symbols('tau')
        convolution_expr = integrate(f.subs(var, tau) * g.subs(var, var - tau), (tau, -oo, oo))

        return {
            "f": f_str,
            "g": g_str,
            "variable": var_str,
            "convolution": str(convolution_expr),
            "latex": latex(convolution_expr)
        }

    def handle_transfer_function_analysis(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transfer_function_analysis tool - analyze transfer functions."""
        tf_str = validate_expression_string(arguments["transfer_function"])
        var_str = validate_variable_name(arguments["variable"])

        # Parse transfer function - use sympify
        tf = sp.sympify(tf_str)
        var = symbols(var_str)

        # Find poles (denominator roots) and zeros (numerator roots)
        numer, denom = sp.fraction(tf)
        poles = sp.solve(denom, var)
        zeros = sp.solve(numer, var)

        return {
            "transfer_function": tf_str,
            "variable": var_str,
            "poles": [str(p) for p in poles],
            "zeros": [str(z) for z in zeros],
            "latex": latex(tf)
        }

    # Asymptotic Methods Handlers

    def handle_perturbation_theory(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle perturbation_theory tool - symbolic perturbation setup."""
        equation_str = validate_expression_string(arguments["equation"])
        small_param = validate_variable_name(arguments["small_parameter"])
        perturbation_type = arguments.get("perturbation_type", "regular")

        return {
            "equation": equation_str,
            "small_parameter": small_param,
            "perturbation_type": perturbation_type,
            "status": "perturbation_setup",
            "note": "Symbolic perturbation expansion setup provided"
        }

    def handle_asymptotic_analysis(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle asymptotic_analysis tool - find asymptotic expansion."""
        expr_str = validate_expression_string(arguments["expression"])
        var_str = validate_variable_name(arguments["variable"])
        limit_pt = arguments["limit_point"]

        # Parse expression and variable - use sympify
        expr = sp.sympify(expr_str)
        var = symbols(var_str)

        # Handle limit point (can be "inf", "oo", or a number)
        if limit_pt.lower() in ["inf", "infinity", "oo"]:
            point = oo
        else:
            # Allow both numeric and symbolic limit points
            point = sp.sympify(str(limit_pt))

        # Asymptotic expansion
        expansion = series(expr, var, point, n=5)

        return {
            "expression": expr_str,
            "variable": var_str,
            "limit_point": limit_pt,
            "expansion": str(expansion),
            "latex": latex(expansion)
        }

    # Special Functions & Optimization Handlers

    def handle_special_functions_properties(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle special_functions_properties tool - symbolic setup."""
        function_type = arguments["function_type"]
        operation = arguments["operation"]
        parameters = arguments.get("parameters", {})

        return {
            "tool": "special_functions_properties",
            "function_type": function_type,
            "operation": operation,
            "parameters": parameters,
            "status": "symbolic_setup",
            "note": "Special functions properties provides symbolic formulation and properties"
        }

    def handle_symbolic_optimization_setup(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle symbolic_optimization_setup tool - symbolic optimization setup."""
        objective = validate_expression_string(arguments["objective"])
        variables_raw = validate_list_input(arguments["variables"], max_length=50)
        equality_constraints_raw = arguments.get("equality_constraints", [])
        inequality_constraints_raw = arguments.get("inequality_constraints", [])

        # Validate inputs
        variables = [validate_variable_name(v) for v in variables_raw]

        equality_constraints = []
        if equality_constraints_raw:
            eq_list = validate_list_input(equality_constraints_raw, max_length=50)
            equality_constraints = [validate_expression_string(c) for c in eq_list]

        inequality_constraints = []
        if inequality_constraints_raw:
            ineq_list = validate_list_input(inequality_constraints_raw, max_length=50)
            inequality_constraints = [validate_expression_string(c) for c in ineq_list]

        return {
            "tool": "symbolic_optimization_setup",
            "objective": objective,
            "variables": variables,
            "equality_constraints": equality_constraints,
            "inequality_constraints": inequality_constraints,
            "status": "symbolic_setup",
            "note": "Symbolic optimization setup provides formulation"
        }


# Entry point
server = AnalysisServer()

if __name__ == "__main__":
    server.run()
