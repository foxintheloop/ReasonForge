"""
ReasonForge Expressions MCP Server

Essential symbolic expression manipulation and basic calculus tools.
Provides 15 fundamental tools for working with symbolic expressions.
"""

from typing import Dict, Any, List

import sympy as sp
from sympy import symbols, diff, integrate, limit, series, simplify, factor, expand, latex

from reasonforge import (
    BaseReasonForgeServer,
    SymbolicAI,
    create_input_schema,
    safe_sympify,
    validate_variable_name,
    validate_expression_string,
    validate_list_input,
    ValidationError,
)


class ExpressionsServer(BaseReasonForgeServer):
    """MCP server for expression manipulation and basic calculus."""

    def __init__(self):
        super().__init__("reasonforge-expressions")
        self.ai = SymbolicAI()

    def register_tools(self):
        """Register all 15 expression manipulation tools."""

        # Variable Management (4 tools)
        self.add_tool(
            name="intro",
            description="Introduce a variable with specified assumptions (real, positive, etc.). Stores variable for later use.",
            handler=self.handle_intro,
            input_schema=create_input_schema(
                properties={
                    "name": {
                        "type": "string",
                        "description": "Variable name (e.g., 'x', 'theta', 'alpha')"
                    },
                    "positive_assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Assumptions to set as True: real, positive, negative, integer, etc."
                    }
                },
                required=["name"]
            )
        )

        self.add_tool(
            name="intro_many",
            description="Introduce multiple variables with the same assumptions simultaneously.",
            handler=self.handle_intro_many,
            input_schema=create_input_schema(
                properties={
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of variable names (e.g., ['x', 'y', 'z'])"
                    },
                    "positive_assumptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Assumptions to set as True"
                    }
                },
                required=["names"]
            )
        )

        self.add_tool(
            name="introduce_expression",
            description="Store an expression with a key for later reference.",
            handler=self.handle_introduce_expression,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to store (e.g., 'x**2 + y**2')"
                    },
                    "key": {
                        "type": "string",
                        "description": "Optional key name. If not provided, auto-generates expr_0, expr_1, etc."
                    }
                },
                required=["expression"]
            )
        )

        self.add_tool(
            name="introduce_function",
            description="Define a function symbol for use in differential equations.",
            handler=self.handle_introduce_function,
            input_schema=create_input_schema(
                properties={
                    "name": {
                        "type": "string",
                        "description": "Function name (e.g., 'f', 'g', 'y')"
                    }
                },
                required=["name"]
            )
        )

        # Expression Operations (5 tools)
        self.add_tool(
            name="simplify_expression",
            description="Simplify an expression using various methods.",
            handler=self.handle_simplify_expression,
            input_schema=create_input_schema(
                properties={
                    "expression_key": {
                        "type": "string",
                        "description": "Key of stored expression or expression string"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["simplify", "trigsimp", "ratsimp"],
                        "description": "Simplification method"
                    }
                },
                required=["expression_key"]
            )
        )

        self.add_tool(
            name="substitute_expression",
            description="Substitute values into an expression.",
            handler=self.handle_substitute_expression,
            input_schema=create_input_schema(
                properties={
                    "expression_key": {
                        "type": "string",
                        "description": "Key of stored expression or expression string"
                    },
                    "substitutions": {
                        "type": "object",
                        "description": "Dictionary mapping variable names to values"
                    }
                },
                required=["expression_key", "substitutions"]
            )
        )

        self.add_tool(
            name="factor_expression",
            description="Factor an expression.",
            handler=self.handle_factor_expression,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression to factor"
                    }
                },
                required=["expression"]
            )
        )

        self.add_tool(
            name="expand_expression",
            description="Expand an expression.",
            handler=self.handle_expand_expression,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression to expand"
                    }
                },
                required=["expression"]
            )
        )

        self.add_tool(
            name="evaluate_expression",
            description="Evaluate an expression numerically.",
            handler=self.handle_evaluate_expression,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression to evaluate"
                    },
                    "substitutions": {
                        "type": "object",
                        "description": "Variable values for evaluation"
                    }
                },
                required=["expression", "substitutions"]
            )
        )

        # Alias for backwards compatibility
        self.add_tool(
            name="substitute_values",
            description="Substitute values into an expression and evaluate.",
            handler=self.handle_evaluate_expression,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression to evaluate"
                    },
                    "substitutions": {
                        "type": "object",
                        "description": "Variable values"
                    }
                },
                required=["expression", "substitutions"]
            )
        )

        # Calculus (4 tools)
        self.add_tool(
            name="differentiate",
            description="Compute the derivative of an expression.",
            handler=self.handle_differentiate,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression to differentiate"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to differentiate with respect to"
                    },
                    "order": {
                        "type": "integer",
                        "description": "Order of derivative (default: 1)",
                        "default": 1
                    }
                },
                required=["expression", "variable"]
            )
        )

        self.add_tool(
            name="integrate",
            description="Compute the integral of an expression.",
            handler=self.handle_integrate,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression to integrate"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to integrate with respect to"
                    }
                },
                required=["expression", "variable"]
            )
        )

        self.add_tool(
            name="compute_limit",
            description="Compute the limit of an expression as a variable approaches a value.",
            handler=self.handle_compute_limit,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable"
                    },
                    "point": {
                        "type": "string",
                        "description": "Point to approach (use 'inf' for infinity, 'zero' for 0)"
                    }
                },
                required=["expression", "variable", "point"]
            )
        )

        self.add_tool(
            name="expand_series",
            description="Compute Taylor/Maclaurin series expansion.",
            handler=self.handle_expand_series,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Expression to expand"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable for expansion"
                    },
                    "point": {
                        "type": "number",
                        "description": "Point around which to expand (default: 0)",
                        "default": 0
                    },
                    "order": {
                        "type": "integer",
                        "description": "Number of terms (default: 10)",
                        "default": 10
                    }
                },
                required=["expression", "variable"]
            )
        )

        # Utilities (2 tools)
        self.add_tool(
            name="print_latex_expression",
            description="Get the LaTeX representation of a stored expression.",
            handler=self.handle_print_latex_expression,
            input_schema=create_input_schema(
                properties={
                    "key": {
                        "type": "string",
                        "description": "Key of stored expression"
                    }
                },
                required=["key"]
            )
        )

        self.add_tool(
            name="solve_word_problem",
            description="Solve a word problem by setting up and solving equations.",
            handler=self.handle_solve_word_problem,
            input_schema=create_input_schema(
                properties={
                    "problem": {
                        "type": "string",
                        "description": "Problem description"
                    },
                    "equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of equations"
                    },
                    "unknowns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of unknown variables"
                    }
                },
                required=["problem", "equations", "unknowns"]
            )
        )

    # Variable Management Handlers

    def handle_intro(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle intro tool - introduce a variable with assumptions."""
        var_name = validate_variable_name(arguments["name"])
        pos_assumptions = arguments.get("positive_assumptions", [])

        # Create assumptions dict
        assumptions = {assumption: True for assumption in pos_assumptions}

        # Create variable with assumptions
        var = symbols(var_name, **assumptions)
        self.ai.variables[var_name] = var

        return {
            "name": var_name,
            "variable": str(var),
            "assumptions": assumptions,
            "latex": latex(var)
        }

    def handle_intro_many(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle intro_many tool - introduce multiple variables."""
        var_names = validate_list_input(arguments["names"])
        pos_assumptions = arguments.get("positive_assumptions", [])

        # Validate all variable names
        var_names = [validate_variable_name(name) for name in var_names]

        assumptions = {assumption: True for assumption in pos_assumptions}

        variables = []
        for var_name in var_names:
            var = symbols(var_name, **assumptions)
            self.ai.variables[var_name] = var
            variables.append(str(var))

        return {
            "names": var_names,
            "variables": variables,
            "assumptions": assumptions,
            "count": len(variables)
        }

    def handle_introduce_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle introduce_expression tool - store an expression."""
        expr_str = validate_expression_string(arguments["expression"])
        key = arguments.get("key")

        if key is None:
            key = self.ai._get_next_key("expr")

        expr = safe_sympify(expr_str)
        self.ai.expressions[key] = expr

        return {
            "expression": str(expr),
            "key": key,
            "stored": True,
            "latex": latex(expr)
        }

    def handle_introduce_function(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle introduce_function tool - define a function symbol."""
        func_name = validate_variable_name(arguments["name"])
        func = sp.Function(func_name)
        self.ai.functions[func_name] = func

        return {
            "name": func_name,
            "function": str(func),
            "type": "undefined_function",
            "usage": f"{func_name}(x)"
        }

    # Expression Operations Handlers

    def handle_simplify_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simplify_expression tool."""
        expr_key = arguments["expression_key"]
        method = arguments.get("method", "simplify")

        # Get expression (from storage or parse)
        if expr_key in self.ai.expressions:
            expr = self.ai.expressions[expr_key]
        else:
            expr = safe_sympify(expr_key)

        # Apply simplification
        if method == "trigsimp":
            result_expr = sp.trigsimp(expr)
        elif method == "ratsimp":
            result_expr = sp.ratsimp(expr)
        else:
            result_expr = simplify(expr)

        return {
            "expression_key": expr_key,
            "method": method,
            "original": str(expr),
            "simplified": str(result_expr),
            "latex": latex(result_expr)
        }

    def handle_substitute_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle substitute_expression tool."""
        expr_key = arguments["expression_key"]
        substitutions = arguments["substitutions"]

        # Get expression
        if expr_key in self.ai.expressions:
            expr = self.ai.expressions[expr_key]
        else:
            expr = safe_sympify(expr_key)

        # Parse substitutions
        subs_parsed = {symbols(validate_variable_name(k)): safe_sympify(v)
                      for k, v in substitutions.items()}

        # Substitute
        result_expr = expr.subs(subs_parsed)

        # Store result
        result_key = self.ai._get_next_key("expr")
        self.ai.expressions[result_key] = result_expr

        return {
            "expression_key": expr_key,
            "substitutions": {k: str(v) for k, v in substitutions.items()},
            "original": str(expr),
            "result": str(result_expr),
            "stored_key": result_key,  # For test compatibility
            "latex": latex(result_expr)
        }

    def handle_factor_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle factor_expression tool."""
        expr_str = validate_expression_string(arguments["expression"])
        expr = safe_sympify(expr_str)

        factored = factor(expr)

        return {
            "expression": str(expr),
            "factored": str(factored),
            "latex": latex(factored)
        }

    def handle_expand_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle expand_expression tool."""
        expr_str = validate_expression_string(arguments["expression"])
        expr = safe_sympify(expr_str)

        expanded = expand(expr)

        return {
            "expression": str(expr),
            "expanded": str(expanded),
            "latex": latex(expanded)
        }

    def handle_evaluate_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evaluate_expression tool."""
        expr_str = validate_expression_string(arguments["expression"])
        substitutions = arguments["substitutions"]

        expr = safe_sympify(expr_str)

        # Parse substitutions
        subs_parsed = {symbols(validate_variable_name(k)): safe_sympify(v)
                      for k, v in substitutions.items()}

        # Evaluate
        result = expr.subs(subs_parsed)

        # Try to get numerical value
        try:
            numerical_value = float(result.evalf())
        except (ValueError, TypeError, AttributeError):
            # Result may not be evaluatable to a float (e.g., symbolic expression)
            numerical_value = None

        return {
            "expression": str(expr),
            "substitutions": {k: str(v) for k, v in substitutions.items()},
            "result": str(result),
            "numerical_value": numerical_value,
            "latex": latex(result)
        }

    # Calculus Handlers

    def handle_differentiate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle differentiate tool."""
        expr_str = validate_expression_string(arguments["expression"])
        var_name = validate_variable_name(arguments["variable"])
        order = arguments.get("order", 1)

        expr = safe_sympify(expr_str)
        var = symbols(var_name)

        derivative = diff(expr, var, order)

        return {
            "expression": str(expr),
            "variable": var_name,
            "order": order,
            "derivative": str(derivative),
            "result": str(derivative),  # For test compatibility
            "latex": latex(derivative)
        }

    def handle_integrate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle integrate tool."""
        expr_str = validate_expression_string(arguments["expression"])
        var_name = validate_variable_name(arguments["variable"])

        expr = safe_sympify(expr_str)
        var = symbols(var_name)

        integral = integrate(expr, var)

        return {
            "expression": str(expr),
            "variable": var_name,
            "integral": str(integral),
            "result": str(integral),  # For test compatibility
            "latex": latex(integral)
        }

    def handle_compute_limit(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compute_limit tool."""
        expr_str = validate_expression_string(arguments["expression"])
        var_name = validate_variable_name(arguments["variable"])
        point_str = arguments["point"]

        expr = safe_sympify(expr_str)
        var = symbols(var_name)

        # Parse limit point
        if point_str.lower() in ["inf", "infinity", "oo"]:
            point = sp.oo
        elif point_str.lower() in ["zero", "0"]:
            point = 0
        elif point_str.lower() == "-inf":
            point = -sp.oo
        else:
            point = safe_sympify(point_str)

        limit_result = limit(expr, var, point)

        return {
            "expression": str(expr),
            "variable": var_name,
            "point": str(point),
            "limit": str(limit_result),
            "result": str(limit_result),  # For test compatibility
            "latex": latex(limit_result)
        }

    def handle_expand_series(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle expand_series tool."""
        expr_str = validate_expression_string(arguments["expression"])
        var_name = validate_variable_name(arguments["variable"])
        point = arguments.get("point", 0)
        order = arguments.get("order", 10)

        expr = safe_sympify(expr_str)
        var = symbols(var_name)

        series_expansion = series(expr, var, point, order)

        return {
            "expression": str(expr),
            "variable": var_name,
            "point": point,
            "order": order,
            "series": str(series_expansion),
            "result": str(series_expansion),  # For test compatibility
            "latex": latex(series_expansion)
        }

    # Utility Handlers

    def handle_print_latex_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle print_latex_expression tool."""
        key = arguments["key"]

        if key not in self.ai.expressions:
            return {"error": f"Expression key '{key}' not found"}

        expr = self.ai.expressions[key]
        latex_str = latex(expr)

        return {
            "key": key,
            "expression": str(expr),
            "latex": latex_str,
            "mathjax": f"$${latex_str}$$"  # For test compatibility
        }

    def handle_solve_word_problem(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solve_word_problem tool."""
        problem = arguments["problem"]
        equations = validate_list_input(arguments["equations"])
        unknowns = validate_list_input(arguments["unknowns"])

        # Validate unknowns
        unknowns = [validate_variable_name(u) for u in unknowns]

        # Parse equations
        equations_parsed = [safe_sympify(eq) for eq in equations]

        # Parse unknowns
        unknowns_parsed = [symbols(u) for u in unknowns]

        # Solve
        solution = sp.solve(equations_parsed, unknowns_parsed)

        # Format solution
        solution_dict = {str(k): str(v) for k, v in solution.items()} if isinstance(solution, dict) else str(solution)

        # Create interpretation
        if isinstance(solution, dict):
            interpretation = ", ".join([f"{k} = {v}" for k, v in solution_dict.items()])
        else:
            interpretation = f"Solution: {solution_dict}"

        return {
            "problem": problem,
            "equations": [str(eq) for eq in equations_parsed],
            "unknowns": unknowns,
            "solutions": solution_dict,  # For test compatibility (plural)
            "interpretation": interpretation,  # For test compatibility
            "latex_solution": latex(solution)
        }


# Entry point
server = ExpressionsServer()

if __name__ == "__main__":
    server.run()
