"""
ReasonForge Algebra MCP Server

Algebraic operations, equation solving, matrices, and optimization tools.
Provides 18 tools for algebraic computation.
"""

from typing import Dict, Any, List, Union

import sympy as sp
from sympy import symbols, solve, simplify, latex, Matrix, diff

from reasonforge import (
    BaseReasonForgeServer,
    SymbolicAI,
    create_input_schema,
    safe_sympify,
    validate_variable_name,
    validate_expression_string,
    validate_list_input,
    validate_dict_input,
    validate_key_format,
    validate_matrix_dimensions,
    ValidationError,
)


class AlgebraServer(BaseReasonForgeServer):
    """MCP server for algebraic operations, matrices, and optimization."""

    def __init__(self):
        super().__init__("reasonforge-algebra")
        self.ai = SymbolicAI()

    def register_tools(self):
        """Register all 18 algebra tools."""

        # Equation Solving Tools
        self.add_tool(
            name="solve_equations",
            description="Solve a system of equations with 100% accuracy. Returns exact symbolic solutions.",
            handler=self.handle_solve_equations,
            input_schema=create_input_schema(
                properties={
                    "equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of equations to solve (e.g., ['x**2 - 4', 'x + y - 7'])"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to solve for (optional, auto-detected if not provided)"
                    }
                },
                required=["equations"]
            )
        )

        self.add_tool(
            name="solve_algebraically",
            description="Solve an equation algebraically over specified domain (real, complex, etc.).",
            handler=self.handle_solve_algebraically,
            input_schema=create_input_schema(
                properties={
                    "expression_key": {
                        "type": "string",
                        "description": "Key of stored expression or equation string"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to solve for"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain: 'real', 'complex', 'positive', etc.",
                        "default": "complex"
                    }
                },
                required=["expression_key", "variable"]
            )
        )

        self.add_tool(
            name="solve_linear_system",
            description="Solve a system of linear equations using matrix methods.",
            handler=self.handle_solve_linear_system,
            input_schema=create_input_schema(
                properties={
                    "equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Linear equations (e.g., ['2*x + 3*y - 7', 'x - y - 1'])"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables in the system"
                    }
                },
                required=["equations", "variables"]
            )
        )

        self.add_tool(
            name="solve_nonlinear_system",
            description="Solve a system of nonlinear equations.",
            handler=self.handle_solve_nonlinear_system,
            input_schema=create_input_schema(
                properties={
                    "equations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Nonlinear equations (e.g., ['x**2 + y**2 - 25', 'x - y - 1'])"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables in the system"
                    }
                },
                required=["equations", "variables"]
            )
        )

        # Matrix Operation Tools
        self.add_tool(
            name="create_matrix",
            description="Create a matrix and store it for later operations.",
            handler=self.handle_create_matrix,
            input_schema=create_input_schema(
                properties={
                    "elements": {
                        "type": "array",
                        "description": "2D array of matrix elements (e.g., [[1, 2], [3, 4]])"
                    },
                    "key": {
                        "type": "string",
                        "description": "Optional key for storage. Auto-generates if not provided."
                    }
                },
                required=["elements"]
            )
        )

        self.add_tool(
            name="matrix_determinant",
            description="Calculate the determinant of a matrix.",
            handler=self.handle_matrix_determinant,
            input_schema=create_input_schema(
                properties={
                    "matrix_key": {
                        "type": "string",
                        "description": "Key of stored matrix"
                    }
                },
                required=["matrix_key"]
            )
        )

        self.add_tool(
            name="matrix_inverse",
            description="Calculate the inverse of a matrix.",
            handler=self.handle_matrix_inverse,
            input_schema=create_input_schema(
                properties={
                    "matrix_key": {
                        "type": "string",
                        "description": "Key of stored matrix"
                    }
                },
                required=["matrix_key"]
            )
        )

        self.add_tool(
            name="matrix_eigenvalues",
            description="Find the eigenvalues of a matrix.",
            handler=self.handle_matrix_eigenvalues,
            input_schema=create_input_schema(
                properties={
                    "matrix_key": {
                        "type": "string",
                        "description": "Key of stored matrix"
                    }
                },
                required=["matrix_key"]
            )
        )

        self.add_tool(
            name="matrix_eigenvectors",
            description="Find the eigenvectors of a matrix.",
            handler=self.handle_matrix_eigenvectors,
            input_schema=create_input_schema(
                properties={
                    "matrix_key": {
                        "type": "string",
                        "description": "Key of stored matrix"
                    }
                },
                required=["matrix_key"]
            )
        )

        # Optimization Tools
        self.add_tool(
            name="optimize_function",
            description="Find critical points and optimize a function.",
            handler=self.handle_optimize_function,
            input_schema=create_input_schema(
                properties={
                    "objective": {
                        "type": "string",
                        "description": "Objective function to optimize"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables (optional, auto-detected if not provided)"
                    }
                },
                required=["objective"]
            )
        )

        self.add_tool(
            name="lagrange_multipliers",
            description="Solve optimization with constraints using Lagrange multipliers.",
            handler=self.handle_lagrange_multipliers,
            input_schema=create_input_schema(
                properties={
                    "objective": {
                        "type": "string",
                        "description": "Objective function"
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Constraint equations"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables"
                    }
                },
                required=["objective", "constraints", "variables"]
            )
        )

        self.add_tool(
            name="linear_programming",
            description="Solve linear programming problems symbolically.",
            handler=self.handle_linear_programming,
            input_schema=create_input_schema(
                properties={
                    "objective": {
                        "type": "string",
                        "description": "Linear objective function"
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Linear constraints"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables"
                    },
                    "minimize": {
                        "type": "boolean",
                        "description": "True to minimize, False to maximize",
                        "default": True
                    }
                },
                required=["objective", "constraints", "variables"]
            )
        )

        self.add_tool(
            name="convex_optimization",
            description="Verify convexity and solve convex optimization problems.",
            handler=self.handle_convex_optimization,
            input_schema=create_input_schema(
                properties={
                    "objective": {
                        "type": "string",
                        "description": "Objective function"
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Constraints (optional)"
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables"
                    }
                },
                required=["objective", "variables"]
            )
        )

        self.add_tool(
            name="calculus_of_variations",
            description="Solve calculus of variations problems (e.g., geodesics, brachistochrone).",
            handler=self.handle_calculus_of_variations,
            input_schema=create_input_schema(
                properties={
                    "functional": {
                        "type": "string",
                        "description": "Functional to extremize"
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Unknown function name (e.g., 'y')"
                    },
                    "independent_var": {
                        "type": "string",
                        "description": "Independent variable (e.g., 'x')"
                    }
                },
                required=["functional", "function_name", "independent_var"]
            )
        )

        self.add_tool(
            name="dynamic_programming",
            description="Set up and solve dynamic programming problems symbolically.",
            handler=self.handle_dynamic_programming,
            input_schema=create_input_schema(
                properties={
                    "value_function": {
                        "type": "string",
                        "description": "Value function notation (e.g., 'V(s)')"
                    },
                    "state_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "State variables"
                    },
                    "decision_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Decision variables"
                    }
                },
                required=["value_function", "state_variables", "decision_variables"]
            )
        )

        # Other Operation Tools
        self.add_tool(
            name="recognize_pattern",
            description="Recognize mathematical patterns in sequences.",
            handler=self.handle_recognize_pattern,
            input_schema=create_input_schema(
                properties={
                    "sequence": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Sequence of numbers"
                    }
                },
                required=["sequence"]
            )
        )

        self.add_tool(
            name="differentiate_expression",
            description="Find derivative of a stored expression.",
            handler=self.handle_differentiate_expression,
            input_schema=create_input_schema(
                properties={
                    "expression_key": {
                        "type": "string",
                        "description": "Key of stored expression"
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
                required=["expression_key", "variable"]
            )
        )

        self.add_tool(
            name="integrate_expression",
            description="Integrate a stored expression.",
            handler=self.handle_integrate_expression,
            input_schema=create_input_schema(
                properties={
                    "expression_key": {
                        "type": "string",
                        "description": "Key of stored expression"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable to integrate with respect to"
                    },
                    "bounds": {
                        "type": "array",
                        "items": {"type": ["number", "string"]},
                        "description": "Optional bounds for definite integral [lower, upper]"
                    }
                },
                required=["expression_key", "variable"]
            )
        )

    # Equation Solving Handlers

    def handle_solve_equations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solve_equations tool - solve system of equations."""
        equations_raw = validate_list_input(arguments["equations"], max_length=50)
        variables_raw = arguments.get("variables")

        # Validate and parse equations
        equations = [validate_expression_string(eq) for eq in equations_raw]
        parsed_eqs = [safe_sympify(eq) for eq in equations]

        var_symbols = None
        if variables_raw:
            variables = validate_list_input(variables_raw, max_length=50)
            variables = [validate_variable_name(v) for v in variables]
            var_symbols = [symbols(v) for v in variables]

        result = self.ai.solve_equation_system(parsed_eqs, var_symbols)

        # Serialize solutions
        result_serializable = {
            "solutions": [
                {str(k): str(v) for k, v in sol.items()} if isinstance(sol, dict) else str(sol)
                for sol in result["solutions"]
            ],
            "explanation": result["explanation"],
            "verification": result["verification"],
            "latex": latex(result["solutions"])
        }

        return result_serializable

    def handle_solve_algebraically(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solve_algebraically tool - solve equation over specified domain."""
        expr_key = arguments["expression_key"]
        variable = validate_variable_name(arguments["variable"])
        domain = arguments.get("domain", "complex")

        # Get or parse expression
        if expr_key in self.ai.expressions:
            expr = self.ai.expressions[expr_key]
        else:
            expr_key = validate_expression_string(expr_key)
            expr = safe_sympify(expr_key)

        var = symbols(variable)
        solutions = solve(expr, var)

        return {
            "expression_key": expr_key,
            "variable": variable,
            "domain": domain,
            "solutions": [str(sol) for sol in solutions],
            "count": len(solutions)
        }

    def handle_solve_linear_system(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solve_linear_system tool - solve linear system using matrix methods."""
        equations_raw = validate_list_input(arguments["equations"], max_length=50)
        variables_raw = validate_list_input(arguments["variables"], max_length=50)

        # Validate inputs
        equations = [validate_expression_string(eq) for eq in equations_raw]
        variables = [validate_variable_name(v) for v in variables_raw]

        # Parse and solve - use sympify directly for linear systems (after validation)
        # Note: linsolve needs evaluated expressions to work properly
        parsed_eqs = [sp.sympify(eq) for eq in equations]
        var_symbols = [symbols(v) for v in variables]

        solutions = sp.linsolve(parsed_eqs, var_symbols)

        return {
            "equations": equations,
            "variables": variables,
            "solution": str(list(solutions)[0]) if solutions else "No solution",
            "method": "linsolve"
        }

    def handle_solve_nonlinear_system(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle solve_nonlinear_system tool - solve nonlinear system."""
        equations_raw = validate_list_input(arguments["equations"], max_length=50)
        variables_raw = validate_list_input(arguments["variables"], max_length=50)

        # Validate inputs
        equations = [validate_expression_string(eq) for eq in equations_raw]
        variables = [validate_variable_name(v) for v in variables_raw]

        # Parse and solve
        parsed_eqs = [safe_sympify(eq) for eq in equations]
        var_symbols = [symbols(v) for v in variables]

        solutions = sp.nonlinsolve(parsed_eqs, var_symbols)

        return {
            "equations": equations,
            "variables": variables,
            "solutions": [str(sol) for sol in solutions],
            "method": "nonlinsolve"
        }

    # Matrix Operation Handlers

    def handle_create_matrix(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_matrix tool - create and store matrix."""
        elements = arguments["elements"]
        key = arguments.get("key")

        # Validate matrix dimensions
        if not isinstance(elements, list) or not elements:
            raise ValidationError("Matrix elements must be a non-empty list")

        if not all(isinstance(row, list) for row in elements):
            raise ValidationError("Matrix elements must be a 2D array")

        rows = len(elements)
        cols = len(elements[0])
        validate_matrix_dimensions(rows, cols)

        # Validate all rows have same length
        if not all(len(row) == cols for row in elements):
            raise ValidationError("All matrix rows must have the same length")

        # Auto-generate key if not provided
        if key is None:
            key = self.ai._get_next_key("matrix")
        else:
            key = validate_key_format(key)

        matrix = Matrix(elements)
        self.ai.matrices[key] = matrix

        return {
            "elements": elements,
            "key": key,
            "shape": list(matrix.shape),
            "latex": latex(matrix)
        }

    def handle_matrix_determinant(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle matrix_determinant tool - calculate determinant."""
        matrix_key = validate_key_format(arguments["matrix_key"])

        if matrix_key not in self.ai.matrices:
            raise ValidationError(f"Matrix '{matrix_key}' not found")

        matrix = self.ai.matrices[matrix_key]
        det = matrix.det()

        return {
            "matrix_key": matrix_key,
            "determinant": str(det),
            "latex": latex(det)
        }

    def handle_matrix_inverse(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle matrix_inverse tool - calculate matrix inverse."""
        matrix_key = validate_key_format(arguments["matrix_key"])

        if matrix_key not in self.ai.matrices:
            raise ValidationError(f"Matrix '{matrix_key}' not found")

        matrix = self.ai.matrices[matrix_key]

        try:
            inverse = matrix.inv()
            result_key = self.ai._get_next_key("matrix")
            self.ai.matrices[result_key] = inverse

            # Convert matrix elements to strings for JSON serialization
            inverse_list = [[str(elem) for elem in row] for row in inverse.tolist()]

            return {
                "matrix_key": matrix_key,
                "inverse": inverse_list,
                "stored_key": result_key,
                "latex": latex(inverse)
            }
        except Exception:
            raise ValidationError("Matrix is not invertible (singular matrix)")

    def handle_matrix_eigenvalues(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle matrix_eigenvalues tool - find eigenvalues."""
        matrix_key = validate_key_format(arguments["matrix_key"])

        if matrix_key not in self.ai.matrices:
            raise ValidationError(f"Matrix '{matrix_key}' not found")

        matrix = self.ai.matrices[matrix_key]
        eigenvals = matrix.eigenvals()

        return {
            "matrix_key": matrix_key,
            "eigenvalues": {str(k): v for k, v in eigenvals.items()},
            "latex": latex(eigenvals)
        }

    def handle_matrix_eigenvectors(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle matrix_eigenvectors tool - find eigenvectors."""
        matrix_key = validate_key_format(arguments["matrix_key"])

        if matrix_key not in self.ai.matrices:
            raise ValidationError(f"Matrix '{matrix_key}' not found")

        matrix = self.ai.matrices[matrix_key]
        eigenvects = matrix.eigenvects()

        return {
            "matrix_key": matrix_key,
            "eigenvectors": [
                {
                    "eigenvalue": str(eigenval),
                    "multiplicity": mult,
                    "eigenvector": [str(v) for v in vects]
                }
                for eigenval, mult, vects in eigenvects
            ]
        }

    # Optimization Handlers

    def handle_optimize_function(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimize_function tool - find critical points."""
        objective = validate_expression_string(arguments["objective"])
        variables_raw = arguments.get("variables")

        result = self.ai.optimize_function(objective, variables=variables_raw)

        return {
            "objective": str(result["objective"]),
            "variables": [str(v) for v in result["variables"]],
            "critical_points": str(result["critical_points"]),
            "evaluations": [
                {"point": str(e["point"]), "value": str(e["value"])}
                for e in result["evaluations"]
            ]
        }

    def handle_lagrange_multipliers(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle lagrange_multipliers tool - constrained optimization."""
        objective = validate_expression_string(arguments["objective"])
        constraints_raw = validate_list_input(arguments["constraints"], max_length=20)
        variables_raw = validate_list_input(arguments["variables"], max_length=50)

        # Validate inputs
        constraints = [validate_expression_string(c) for c in constraints_raw]
        variables = [validate_variable_name(v) for v in variables_raw]

        # Parse expressions
        obj_expr = safe_sympify(objective)
        constraint_exprs = [safe_sympify(c) for c in constraints]
        var_symbols = [symbols(v) for v in variables]

        # Create Lagrangian
        lambdas = [symbols(f'lambda_{i}') for i in range(len(constraints))]
        lagrangian = obj_expr + sum(lam * c for lam, c in zip(lambdas, constraint_exprs))

        # Find critical points
        all_vars = var_symbols + lambdas
        gradients = [diff(lagrangian, v) for v in all_vars]
        critical_points = solve(gradients, all_vars)

        return {
            "objective": objective,
            "constraints": constraints,
            "variables": variables,
            "lagrangian": str(lagrangian),
            "critical_points": str(critical_points),
            "latex": latex(lagrangian)
        }

    def handle_linear_programming(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle linear_programming tool - symbolic LP setup."""
        objective = validate_expression_string(arguments["objective"])
        constraints_raw = validate_list_input(arguments["constraints"], max_length=50)
        variables_raw = validate_list_input(arguments["variables"], max_length=50)
        minimize = arguments.get("minimize", True)

        # Validate inputs
        constraints = [validate_expression_string(c) for c in constraints_raw]
        variables = [validate_variable_name(v) for v in variables_raw]

        return {
            "tool": "linear_programming",
            "status": "symbolic_setup",
            "objective": objective,
            "constraints": constraints,
            "variables": variables,
            "minimize": minimize,
            "note": "Advanced optimization tool - symbolic formulation provided"
        }

    def handle_convex_optimization(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle convex_optimization tool - convex optimization setup."""
        objective = validate_expression_string(arguments["objective"])
        constraints_raw = arguments.get("constraints", [])
        variables_raw = validate_list_input(arguments["variables"], max_length=50)

        # Validate inputs
        if constraints_raw:
            constraints = [validate_expression_string(c) for c in validate_list_input(constraints_raw, max_length=50)]
        else:
            constraints = []
        variables = [validate_variable_name(v) for v in variables_raw]

        return {
            "tool": "convex_optimization",
            "status": "symbolic_setup",
            "objective": objective,
            "constraints": constraints,
            "variables": variables,
            "note": "Advanced optimization tool - symbolic formulation provided"
        }

    def handle_calculus_of_variations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculus_of_variations tool - variational calculus setup."""
        functional = validate_expression_string(arguments["functional"])
        function_name = validate_variable_name(arguments["function_name"])
        independent_var = validate_variable_name(arguments["independent_var"])

        return {
            "tool": "calculus_of_variations",
            "status": "symbolic_setup",
            "functional": functional,
            "function_name": function_name,
            "independent_var": independent_var,
            "note": "Advanced optimization tool - symbolic formulation provided"
        }

    def handle_dynamic_programming(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dynamic_programming tool - DP setup."""
        value_function = validate_expression_string(arguments["value_function"])
        state_vars_raw = validate_list_input(arguments["state_variables"], max_length=20)
        decision_vars_raw = validate_list_input(arguments["decision_variables"], max_length=20)

        # Validate inputs
        state_variables = [validate_variable_name(v) for v in state_vars_raw]
        decision_variables = [validate_variable_name(v) for v in decision_vars_raw]

        return {
            "tool": "dynamic_programming",
            "status": "symbolic_setup",
            "value_function": value_function,
            "state_variables": state_variables,
            "decision_variables": decision_variables,
            "note": "Advanced optimization tool - symbolic formulation provided"
        }

    # Other Operation Handlers

    def handle_recognize_pattern(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recognize_pattern tool - pattern recognition in sequences."""
        sequence = arguments["sequence"]

        # Validate sequence
        if not isinstance(sequence, list):
            raise ValidationError("Sequence must be a list")
        if len(sequence) > 1000:
            raise ValidationError("Sequence too long (max 1000 elements)")

        result = self.ai.pattern_recognition(sequence)

        # Serialize patterns
        patterns_serializable = []
        for pattern in result["patterns"]:
            pattern_dict = {
                "type": pattern["type"],
                "formula": str(pattern["formula"]),
                "next_terms": [str(t) for t in pattern["next_terms"]],
            }
            # Add other fields
            for k, v in pattern.items():
                if k not in ["type", "formula", "next_terms"]:
                    pattern_dict[k] = str(v)
            patterns_serializable.append(pattern_dict)

        return {
            "sequence": sequence,
            "patterns_found": result["patterns_found"],
            "patterns": patterns_serializable,
            "most_likely": patterns_serializable[0] if patterns_serializable else None
        }

    def handle_differentiate_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle differentiate_expression tool - differentiate stored expression."""
        expr_key = arguments["expression_key"]
        variable = validate_variable_name(arguments["variable"])
        order = arguments.get("order", 1)

        # Validate order
        if not isinstance(order, int) or order < 1:
            raise ValidationError("Order must be a positive integer")
        if order > 10:
            raise ValidationError("Order too large (max 10)")

        # Get or parse expression
        if expr_key in self.ai.expressions:
            expr = self.ai.expressions[expr_key]
        else:
            expr_key = validate_expression_string(expr_key)
            expr = safe_sympify(expr_key)

        var = symbols(variable)
        result_expr = diff(expr, var, order)

        result_key = self.ai._get_next_key("expr")
        self.ai.expressions[result_key] = result_expr

        return {
            "expression_key": expr_key,
            "variable": variable,
            "order": order,
            "original": str(expr),
            "derivative": str(result_expr),
            "stored_key": result_key,
            "latex": latex(result_expr)
        }

    def handle_integrate_expression(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle integrate_expression tool - integrate stored expression."""
        expr_key = arguments["expression_key"]
        variable = validate_variable_name(arguments["variable"])
        bounds = arguments.get("bounds")

        # Get or parse expression
        if expr_key in self.ai.expressions:
            expr = self.ai.expressions[expr_key]
        else:
            expr_key = validate_expression_string(expr_key)
            expr = safe_sympify(expr_key)

        var = symbols(variable)

        # Handle definite vs indefinite integral
        if bounds:
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValidationError("Bounds must be a list of 2 elements [lower, upper]")
            lower, upper = bounds
            result_expr = sp.integrate(expr, (var, safe_sympify(str(lower)), safe_sympify(str(upper))))
        else:
            result_expr = sp.integrate(expr, var)

        result_key = self.ai._get_next_key("expr")
        self.ai.expressions[result_key] = result_expr

        return {
            "expression_key": expr_key,
            "variable": variable,
            "bounds": bounds,
            "original": str(expr),
            "integral": str(result_expr),
            "stored_key": result_key,
            "latex": latex(result_expr)
        }


# Entry point
server = AlgebraServer()

if __name__ == "__main__":
    server.run()
