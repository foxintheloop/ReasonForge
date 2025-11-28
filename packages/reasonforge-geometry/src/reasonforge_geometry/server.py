"""ReasonForge Geometry - Vector/Tensor Calculus and Visualization
15 tools for geometry, general relativity, and visualization."""

from typing import Dict, Any

import sympy as sp
from sympy import symbols, latex, simplify, Matrix
from sympy.vector import CoordSys3D, divergence, curl, gradient
from sympy.physics.units import convert_to
from sympy.plotting import plot

from reasonforge import (
    BaseReasonForgeServer,
    SymbolicAI,
    create_input_schema,
    safe_sympify,
    validate_variable_name,
    validate_expression_string,
    validate_key_format,
    ValidationError,
)


class GeometryServer(BaseReasonForgeServer):
    """MCP server for geometry, tensor calculus, and visualization."""

    def __init__(self):
        super().__init__("reasonforge-geometry")
        self.ai = SymbolicAI()

    def register_tools(self):
        """Register all 15 geometry tools."""

        # Vector Calculus Tools
        self.add_tool(
            name="create_coordinate_system",
            description="Create coordinate system (Cartesian, cylindrical, spherical).",
            handler=self.handle_create_coordinate_system,
            input_schema=create_input_schema(
                properties={
                    "name": {"type": "string"},
                    "type": {"type": "string"}
                },
                required=["name", "type"]
            )
        )

        self.add_tool(
            name="create_vector_field",
            description="Create vector field in specified coordinates.",
            handler=self.handle_create_vector_field,
            input_schema=create_input_schema(
                properties={
                    "coord_system": {"type": "string"},
                    "components": {"type": "object"}
                },
                required=["coord_system", "components"]
            )
        )

        self.add_tool(
            name="calculate_curl",
            description="Calculate curl of vector field.",
            handler=self.handle_calculate_curl,
            input_schema=create_input_schema(
                properties={"field_name": {"type": "string"}},
                required=["field_name"]
            )
        )

        self.add_tool(
            name="calculate_divergence",
            description="Calculate divergence of vector field.",
            handler=self.handle_calculate_divergence,
            input_schema=create_input_schema(
                properties={"field_name": {"type": "string"}},
                required=["field_name"]
            )
        )

        self.add_tool(
            name="calculate_gradient",
            description="Calculate gradient of scalar field.",
            handler=self.handle_calculate_gradient,
            input_schema=create_input_schema(
                properties={
                    "scalar_field": {"type": "string"},
                    "coord_system": {"type": "string"}
                },
                required=["scalar_field"]
            )
        )

        # Tensor Calculus & GR Tools
        self.add_tool(
            name="create_predefined_metric",
            description="Create predefined spacetime metric (Schwarzschild, Kerr, FLRW, etc.).",
            handler=self.handle_create_predefined_metric,
            input_schema=create_input_schema(
                properties={
                    "metric_type": {"type": "string"},
                    "parameters": {"type": "object"}
                },
                required=["metric_type"]
            )
        )

        self.add_tool(
            name="search_predefined_metrics",
            description="Search available predefined metrics.",
            handler=self.handle_search_predefined_metrics,
            input_schema=create_input_schema(
                properties={"query": {"type": "string"}}
            )
        )

        self.add_tool(
            name="calculate_tensor",
            description="Calculate tensor (Ricci, Einstein, Weyl, etc.) from metric.",
            handler=self.handle_calculate_tensor,
            input_schema=create_input_schema(
                properties={
                    "metric_key": {"type": "string"},
                    "tensor_type": {"type": "string"}
                },
                required=["metric_key", "tensor_type"]
            )
        )

        self.add_tool(
            name="create_custom_metric",
            description="Create custom metric tensor.",
            handler=self.handle_create_custom_metric,
            input_schema=create_input_schema(
                properties={
                    "components": {"type": "array"},
                    "coordinates": {"type": "array"}
                },
                required=["components"]
            )
        )

        self.add_tool(
            name="print_latex_tensor",
            description="Get LaTeX representation of tensor.",
            handler=self.handle_print_latex_tensor,
            input_schema=create_input_schema(
                properties={"key": {"type": "string"}},
                required=["key"]
            )
        )

        # Units Tools
        self.add_tool(
            name="convert_to_units",
            description="Convert expression to different units.",
            handler=self.handle_convert_to_units,
            input_schema=create_input_schema(
                properties={
                    "expression": {"type": "string"},
                    "target_units": {"type": "string"}
                },
                required=["expression", "target_units"]
            )
        )

        self.add_tool(
            name="quantity_simplify_units",
            description="Simplify units in expression.",
            handler=self.handle_quantity_simplify_units,
            input_schema=create_input_schema(
                properties={"expression": {"type": "string"}},
                required=["expression"]
            )
        )

        # Visualization Tools
        self.add_tool(
            name="plot_symbolic_function",
            description="Plot symbolic function.",
            handler=self.handle_plot_symbolic_function,
            input_schema=create_input_schema(
                properties={
                    "expression": {"type": "string"},
                    "variable": {"type": "string"},
                    "range": {"type": "object"}
                },
                required=["expression", "variable"]
            )
        )

        self.add_tool(
            name="contour_plot_symbolic",
            description="Create contour plot.",
            handler=self.handle_contour_plot_symbolic,
            input_schema=create_input_schema(
                properties={
                    "expression": {"type": "string"},
                    "x_range": {"type": "object"},
                    "y_range": {"type": "object"}
                },
                required=["expression"]
            )
        )

        self.add_tool(
            name="vector_field_plot",
            description="Visualize vector field.",
            handler=self.handle_vector_field_plot,
            input_schema=create_input_schema(
                properties={
                    "field_components": {"type": "object"},
                    "dimension": {"type": "integer"}
                },
                required=["field_components"]
            )
        )

    # Vector Calculus Handlers

    def handle_create_coordinate_system(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_coordinate_system tool - create coordinate system."""
        coord_name = validate_variable_name(arguments["name"])
        coord_type = arguments["type"]

        # Validate coordinate type
        if coord_type not in ["Cartesian", "cylindrical", "spherical"]:
            raise ValidationError(f"Unknown coordinate type: {coord_type}. Use Cartesian, cylindrical, or spherical")

        # Create coordinate system
        if coord_type == "Cartesian":
            coord_system = CoordSys3D(coord_name)
        elif coord_type == "cylindrical":
            coord_system = CoordSys3D(coord_name, transformation='cylindrical')
        elif coord_type == "spherical":
            coord_system = CoordSys3D(coord_name, transformation='spherical')

        self.ai.coordinate_systems[coord_name] = coord_system

        return {
            "name": coord_name,
            "type": coord_type,
            "basis_vectors": {
                "i": str(coord_system.i),
                "j": str(coord_system.j),
                "k": str(coord_system.k)
            },
            "origin": str(coord_system.origin)
        }

    def handle_create_vector_field(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_vector_field tool - create vector field."""
        coord_system_name = validate_variable_name(arguments["coord_system"])
        components = arguments["components"]

        if coord_system_name not in self.ai.coordinate_systems:
            raise ValidationError(f"Coordinate system '{coord_system_name}' not found. Create it first.")

        coord_sys = self.ai.coordinate_systems[coord_system_name]

        # Parse component expressions - use sympify after validation
        i_comp_str = validate_expression_string(components.get("i", "0"))
        j_comp_str = validate_expression_string(components.get("j", "0"))
        k_comp_str = validate_expression_string(components.get("k", "0"))

        i_comp = sp.sympify(i_comp_str)
        j_comp = sp.sympify(j_comp_str)
        k_comp = sp.sympify(k_comp_str)

        # Create vector field
        vector_field = i_comp * coord_sys.i + j_comp * coord_sys.j + k_comp * coord_sys.k

        # Store with auto-generated key
        field_key = f"field_{len(self.ai.vector_fields)}"
        self.ai.vector_fields[field_key] = vector_field

        return {
            "field_key": field_key,
            "coordinate_system": coord_system_name,
            "components": {
                "i": i_comp_str,
                "j": j_comp_str,
                "k": k_comp_str
            },
            "vector_field": str(vector_field)
        }

    def handle_calculate_curl(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculate_curl tool - calculate curl of vector field."""
        field_name = validate_key_format(arguments["field_name"])

        if field_name not in self.ai.vector_fields:
            raise ValidationError(f"Vector field '{field_name}' not found.")

        vector_field = self.ai.vector_fields[field_name]
        curl_result = curl(vector_field)

        return {
            "field_name": field_name,
            "curl": str(curl_result),
            "latex": latex(curl_result)
        }

    def handle_calculate_divergence(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculate_divergence tool - calculate divergence of vector field."""
        field_name = validate_key_format(arguments["field_name"])

        if field_name not in self.ai.vector_fields:
            raise ValidationError(f"Vector field '{field_name}' not found.")

        vector_field = self.ai.vector_fields[field_name]
        div_result = divergence(vector_field)

        return {
            "field_name": field_name,
            "divergence": str(div_result),
            "latex": latex(div_result)
        }

    def handle_calculate_gradient(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculate_gradient tool - calculate gradient of scalar field."""
        scalar_field_str = validate_expression_string(arguments["scalar_field"])
        coord_system_name = arguments.get("coord_system")

        if coord_system_name:
            coord_system_name = validate_variable_name(coord_system_name)
            if coord_system_name not in self.ai.coordinate_systems:
                raise ValidationError(f"Coordinate system '{coord_system_name}' not found.")
            coord_sys = self.ai.coordinate_systems[coord_system_name]
        else:
            # Use default Cartesian coordinates
            coord_sys = CoordSys3D('C')

        scalar_field = sp.sympify(scalar_field_str)
        grad_result = gradient(scalar_field, coord_sys)

        return {
            "scalar_field": scalar_field_str,
            "gradient": str(grad_result),
            "latex": latex(grad_result)
        }

    # Tensor Calculus & GR Handlers

    def handle_create_predefined_metric(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_predefined_metric tool - create predefined spacetime metric."""
        metric_type = arguments["metric_type"]
        parameters = arguments.get("parameters", {})

        # Create predefined metrics
        if metric_type == "Schwarzschild":
            M = symbols('M', positive=True)
            r, theta = symbols('r theta', real=True)

            g_tt = -(1 - 2*M/r)
            g_rr = 1/(1 - 2*M/r)
            g_theta_theta = r**2
            g_phi_phi = r**2 * sp.sin(theta)**2

            metric = Matrix([
                [g_tt, 0, 0, 0],
                [0, g_rr, 0, 0],
                [0, 0, g_theta_theta, 0],
                [0, 0, 0, g_phi_phi]
            ])
            coords = ['t', 'r', 'theta', 'phi']

        elif metric_type == "Minkowski":
            metric = Matrix([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            coords = ['t', 'x', 'y', 'z']

        elif metric_type == "FLRW":
            t, r, theta = symbols('t r theta', real=True)
            a = sp.Function('a')(t)
            k = symbols('k', real=True)

            metric = Matrix([
                [-1, 0, 0, 0],
                [0, a**2/(1-k*r**2), 0, 0],
                [0, 0, a**2*r**2, 0],
                [0, 0, 0, a**2*r**2*sp.sin(theta)**2]
            ])
            coords = ['t', 'r', 'theta', 'phi']

        else:
            raise ValidationError(f"Unknown metric type: {metric_type}. Use: Schwarzschild, Minkowski, FLRW")

        # Store metric
        metric_key = f"metric_{metric_type.lower()}_{len(self.ai.metrics)}"
        self.ai.metrics[metric_key] = {
            "metric": metric,
            "coordinates": coords,
            "type": metric_type
        }

        return {
            "metric_key": metric_key,
            "metric_type": metric_type,
            "coordinates": coords,
            "metric_components": [[str(metric[i, j]) for j in range(metric.cols)] for i in range(metric.rows)],
            "latex": latex(metric)
        }

    def handle_search_predefined_metrics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search_predefined_metrics tool - search available metrics."""
        query = arguments.get("query", "").lower()

        available_metrics = [
            {
                "name": "Schwarzschild",
                "description": "Schwarzschild metric for spherically symmetric black holes",
                "coordinates": "t, r, theta, phi",
                "parameters": "M (mass)"
            },
            {
                "name": "Minkowski",
                "description": "Minkowski metric for flat spacetime (special relativity)",
                "coordinates": "t, x, y, z",
                "parameters": "none"
            },
            {
                "name": "FLRW",
                "description": "Friedmann-Lemaitre-Robertson-Walker metric for cosmology",
                "coordinates": "t, r, theta, phi",
                "parameters": "a(t) (scale factor), k (curvature)"
            }
        ]

        # Filter by query if provided
        if query:
            filtered = [m for m in available_metrics
                       if query in m["name"].lower() or query in m["description"].lower()]
        else:
            filtered = available_metrics

        return {
            "query": query,
            "found": len(filtered),
            "metrics": filtered
        }

    def handle_calculate_tensor(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculate_tensor tool - calculate tensor from metric."""
        metric_key = validate_key_format(arguments["metric_key"])
        tensor_type = arguments["tensor_type"]

        if metric_key not in self.ai.metrics:
            raise ValidationError(f"Metric '{metric_key}' not found.")

        metric_data = self.ai.metrics[metric_key]
        g = metric_data["metric"]
        coords_symbols = [symbols(c, real=True) for c in metric_data["coordinates"]]

        # Calculate requested tensor
        if tensor_type == "inverse":
            tensor_result = g.inv()
            tensor_name = "Inverse Metric (g^μν)"

        elif tensor_type == "determinant":
            tensor_result = g.det()
            tensor_name = "Metric Determinant"

        elif tensor_type == "Christoffel":
            # Christoffel symbols calculation
            dim = len(coords_symbols)
            g_inv = g.inv()

            christoffel_sample = []
            for mu in range(min(2, dim)):
                for nu in range(min(2, dim)):
                    for rho in range(min(2, dim)):
                        term = 0
                        for sigma in range(dim):
                            term += g_inv[mu, sigma] * (
                                sp.diff(g[sigma, rho], coords_symbols[nu]) +
                                sp.diff(g[sigma, nu], coords_symbols[rho]) -
                                sp.diff(g[nu, rho], coords_symbols[sigma])
                            ) / 2
                        christoffel_sample.append(f"Γ^{mu}_{nu}{rho} = {simplify(term)}")

            return {
                "metric_key": metric_key,
                "tensor_type": "Christoffel",
                "note": "Sample Christoffel symbols (full calculation requires 3D tensor)",
                "samples": christoffel_sample
            }

        else:
            raise ValidationError(f"Tensor type '{tensor_type}' not yet implemented. Try: inverse, determinant, Christoffel")

        # Store tensor
        tensor_key = f"tensor_{tensor_type}_{len(self.ai.tensor_objects)}"
        self.ai.tensor_objects[tensor_key] = tensor_result

        return {
            "metric_key": metric_key,
            "tensor_type": tensor_type,
            "tensor_key": tensor_key,
            "tensor_name": tensor_name,
            "result": str(tensor_result),
            "latex": latex(tensor_result)
        }

    def handle_create_custom_metric(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_custom_metric tool - create custom metric tensor."""
        components = arguments["components"]
        coordinates = arguments.get("coordinates", ["t", "x", "y", "z"])

        # Validate components is a list
        if not isinstance(components, list):
            raise ValidationError("Components must be a 2D array")

        # Create metric matrix
        metric = Matrix(components)

        # Store metric
        metric_key = f"metric_custom_{len(self.ai.metrics)}"
        self.ai.metrics[metric_key] = {
            "metric": metric,
            "coordinates": coordinates,
            "type": "custom"
        }

        return {
            "metric_key": metric_key,
            "coordinates": coordinates,
            "metric_components": [[str(metric[i, j]) for j in range(metric.cols)] for i in range(metric.rows)],
            "latex": latex(metric)
        }

    def handle_print_latex_tensor(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle print_latex_tensor tool - get LaTeX representation."""
        key = validate_key_format(arguments["key"])

        # Check in all storage locations
        if key in self.ai.tensor_objects:
            obj = self.ai.tensor_objects[key]
        elif key in self.ai.metrics:
            obj = self.ai.metrics[key]["metric"]
        elif key in self.ai.matrices:
            obj = self.ai.matrices[key]
        else:
            raise ValidationError(f"Tensor '{key}' not found.")

        return {
            "key": key,
            "latex": latex(obj),
            "string": str(obj)
        }

    # Units Handlers

    def handle_convert_to_units(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle convert_to_units tool - convert expression to different units."""
        expression_str = validate_expression_string(arguments["expression"])
        target_units_str = arguments["target_units"]

        # Parse expression - use sympify
        expr = sp.sympify(expression_str)

        # Get target unit from variables
        target_unit = self.ai.variables.get(target_units_str)
        if target_unit is None:
            raise ValidationError(f"Unknown unit: {target_units_str}")

        # Convert
        converted = convert_to(expr, target_unit)

        return {
            "original_expression": expression_str,
            "target_units": target_units_str,
            "converted": str(converted),
            "latex": latex(converted)
        }

    def handle_quantity_simplify_units(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantity_simplify_units tool - simplify units in expression."""
        expression_str = validate_expression_string(arguments["expression"])

        # Parse and simplify - use sympify
        expr = sp.sympify(expression_str)
        simplified = simplify(expr)

        return {
            "original_expression": expression_str,
            "simplified": str(simplified),
            "latex": latex(simplified)
        }

    # Visualization Handlers

    def handle_plot_symbolic_function(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plot_symbolic_function tool - plot symbolic function."""
        expression_str = validate_expression_string(arguments["expression"])
        variable = validate_variable_name(arguments["variable"])
        range_dict = arguments.get("range", {"min": -10, "max": 10})

        # Validate range
        if not isinstance(range_dict, dict):
            raise ValidationError("Range must be a dictionary with 'min' and 'max' keys")

        expr = sp.sympify(expression_str)
        var_symbol = symbols(variable)

        # Create plot (returns plot object, not displayed in MCP)
        p = plot(expr, (var_symbol, range_dict.get("min", -10), range_dict.get("max", 10)), show=False)

        return {
            "expression": expression_str,
            "variable": variable,
            "range": range_dict,
            "note": "Plot created but not displayed (MCP servers cannot render graphics). Use in Jupyter notebook or save to file.",
            "latex": latex(expr)
        }

    def handle_contour_plot_symbolic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle contour_plot_symbolic tool - create contour plot."""
        expression_str = validate_expression_string(arguments["expression"])
        x_range = arguments.get("x_range", {"min": -5, "max": 5})
        y_range = arguments.get("y_range", {"min": -5, "max": 5})

        expr = sp.sympify(expression_str)

        # Extract variables from expression
        vars_in_expr = list(expr.free_symbols)
        if len(vars_in_expr) < 2:
            raise ValidationError("Expression must contain at least 2 variables for contour plot.")

        return {
            "expression": expression_str,
            "variables": [str(v) for v in vars_in_expr],
            "x_range": x_range,
            "y_range": y_range,
            "note": "Contour plot specification created but not displayed (MCP servers cannot render graphics).",
            "latex": latex(expr)
        }

    def handle_vector_field_plot(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle vector_field_plot tool - visualize vector field."""
        field_components = arguments["field_components"]
        dimension = arguments.get("dimension", 2)

        # Validate dimension
        if not isinstance(dimension, int) or dimension < 2 or dimension > 3:
            raise ValidationError("Dimension must be 2 or 3")

        return {
            "field_components": field_components,
            "dimension": dimension,
            "note": "Vector field plot specification created but not displayed (MCP servers cannot render graphics).",
            "suggestion": "Use matplotlib's quiver() function in a Jupyter notebook to visualize."
        }


# Entry point
server = GeometryServer()

if __name__ == "__main__":
    server.run()
