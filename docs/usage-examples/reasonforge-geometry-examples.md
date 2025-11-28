# reasonforge-geometry (15 tools)

**Vector/tensor calculus, general relativity, and visualization**

**Installation:** `pip install reasonforge-geometry`

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-analysis](./reasonforge-analysis-examples.md) | [Next: reasonforge-statistics](./reasonforge-statistics-examples.md)

---

## Vector Calculus (5 tools)

### Example 51: Create Coordinate System

**User:** "Set up a cylindrical coordinate system"

**Claude with ReasonForge:**
```
Using create_coordinate_system tool from reasonforge-geometry...

Result:
{
  "name": "cyl",
  "type": "cylindrical",
  "coordinates": ["rho", "phi", "z"],
  "transformation": {
    "x": "rho*cos(phi)",
    "y": "rho*sin(phi)",
    "z": "z"
  }
}
```

### Example 52: Create Vector Field

**User:** "Create a vector field F = (y, -x, 0) in Cartesian coordinates"

**Claude with ReasonForge:**
```
Using create_vector_field tool from reasonforge-geometry...

Result:
{
  "coord_system": "cart",
  "components": {"x": "y", "y": "-x", "z": "0"},
  "field_name": "field_1",
  "latex": "\\mathbf{F} = y\\mathbf{i} - x\\mathbf{j}"
}
```

### Example 53: Calculate Curl

**User:** "Find the curl of the vector field F = (y, -x, 0)"

**Claude with ReasonForge:**
```
Using calculate_curl tool from reasonforge-geometry...

Result:
{
  "field_name": "field_1",
  "curl": "(0, 0, -2)",
  "latex": "\\nabla \\times \\mathbf{F} = -2\\mathbf{k}",
  "interpretation": "Constant rotation around z-axis"
}
```

### Example 54: Calculate Divergence

**User:** "Calculate the divergence of F = (x, y, z)"

**Claude with ReasonForge:**
```
Using calculate_divergence tool from reasonforge-geometry...

Result:
{
  "field_name": "field_2",
  "divergence": "3",
  "latex": "\\nabla \\cdot \\mathbf{F} = 3",
  "interpretation": "Uniform expansion in all directions"
}
```

### Example 55: Calculate Gradient

**User:** "Find the gradient of f = x^2 + y^2 + z^2"

**Claude with ReasonForge:**
```
Using calculate_gradient tool from reasonforge-geometry...

Result:
{
  "scalar_field": "x**2 + y**2 + z**2",
  "coord_system": "cart",
  "gradient": "(2*x, 2*y, 2*z)",
  "latex": "\\nabla f = 2x\\mathbf{i} + 2y\\mathbf{j} + 2z\\mathbf{k}",
  "magnitude": "2*sqrt(x**2 + y**2 + z**2)"
}
```

## Tensor Calculus & GR (5 tools)

### Example 56: Create Predefined Metric

**User:** "Load the Schwarzschild metric with mass M"

**Claude with ReasonForge:**
```
Using create_predefined_metric tool from reasonforge-geometry...

Result:
{
  "metric_type": "Schwarzschild",
  "parameters": {"M": "M"},
  "key": "schwarzschild_1",
  "coordinates": ["t", "r", "theta", "phi"],
  "signature": "(-,+,+,+)",
  "metric_tensor": "diag(-(1 - 2*M/r), 1/(1 - 2*M/r), r**2, r**2*sin(theta)**2)"
}
```

### Example 57: Search Predefined Metrics

**User:** "What spacetime metrics are available?"

**Claude with ReasonForge:**
```
Using search_predefined_metrics tool from reasonforge-geometry...

Result:
{
  "query": "",
  "available_metrics": [
    {"name": "Schwarzschild", "description": "Spherically symmetric vacuum solution"},
    {"name": "Kerr", "description": "Rotating black hole metric"},
    {"name": "Minkowski", "description": "Flat spacetime metric"},
    {"name": "FLRW", "description": "Friedmann-Lemaître-Robertson-Walker cosmological metric"},
    {"name": "ReissnerNordstrom", "description": "Charged black hole metric"}
  ],
  "count": 5
}
```

### Example 58: Calculate Tensor

**User:** "Calculate the Ricci tensor for the Schwarzschild metric"

**Claude with ReasonForge:**
```
Using calculate_tensor tool from reasonforge-geometry...

Result:
{
  "metric_key": "schwarzschild_1",
  "tensor_type": "Ricci",
  "result": "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]",
  "interpretation": "Vacuum solution (Ricci tensor vanishes)",
  "stored_key": "ricci_1"
}
```

### Example 59: Create Custom Metric

**User:** "Create a custom 2D metric with components g_00=1, g_11=-1"

**Claude with ReasonForge:**
```
Using create_custom_metric tool from reasonforge-geometry...

Result:
{
  "components": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "coordinates": ["t", "x", "y", "z"],
  "key": "custom_metric_1",
  "determinant": "-1",
  "signature": "(+,-,+,+)"
}
```

### Example 60: Print LaTeX Tensor

**User:** "Show the Ricci tensor in LaTeX format"

**Claude with ReasonForge:**
```
Using print_latex_tensor tool from reasonforge-geometry...

Result:
{
  "key": "ricci_1",
  "type": "tensor",
  "latex": "R_{\\mu\\nu} = \\begin{bmatrix} 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 \\end{bmatrix}"
}
```

## Units (2 tools)

### Example 61: Convert to Units

**User:** "Convert 5 meters per second to kilometers per hour"

**Claude with ReasonForge:**
```
Using convert_to_units tool from reasonforge-geometry...

Result:
{
  "expression": "5*meter/second",
  "target_units": "kilometer/hour",
  "result": "18*kilometer/hour",
  "conversion_factor": "3.6"
}
```

### Example 62: Quantity Simplify Units

**User:** "Simplify the units of (kg*m/s^2)*m"

**Claude with ReasonForge:**
```
Using quantity_simplify_units tool from reasonforge-geometry...

Result:
{
  "expression": "(kilogram*meter/second**2)*meter",
  "simplified": "kilogram*meter**2/second**2",
  "interpretation": "Joules (energy unit)",
  "latex": "\\frac{kg \\cdot m^{2}}{s^{2}}"
}
```

## Visualization (3 tools)

### Example 63: Plot Symbolic Function

**User:** "Plot y = x^2 - 4x + 3 and find critical points"

**Claude with ReasonForge:**
```
Using plot_symbolic_function tool from reasonforge-geometry...

Result:
{
  "expression": "x**2 - 4*x + 3",
  "variable": "x",
  "range": {"min": -1, "max": 5},
  "output_format": ["latex", "critical_points"],
  "latex": "y = x^2 - 4x + 3",
  "critical_points": [
    {"x": "2", "y": "-1", "type": "minimum"}
  ],
  "zeros": ["1", "3"],
  "domain": "(-∞, ∞)"
}
```

### Example 64: Contour Plot Symbolic

**User:** "Create contour plot for z = x^2 + y^2 with level curves at z=1,4,9"

**Claude with ReasonForge:**
```
Using contour_plot_symbolic tool from reasonforge-geometry...

Result:
{
  "expression": "x**2 + y**2",
  "x_range": {"min": -5, "max": 5},
  "y_range": {"min": -5, "max": 5},
  "level_curves": [1, 4, 9],
  "level_equations": [
    "x**2 + y**2 = 1",
    "x**2 + y**2 = 4",
    "x**2 + y**2 = 9"
  ],
  "gradient": "(2*x, 2*y)",
  "critical_points": [{"x": "0", "y": "0", "value": "0"}]
}
```

### Example 65: Vector Field Plot

**User:** "Visualize the vector field F = (-y, x) and analyze it"

**Claude with ReasonForge:**
```
Using vector_field_plot tool from reasonforge-geometry...

Result:
{
  "field_components": {"x": "-y", "y": "x"},
  "dimension": 2,
  "divergence": "0",
  "curl": "2",
  "critical_points": [{"x": "0", "y": "0"}],
  "interpretation": "Circular flow around origin with constant curl"
}
```

---

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-analysis](./reasonforge-analysis-examples.md) | [Next: reasonforge-statistics](./reasonforge-statistics-examples.md)
