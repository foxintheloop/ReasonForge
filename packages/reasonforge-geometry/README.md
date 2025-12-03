# ReasonForge Geometry

**Vector/tensor calculus, general relativity, and visualization - 15 tools**

An MCP (Model Context Protocol) server that provides Claude with geometric and tensor calculus capabilities using SymPy's symbolic engine.

> **Beta Status**: This package is functional but still undergoing testing. Some edge cases may not be fully covered. Please report issues on [GitHub](https://github.com/foxintheloop/ReasonForge/issues).

## Capabilities

- **Vector Calculus** - Gradient, divergence, curl in arbitrary coordinate systems
- **Tensor Calculus** - Christoffel symbols, Riemann and Ricci tensors, Einstein tensor
- **General Relativity** - Predefined metrics (Schwarzschild, Kerr, FLRW, etc.)
- **Unit Conversions** - Symbolic unit handling and conversions
- **Visualization** - 2D/3D plots, contour plots, vector field visualization

## Installation

```bash
pip install reasonforge-geometry
```

Or install from source:

```bash
git clone https://github.com/foxintheloop/ReasonForge.git
cd ReasonForge
pip install -e packages/reasonforge -e packages/reasonforge-geometry
```

## Claude Desktop Configuration

Add to your Claude Desktop config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reasonforge-geometry": {
      "command": "python",
      "args": ["-m", "reasonforge_geometry"]
    }
  }
}
```

## Tools

### Vector Calculus (5 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `create_coordinate_system` | Create coordinate system (Cartesian, spherical, cylindrical) | Set up spherical coordinates |
| `create_vector_field` | Define symbolic vector fields | F = x*i + y*j + z*k |
| `calculate_curl` | Compute curl of vector field | curl(F) |
| `calculate_divergence` | Compute divergence of vector field | div(F) |
| `calculate_gradient` | Compute gradient of scalar field | grad(phi) |

### Tensor Calculus & General Relativity (5 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `create_predefined_metric` | Load standard spacetime metrics | Schwarzschild, Kerr, FLRW |
| `search_predefined_metrics` | Search available metric library | Find all black hole metrics |
| `calculate_tensor` | Compute Christoffel, Riemann, Ricci, Einstein tensors | Curvature calculations |
| `create_custom_metric` | Define custom metric tensors | Custom spacetime geometry |
| `print_latex_tensor` | LaTeX output for tensors | Publication-ready equations |

### Unit Handling (2 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `convert_to_units` | Convert between unit systems | meters to feet |
| `quantity_simplify_units` | Simplify compound units | kg*m/s^2 to N |

### Visualization (3 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `plot_symbolic_function` | Plot 2D symbolic functions | Graph sin(x)*e^(-x) |
| `contour_plot_symbolic` | Generate contour plots | Level curves of x^2 + y^2 |
| `vector_field_plot` | Visualize vector fields | Flow field visualization |

## Example Usage

Once configured, you can ask Claude:

**Vector Calculus:**
- "Calculate the divergence of F = x^2*i + y^2*j + z^2*k"
- "Find the curl of the velocity field v = -y*i + x*j"
- "Compute the gradient of phi = x^2 + y^2 + z^2 in spherical coordinates"

**General Relativity:**
- "Show me the Schwarzschild metric for a black hole"
- "Calculate the Christoffel symbols for the FLRW metric"
- "What is the Ricci scalar for the Kerr metric?"

**Visualization:**
- "Plot the function sin(x)*cos(y) as a contour plot"
- "Visualize the vector field F = -y*i + x*j"

## Dependencies

- Python >= 3.10
- mcp >= 1.0.0
- sympy >= 1.12
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- reasonforge (core library)

## Running Tests

```bash
pytest packages/reasonforge-geometry/tests/ -v
```

## License

MIT License - See [LICENSE](../../LICENSE) for details.

## Related Packages

- [reasonforge](https://pypi.org/project/reasonforge/) - Core symbolic computation library
- [reasonforge-physics](https://pypi.org/project/reasonforge-physics/) - Classical and quantum physics
- [reasonforge-analysis](https://pypi.org/project/reasonforge-analysis/) - Differential equations and transforms
- [reasonforge-algebra](https://pypi.org/project/reasonforge-algebra/) - Algebraic operations
