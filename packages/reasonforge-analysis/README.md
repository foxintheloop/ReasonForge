# ReasonForge Analysis

**Advanced calculus, differential equations, and transforms - 17 tools**

An MCP (Model Context Protocol) server that provides Claude with advanced mathematical analysis capabilities using SymPy's symbolic engine.

> **Beta Status**: This package is functional but still undergoing testing. Some edge cases may not be fully covered. Please report issues on [GitHub](https://github.com/foxintheloop/ReasonForge/issues).

## Capabilities

- **Differential Equations** - Solve ODEs and PDEs symbolically with initial/boundary conditions
- **Physics PDEs** - Schrodinger, wave, and heat equation solvers
- **Integral Transforms** - Laplace, Fourier, Z-transform, Mellin, and custom transforms
- **Signal Processing** - Convolution and transfer function analysis
- **Asymptotic Methods** - Perturbation theory and asymptotic expansions

## Installation

```bash
pip install reasonforge-analysis
```

Or install from source:

```bash
git clone https://github.com/foxintheloop/ReasonForge.git
cd ReasonForge
pip install -e packages/reasonforge -e packages/reasonforge-analysis
```

## Claude Desktop Configuration

Add to your Claude Desktop config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reasonforge-analysis": {
      "command": "python",
      "args": ["-m", "reasonforge_analysis"]
    }
  }
}
```

## Tools

### Differential Equations (3 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `dsolve_ode` | Solve ordinary differential equations | Solve y'' + y = 0 |
| `pdsolve_pde` | Solve partial differential equations | Solve heat equation |
| `symbolic_ode_initial_conditions` | Apply initial conditions to ODE solutions | y(0) = 1, y'(0) = 0 |

### Physics PDEs (3 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `schrodinger_equation_solver` | Solve time-independent Schrodinger equation | Quantum harmonic oscillator |
| `wave_equation_solver` | Solve wave equation with boundary conditions | Vibrating string |
| `heat_equation_solver` | Solve heat/diffusion equation | Temperature distribution |

### Integral Transforms (5 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `laplace_transform` | Compute Laplace transforms | Transform e^(-at) |
| `fourier_transform` | Compute Fourier transforms | Frequency analysis |
| `z_transform` | Compute Z-transforms for discrete signals | Digital signal processing |
| `mellin_transform` | Compute Mellin transforms | Special function analysis |
| `integral_transforms_custom` | Custom integral transforms | User-defined kernels |

### Signal Processing (2 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `convolution` | Symbolic convolution of functions | System response |
| `transfer_function_analysis` | Analyze transfer functions | Poles, zeros, stability |

### Asymptotic Methods (2 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `perturbation_theory` | Perturbation expansions | Approximate solutions |
| `asymptotic_analysis` | Asymptotic series and limits | Behavior at infinity |

### Special Functions (2 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `special_functions_properties` | Properties of special functions | Bessel, Legendre, etc. |
| `symbolic_optimization_setup` | Set up optimization problems | Constraints and objectives |

## Example Usage

Once configured, you can ask Claude:

**Differential Equations:**
- "Solve the differential equation y'' + 4y = 0 with y(0) = 1, y'(0) = 0"
- "Solve the heat equation for a rod with insulated ends"

**Transforms:**
- "Find the Laplace transform of sin(3t)"
- "Compute the inverse Fourier transform of 1/(1 + omega^2)"

**Signal Processing:**
- "Convolve e^(-t) with sin(t) for t > 0"
- "Find the poles and zeros of H(s) = (s + 1)/(s^2 + 2s + 5)"

## Dependencies

- Python >= 3.10
- mcp >= 1.0.0
- sympy >= 1.12
- scipy >= 1.11.0
- reasonforge (core library)

## Running Tests

```bash
pytest packages/reasonforge-analysis/tests/ -v
```

## License

MIT License - See [LICENSE](../../LICENSE) for details.

## Related Packages

- [reasonforge](https://pypi.org/project/reasonforge/) - Core symbolic computation library
- [reasonforge-logic](https://pypi.org/project/reasonforge-logic/) - Formal reasoning and logic
- [reasonforge-algebra](https://pypi.org/project/reasonforge-algebra/) - Algebraic operations
- [reasonforge-geometry](https://pypi.org/project/reasonforge-geometry/) - Vector/tensor calculus
