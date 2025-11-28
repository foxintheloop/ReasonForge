# ReasonForge Algebra

**Algebraic operations, equation solving, matrices, and optimization MCP server - 18 tools**

Complete toolkit for algebraic computation including equation solving, matrix operations, and optimization.

## Features

- **18 Specialized Tools**: Comprehensive algebra capabilities
- **Equation Solving**: Linear, nonlinear, systems
- **Matrix Operations**: Determinants, inverses, eigenvalues/vectors
- **Optimization**: Lagrange multipliers, linear/convex programming
- **SymPy-Powered**: Deterministic symbolic computation

## Installation

```bash
pip install reasonforge-algebra
```

## Tools (18)

### Equation Solving (4 tools)
- **solve_equations**: Solve systems of equations
- **solve_algebraically**: Solve over specific domains (real, complex)
- **solve_linear_system**: Linear system solver
- **solve_nonlinear_system**: Nonlinear system solver

### Matrix Operations (5 tools)
- **create_matrix**: Create and store matrices
- **matrix_determinant**: Calculate determinants
- **matrix_inverse**: Find matrix inverses
- **matrix_eigenvalues**: Find eigenvalues
- **matrix_eigenvectors**: Find eigenvectors

### Optimization (6 tools)
- **optimize_function**: Find critical points
- **lagrange_multipliers**: Constrained optimization
- **linear_programming**: Linear program solver
- **convex_optimization**: Convex optimization
- **calculus_of_variations**: Variational calculus
- **dynamic_programming**: Dynamic programming setup

### Other (3 tools)
- **recognize_pattern**: Pattern recognition in sequences
- **differentiate_expression**: Differentiate stored expressions
- **integrate_expression**: Integrate stored expressions

## Examples

### Solving Equations
```
User: "Solve x^2 - 5x + 6 = 0"
Claude: Uses solve_equations...
Result: x = 2, x = 3
```

### Matrix Eigenvalues
```
User: "Find eigenvalues of [[1, 2], [3, 4]]"
Claude: Uses matrix_eigenvalues...
Result: λ₁ = (5-√33)/2, λ₂ = (5+√33)/2
```

### Optimization
```
User: "Maximize xy subject to x^2 + y^2 = 1"
Claude: Uses lagrange_multipliers...
Result: Critical points at (±1/√2, ±1/√2)
```

## Dependencies

- `mcp>=1.0.0` - Model Context Protocol
- `sympy>=1.12` - Symbolic mathematics
- `reasonforge>=0.1.0` - Core library

## License

MIT License

## Part of ReasonForge Ecosystem

- [reasonforge](https://github.com/yourusername/reasonforge) - Core library
- [reasonforge-expressions](https://github.com/yourusername/reasonforge-expressions) - Essentials
- **reasonforge-algebra** - This package
- [reasonforge-analysis](https://github.com/yourusername/reasonforge-analysis) - Advanced calculus
- [reasonforge-geometry](https://github.com/yourusername/reasonforge-geometry) - Vector/tensor calculus
- [reasonforge-statistics](https://github.com/yourusername/reasonforge-statistics) - Statistics
- [reasonforge-physics](https://github.com/yourusername/reasonforge-physics) - Physics
- [reasonforge-logic](https://github.com/yourusername/reasonforge-logic) - Logic & reasoning
