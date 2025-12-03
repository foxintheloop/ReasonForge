# ReasonForge Core Library

**Standalone symbolic computation library for mathematical reasoning**

ReasonForge is a pure Python library that provides symbolic mathematical computation using SymPy. It serves as the core computation engine for all ReasonForge MCP servers.

## Features

- **Deterministic**: Exact symbolic computation using SymPy
- **Complete State Management**: Variables, expressions, matrices, quantum states
- **Advanced Mathematics**: Calculus, algebra, optimization, pattern recognition
- **Pure Python**: No MCP dependencies, use anywhere
- **Well-Tested**: Foundation for 462 passing tests across MCP servers

## Installation

```bash
pip install reasonforge
```

To install the complete ReasonForge ecosystem with all domain-specific packages:

```bash
pip install reasonforge[all]
```

This installs: reasonforge-logic, reasonforge-algebra, reasonforge-expressions, reasonforge-analysis, reasonforge-geometry, reasonforge-statistics, and reasonforge-physics.

## Quick Start

```python
from reasonforge import SymbolicAI

# Create symbolic AI instance
ai = SymbolicAI()

# Define variables
x, y = ai.define_variables(['x', 'y'])

# Solve equations
result = ai.solve_equation_system([
    x**2 + y**2 - 25,  # Circle
    x + y - 7           # Line
])

print(result['solutions'])
# [{'x': 3, 'y': 4}, {'x': 4, 'y': 3}]
```

## Core Capabilities

### Equation Solving
```python
ai.solve_equation_system([x**2 - 5*x + 6], [x])
```

### Calculus Operations
```python
ai.perform_calculus("sin(x)*cos(x)", "x", operation="diff")
ai.perform_calculus("x**2", "x", operation="integrate")
```

### Optimization
```python
ai.optimize_function("x**2 + y**2", variables=[x, y])
```

### Pattern Recognition
```python
ai.pattern_recognition([1, 4, 9, 16, 25])  # Recognizes n^2 pattern
```

### Matrix Operations
```python
ai.matrix_operations([[[1, 2], [3, 4]]], operation="eigenvalues")
```

### Logical Reasoning
```python
ai.logical_reasoning(["A -> B", "A"], conclusion="B")
```

## State Management

The `SymbolicAI` class maintains state for:
- **Variables**: Symbolic variables with assumptions
- **Expressions**: Named expressions with auto-incrementing keys
- **Functions**: Function symbols for differential equations
- **Matrices**: Matrix objects with keys
- **Metrics**: Tensor/metric objects for general relativity
- **Coordinate Systems**: Vector calculus coordinate systems
- **Quantum States**: Quantum state vectors and density matrices

## Dependencies

- `sympy>=1.12` - Symbolic mathematics
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Visualization

## Used By

ReasonForge core library is used by:
- [reasonforge-expressions](https://pypi.org/project/reasonforge-expressions/) - Essential symbolic manipulation
- [reasonforge-algebra](https://pypi.org/project/reasonforge-algebra/) - Algebraic operations
- [reasonforge-analysis](https://pypi.org/project/reasonforge-analysis/) - Advanced calculus
- [reasonforge-geometry](https://pypi.org/project/reasonforge-geometry/) - Vector/tensor calculus
- [reasonforge-statistics](https://pypi.org/project/reasonforge-statistics/) - Probability & statistics
- [reasonforge-physics](https://pypi.org/project/reasonforge-physics/) - Physics & quantum computing
- [reasonforge-logic](https://pypi.org/project/reasonforge-logic/) - Symbolic reasoning & logic

## License

MIT License - See LICENSE file for details

## Citation

```bibtex
@software{reasonforge_core,
  title = {ReasonForge: Core Symbolic Computation Library},
  author = {Derek Fox},
  year = {2025},
  description = {Standalone Python library for exact symbolic mathematical reasoning},
  url = {https://github.com/foxintheloop/ReasonForge}
}
```
