# ReasonForge Expressions

**Essential symbolic expression manipulation MCP server - 15 fundamental tools**

ReasonForge Expressions is a focused MCP server providing the most essential tools for symbolic expression manipulation and basic calculus. Perfect for students, educators, and anyone needing fundamental mathematical computation.

## Features

- **15 Essential Tools**: Carefully selected core functionality
- **90% Token Reduction**: ~800 tokens vs ~8,000 for full server
- **SymPy-Powered**: Deterministic symbolic computation
- **Fast & Lightweight**: Minimal dependencies
- **Foundation Package**: Used by all other ReasonForge servers

## Installation

```bash
pip install reasonforge-expressions
```

## Usage

### With Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reasonforge-expressions": {
      "command": "python",
      "args": ["-m", "reasonforge_expressions"]
    }
  }
}
```

### Standalone

```bash
python -m reasonforge_expressions
```

## Tools (15)

### Variable Management (4 tools)
- **intro**: Introduce a variable with assumptions (real, positive, etc.)
- **intro_many**: Introduce multiple variables simultaneously
- **introduce_expression**: Store an expression with a key
- **introduce_function**: Define a function symbol for differential equations

### Expression Operations (5 tools)
- **simplify_expression**: Simplify expressions using various methods
- **substitute_expression**: Substitute values into expressions
- **substitute_values**: Substitute and evaluate numerically
- **expand_expression**: Expand algebraic expressions
- **factor_expression**: Factor expressions

### Basic Calculus (4 tools)
- **differentiate**: Compute derivatives (any order)
- **integrate**: Compute indefinite integrals
- **compute_limit**: Calculate limits
- **expand_series**: Taylor/Maclaurin series expansion

### Utilities (2 tools)
- **print_latex_expression**: Get LaTeX representation
- **solve_word_problem**: Solve word problems symbolically

## Examples

### Differentiation
```
User: "What is the derivative of sin(x)*cos(x)?"
Claude: Uses differentiate tool...
Result: sin(x)*cos(x) - sin(x)^2
```

### Variable Introduction
```
User: "Create a positive real variable x"
Claude: Uses intro tool with assumptions=['real', 'positive']
Result: Variable 'x' created with assumptions: {real: True, positive: True}
```

### Series Expansion
```
User: "Taylor series for e^x around x=0, order 5"
Claude: Uses expand_series tool...
Result: 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + O(x^6)
```

## Why Choose Expressions?

Perfect for users who need:
- ✅ **Basic symbolic manipulation** - Variables, simplification, substitution
- ✅ **Fundamental calculus** - Derivatives, integrals, limits, series
- ✅ **Lightweight footprint** - Only ~15 tools vs 113 in full server
- ✅ **Fast startup** - Minimal dependencies
- ✅ **Educational use** - Students, teachers, tutoring

## Upgrade Path

Need more capabilities? Install additional ReasonForge packages:

```bash
# Add equation solving and matrices
pip install reasonforge-algebra

# Add differential equations and transforms
pip install reasonforge-analysis

# Add statistics and probability
pip install reasonforge-statistics
```

## Dependencies

- `mcp>=1.0.0` - Model Context Protocol
- `sympy>=1.12` - Symbolic mathematics
- `reasonforge>=0.1.0` - Core computation library

## License

MIT License - See LICENSE file for details

## Part of ReasonForge Ecosystem

- [reasonforge](https://github.com/yourusername/reasonforge) - Core library
- **reasonforge-expressions** - This package (essentials)
- [reasonforge-algebra](https://github.com/yourusername/reasonforge-algebra) - Equation solving, matrices
- [reasonforge-analysis](https://github.com/yourusername/reasonforge-analysis) - Advanced calculus
- [reasonforge-geometry](https://github.com/yourusername/reasonforge-geometry) - Vector/tensor calculus
- [reasonforge-statistics](https://github.com/yourusername/reasonforge-statistics) - Probability & statistics
- [reasonforge-physics](https://github.com/yourusername/reasonforge-physics) - Physics & quantum
- [reasonforge-logic](https://github.com/yourusername/reasonforge-logic) - Symbolic reasoning & logic
