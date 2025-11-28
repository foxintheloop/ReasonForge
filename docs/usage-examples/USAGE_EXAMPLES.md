# ReasonForge - Complete Usage Examples

This document provides comprehensive usage examples for all 110 tools available across the ReasonForge modular ecosystem. Each example shows a natural language user query and the corresponding tool usage with sample output.

## Modular Architecture

ReasonForge is organized into 7 specialized MCP server packages plus a core library:

| Package | Tools | Description |
|---------|-------|-------------|
| [reasonforge-expressions](./docs/usage-examples/reasonforge-expressions-examples.md) | 15 | Essential tools for symbolic manipulation, basic calculus ⭐ |
| [reasonforge-algebra](./docs/usage-examples/reasonforge-algebra-examples.md) | 18 | Equation solving, matrices, optimization |
| [reasonforge-analysis](./docs/usage-examples/reasonforge-analysis-examples.md) | 17 | Differential equations, transforms, signal processing |
| [reasonforge-geometry](./docs/usage-examples/reasonforge-geometry-examples.md) | 15 | Vector/tensor calculus, general relativity, visualization |
| [reasonforge-statistics](./docs/usage-examples/reasonforge-statistics-examples.md) | 16 | Probability, inference, data science |
| [reasonforge-physics](./docs/usage-examples/reasonforge-physics-examples.md) | 16 | Classical mechanics, quantum computing, EM |
| [reasonforge-logic](./docs/usage-examples/reasonforge-logic-examples.md) | 13 | Symbolic AI, formal logic, knowledge systems |

**Total: 110 tools across 7 packages**

---

## Quick Links by Category

### Mathematics
- [Variable Management](./docs/usage-examples/reasonforge-expressions-examples.md#variable-management-4-tools) (4 tools)
- [Expression Operations](./docs/usage-examples/reasonforge-expressions-examples.md#expression-operations-5-tools) (5 tools)
- [Basic Calculus](./docs/usage-examples/reasonforge-expressions-examples.md#basic-calculus-4-tools) (4 tools)
- [Equation Solving](./docs/usage-examples/reasonforge-algebra-examples.md#equation-solving-4-tools) (4 tools)
- [Matrix Operations](./docs/usage-examples/reasonforge-algebra-examples.md#matrix-operations-5-tools) (5 tools)
- [Optimization](./docs/usage-examples/reasonforge-algebra-examples.md#optimization-6-tools) (6 tools)

### Analysis & Transforms
- [Differential Equations](./docs/usage-examples/reasonforge-analysis-examples.md#differential-equations-3-tools) (3 tools)
- [Physics PDEs](./docs/usage-examples/reasonforge-analysis-examples.md#physics-pdes-3-tools) (3 tools)
- [Transforms](./docs/usage-examples/reasonforge-analysis-examples.md#transforms-5-tools) (5 tools)
- [Signal Processing](./docs/usage-examples/reasonforge-analysis-examples.md#signal-processing-2-tools) (2 tools)

### Geometry & Physics
- [Vector Calculus](./docs/usage-examples/reasonforge-geometry-examples.md#vector-calculus-5-tools) (5 tools)
- [Tensor Calculus & GR](./docs/usage-examples/reasonforge-geometry-examples.md#tensor-calculus--gr-5-tools) (5 tools)
- [Classical Mechanics](./docs/usage-examples/reasonforge-physics-examples.md#classical-mechanics-3-tools) (3 tools)
- [Quantum Computing](./docs/usage-examples/reasonforge-physics-examples.md#quantum-computing-10-tools) (10 tools)

### Statistics & Data Science
- [Probability](./docs/usage-examples/reasonforge-statistics-examples.md#probability-3-tools) (3 tools)
- [Inference](./docs/usage-examples/reasonforge-statistics-examples.md#inference-4-tools) (4 tools)
- [Regression](./docs/usage-examples/reasonforge-statistics-examples.md#regression-2-tools) (2 tools)
- [Data Science](./docs/usage-examples/reasonforge-statistics-examples.md#data-science-7-tools) (7 tools)

### Logic & AI
- [Symbolic AI](./docs/usage-examples/reasonforge-logic-examples.md#symbolic-ai-6-tools) (6 tools)
- [Formal Logic](./docs/usage-examples/reasonforge-logic-examples.md#formal-logic-4-tools) (4 tools)
- [Knowledge Systems](./docs/usage-examples/reasonforge-logic-examples.md#knowledge-systems-2-tools) (2 tools)
- [Proof Generation](./docs/usage-examples/reasonforge-logic-examples.md#proof-generation-1-tool) (1 tool)

---

## Installation

All tools provide exact symbolic results with LaTeX formatting for mathematical expressions.

**Install only what you need:**
```bash
# Essentials only
pip install reasonforge-expressions

# Data science workflow
pip install reasonforge-expressions reasonforge-statistics

# Physics research
pip install reasonforge-expressions reasonforge-algebra reasonforge-physics

# Everything
pip install reasonforge-expressions reasonforge-algebra reasonforge-analysis \
    reasonforge-geometry reasonforge-statistics reasonforge-physics reasonforge-logic
```

---

## Example Files

Each package has its own detailed examples file:

1. **[reasonforge-expressions-examples.md](./docs/usage-examples/reasonforge-expressions-examples.md)** - Examples 1-15
2. **[reasonforge-algebra-examples.md](./docs/usage-examples/reasonforge-algebra-examples.md)** - Examples 16-33
3. **[reasonforge-analysis-examples.md](./docs/usage-examples/reasonforge-analysis-examples.md)** - Examples 34-50
4. **[reasonforge-geometry-examples.md](./docs/usage-examples/reasonforge-geometry-examples.md)** - Examples 51-65
5. **[reasonforge-statistics-examples.md](./docs/usage-examples/reasonforge-statistics-examples.md)** - Examples 66-81
6. **[reasonforge-physics-examples.md](./docs/usage-examples/reasonforge-physics-examples.md)** - Examples 82-97
7. **[reasonforge-logic-examples.md](./docs/usage-examples/reasonforge-logic-examples.md)** - Examples 98-110
