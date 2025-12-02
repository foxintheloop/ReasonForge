# ReasonForge

**Exact math for LLMs. No hallucinations.**

ReasonForge is a modular mathematics toolkit that integrates with Claude and other LLMs via MCP (Model Context Protocol). Instead of letting the LLM guess at calculations, ReasonForge routes mathematical operations through SymPy, NumPy, and SciPy — returning provably correct results.

111 tools. 7 domain-specific servers. Zero probabilistic math.

## The Problem

Ask an LLM to solve a differential equation, simplify a complex expression, or find a sequence pattern, and you'll get plausible-looking answers that are wrong 15-30% of the time. For research, finance, or engineering applications, "usually right" isn't good enough.

## The Solution

ReasonForge gives your LLM access to a real computation engine. The LLM handles natural language understanding and problem decomposition; ReasonForge handles the actual math.

```
User: "Find the pattern in [1, 4, 9, 16, 25] and give me the equation"

Claude + ReasonForge:
  → Calls find_sequence_pattern tool
  → Returns: n² (exact symbolic formula, not a guess)
```

## Quick Start

```bash
git clone https://github.com/foxintheloop/ReasonForge.git
cd ReasonForge
pip install -e packages/reasonforge -e packages/reasonforge-logic
```

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "reasonforge-logic": {
      "command": "python",
      "args": ["-m", "reasonforge_logic"]
    }
  }
}
```

Done. Claude now has access to 13 symbolic reasoning tools.

## Example Queries

- *"Convert (A OR B) AND C to conjunctive normal form"*
- *"Prove that a + b = b + a using commutativity"*
- *"Solve the system: 2x + y = 5, x - y = 1"*
- *"Find the derivative of x³ sin(x)"*

## Architecture

Install only what you need:

```
ReasonForge Ecosystem
│
├── reasonforge-core          # Pure Python symbolic engine (no MCP)
│
├── reasonforge-logic         # 13 tools ✅ Production Ready
│   └── Symbolic reasoning, formal logic, knowledge systems
│
├── reasonforge-expressions   # 16 tools (Beta)
│   └── Variable management, expression operations, basic calculus
│
├── reasonforge-algebra       # 18 tools (Beta)
│   └── Equation solving, matrices, optimization
│
├── reasonforge-analysis      # 17 tools (Beta)
│   └── Differential equations, transforms, signal processing
│
├── reasonforge-geometry      # 15 tools (Beta)
│   └── Vector/tensor calculus, coordinate systems
│
├── reasonforge-statistics    # 16 tools (Beta)
│   └── Probability, distributions, hypothesis testing
│
└── reasonforge-physics       # 16 tools (Beta)
    └── Classical mechanics, quantum states, electromagnetism
```

**Why modular?** A monolithic server loads ~8,000 tokens of tool definitions into context. A single package loads ~800-1,200 — an 85-90% reduction.

## Package Status

| Package | Tools | Status | Use Case |
|---------|-------|--------|----------|
| reasonforge-logic | 13 | ✅ Ready | Formal proofs, boolean logic, pattern recognition |
| reasonforge-expressions | 16 | Beta | Expression simplification, substitution, basic calculus |
| reasonforge-algebra | 18 | Beta | Linear systems, polynomial roots, matrix operations |
| reasonforge-analysis | 17 | Beta | ODEs, PDEs, Laplace/Fourier transforms |
| reasonforge-geometry | 15 | Beta | Vectors, tensors, coordinate transforms |
| reasonforge-statistics | 16 | Beta | Distributions, inference, regression |
| reasonforge-physics | 16 | Beta | Mechanics, E&M, quantum computing |

> **Note:** Only `reasonforge-logic` is fully tested for production. Other packages are functional but may have edge cases.

## Design Goals

| Goal | Status |
|------|--------|
| Exact symbolic computation | ✅ Implemented via SymPy |
| Deterministic results | ✅ Achieved for supported operations |
| Reproducibility | ✅ Same input → same output |
| Full domain coverage | In progress |

## Documentation

- [Usage Examples](USAGE_EXAMPLES.md) — 110 examples across all packages
- Individual package READMEs in `/packages/`

## Contributing

```bash
# Run all tests
pytest packages/

# Run specific package tests
pytest packages/reasonforge-logic/tests/ -v
```

PRs welcome for new tools, packages, or docs.

## License

MIT — see [LICENSE](LICENSE)

## Citation

```bibtex
@software{reasonforge,
  title = {ReasonForge: Mathematics Toolkit for LLM Integration},
  author = {Derek Fox},
  year = {2025},
  url = {https://github.com/foxintheloop/ReasonForge}
}
```

---

Built on [SymPy](https://www.sympy.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and the [Model Context Protocol](https://modelcontextprotocol.io/).
