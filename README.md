# ReasonForge - Mathematics Toolkit

> **Modular Mathematics for LLM Integration**

ReasonForge is a comprehensive ecosystem of mathematics packages built on SymPy, NumPy, SciPy, and scikit-learn, exposed via MCP (Model Context Protocol) servers. It provides 111 specialized tools across 7 focused servers, each targeting specific mathematical domains. Our goal is to achieve 100% accuracy and zero hallucinations for supported mathematical operations.

## Package Status

| Package | Tools | Status | Description |
|---------|-------|--------|-------------|
| **reasonforge-logic** | 13 | **Ready** | Symbolic reasoning and formal logic |
| reasonforge-expressions | 16 | Beta | Expression manipulation |
| reasonforge-algebra | 18 | Beta | Equation solving |
| reasonforge-analysis | 17 | Beta | Calculus and DEs |
| reasonforge-geometry | 15 | Beta | Vector/tensor calculus |
| reasonforge-statistics | 16 | Beta | Probability and statistics |
| reasonforge-physics | 16 | Beta | Physics simulations |

> **Note**: Only `reasonforge-logic` has been fully tested and documented for production use. Other packages are functional but may have edge cases not yet addressed.

**Key Features:**
- 🎯 **Modular Architecture** - Install only what you need
- 🔧 **SymPy-Powered** - Built on the proven SymPy symbolic mathematics library
- 📦 **Domain-Focused Packages** - Following industry standards (SymPy, SciPy, MATLAB)
- 🔬 **Research-Grade** - From quantum state representations to general relativity metrics
- 🎯 **Goal: Exact Computation** - Working toward 100% accuracy for supported operations

## Why ReasonForge?

ReasonForge provides **modular mathematics** for LLM applications. Instead of loading 111 tools you don't need, install exactly what your workflow requires.

## Architecture

### Core Library + 7 Specialized Servers

```
ReasonForge Ecosystem
│
├── reasonforge-core (Core Library)
│   └── Pure Python symbolic computation engine (no MCP)
│
├── reasonforge-expressions (16 tools) ⭐ ESSENTIALS
│   └── Variable management, expression operations, basic calculus
│
├── reasonforge-algebra (18 tools)
│   └── Equation solving, matrices, optimization
│
├── reasonforge-analysis (17 tools)
│   └── Differential equations, transforms, signal processing
│
├── reasonforge-geometry (15 tools)
│   └── Vector/tensor calculus, general relativity, visualization
│
├── reasonforge-statistics (16 tools)
│   └── Probability, statistics, data science
│
├── reasonforge-physics (16 tools)
│   └── Classical mechanics, quantum computing, electromagnetism
│
└── reasonforge-logic (13 tools) ✅ READY
    └── Symbolic reasoning, formal logic, knowledge systems
```

**Total: 111 tools across 7 modular servers**

## Quick Start (reasonforge-logic)

The production-ready MCP server with 13 tools for symbolic reasoning and formal logic.

### Installation

```bash
git clone https://github.com/foxintheloop/ReasonForge.git
cd ReasonForge
pip install -e packages/reasonforge -e packages/reasonforge-logic
```

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

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

### Example Prompts

- "Find the pattern in [1, 4, 9, 16, 25] and give me the equation"
- "Convert (A OR B) AND C to conjunctive normal form"
- "Prove that a + b = b + a using commutativity"

## Package Documentation

| Package | Tools | Status | Documentation |
|---------|-------|--------|---------------|
| **reasonforge-logic** | 13 | ✅ Ready | [README](packages/reasonforge-logic/README.md) |
| reasonforge-expressions | 16 | Beta | [README](packages/reasonforge-expressions/README.md) |
| reasonforge-algebra | 18 | Beta | [README](packages/reasonforge-algebra/README.md) |
| reasonforge-analysis | 17 | Beta | [README](packages/reasonforge-analysis/README.md) |
| reasonforge-geometry | 15 | Beta | [README](packages/reasonforge-geometry/README.md) |
| reasonforge-statistics | 16 | Beta | [README](packages/reasonforge-statistics/README.md) |
| reasonforge-physics | 16 | Beta | [README](packages/reasonforge-physics/README.md) |

## Usage Examples

Comprehensive usage examples for all 110 tools are available:

- **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Index with quick links by category
- **Individual package examples:**
  - [reasonforge-expressions](docs/usage-examples/reasonforge-expressions-examples.md) - Examples 1-15
  - [reasonforge-algebra](docs/usage-examples/reasonforge-algebra-examples.md) - Examples 16-33
  - [reasonforge-analysis](docs/usage-examples/reasonforge-analysis-examples.md) - Examples 34-50
  - [reasonforge-geometry](docs/usage-examples/reasonforge-geometry-examples.md) - Examples 51-65
  - [reasonforge-statistics](docs/usage-examples/reasonforge-statistics-examples.md) - Examples 66-81
  - [reasonforge-physics](docs/usage-examples/reasonforge-physics-examples.md) - Examples 82-97
  - [reasonforge-logic](docs/usage-examples/reasonforge-logic-examples.md) - Examples 98-110

Each example shows a natural language user query and the corresponding tool usage with sample output.

## Design Goals

### What We're Working Toward

| Goal | Status |
|------|--------|
| Exact symbolic computation | Implemented via SymPy |
| Deterministic results | Achieved for most operations |
| Reproducibility | Achieved |
| Full coverage of mathematical domains | In progress |

## Contributing

Contributions welcome! Each package is independent, making it easy to:
- Add new tools to existing packages
- Create new domain-specific packages
- Improve documentation and examples

### Running Tests

```bash
# Run all tests
pytest packages/

# Run tests for a specific package
pytest packages/reasonforge-logic/tests/ -v
```

## License

MIT License - See [LICENSE](LICENSE) file for details

## Citation

If you use ReasonForge in research or production:

```bibtex
@software{reasonforge_ecosystem,
  title = {ReasonForge: Mathematics Toolkit},
  author = {Derek Fox},
  year = {2025},
  description = {Modular MCP server ecosystem with 111 mathematics tools across 7 packages},
  url = {https://github.com/foxintheloop/reasonforge},
  note = {SymPy-powered computation for LLM integration}
}
```

## Acknowledgments

- **SymPy Team**: World-class symbolic mathematics library
- **Anthropic**: Model Context Protocol specification and Claude
- **MCP Community**: Tools, examples, and collaborative development
- **Open Source Contributors**: SciPy, NumPy, Matplotlib, and more
