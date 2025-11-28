# ReasonForge Logic

**Symbolic reasoning and formal logic MCP server - 13 tools**

An MCP (Model Context Protocol) server that provides Claude with symbolic reasoning and formal logic capabilities using SymPy's mathematical engine.

## Test Status

| Tool | Unit Test | Benchmark | Status |
|------|-----------|-----------|--------|
| pattern_to_equation | PASS | PASS | Stable |
| symbolic_knowledge_extraction | PASS | PASS | Stable |
| symbolic_theorem_proving | PASS | PASS | Stable |
| feature_extraction | PASS | PASS | Stable |
| structure_mapping | PASS | PASS | Stable |
| automated_conjecture | PASS | PASS | Stable |
| first_order_logic | PASS | PASS | Stable |
| propositional_logic_advanced | PASS | PASS | Stable |
| knowledge_graph_reasoning | PASS | PASS | Stable |
| constraint_satisfaction | PASS | PASS | Stable |
| modal_logic | PASS | PASS | Stable |
| fuzzy_logic | PASS | PASS | Stable |
| generate_proof | PASS | PASS | Stable |

**Overall: 13/13 tests passing (100%)**

## Installation

```bash
# Clone the repository
git clone https://github.com/foxintheloop/ReasonForge.git
cd ReasonForge

# Install core + logic (single command)
pip install -e packages/reasonforge -e packages/reasonforge-logic
```

## Claude Desktop Configuration

Add to your Claude Desktop config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

### Option 1: Virtual Environment (Recommended for Development)

Use this if you have a project virtual environment with dependencies installed:

**Windows:**
```json
{
  "mcpServers": {
    "reasonforge-logic": {
      "command": "C:\\path\\to\\your\\venv\\Scripts\\python.exe",
      "args": ["-m", "reasonforge_logic"],
      "env": {
        "PYTHONPATH": "C:\\path\\to\\ReasonForge\\packages\\reasonforge\\src;C:\\path\\to\\ReasonForge\\packages\\reasonforge-logic\\src"
      }
    }
  }
}
```

**macOS/Linux:**
```json
{
  "mcpServers": {
    "reasonforge-logic": {
      "command": "/path/to/your/venv/bin/python",
      "args": ["-m", "reasonforge_logic"],
      "env": {
        "PYTHONPATH": "/path/to/ReasonForge/packages/reasonforge/src:/path/to/ReasonForge/packages/reasonforge-logic/src"
      }
    }
  }
}
```

### Option 2: Global Installation

If you installed the packages globally with pip:

```bash
pip install -e packages/reasonforge -e packages/reasonforge-logic
```

Then use a simpler configuration:

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

**Note:** The `python` command must point to the Python installation where the packages are installed.

## Tools

### Pattern Recognition (5 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `pattern_to_equation` | Fit equations to data patterns | Find formula for sequence [1, 4, 9, 16, 25] |
| `symbolic_knowledge_extraction` | Extract logical rules from data | Derive rules from structured data |
| `symbolic_theorem_proving` | Prove theorems from premises | Prove goal follows from axioms |
| `feature_extraction` | Extract common features from examples | Distinguish shapes by features |
| `structure_mapping` | Find structural mappings between domains | Map relations across domains |

### Logic Systems (5 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `automated_conjecture` | Generate mathematical conjectures | Generate number theory conjectures |
| `first_order_logic` | Parse, normalize, unify FOL formulas | Parse "forall x, P(x) -> Q(x)" |
| `propositional_logic_advanced` | CNF, DNF, simplify, satisfiability | Convert "(A \| B) & C" to CNF |
| `knowledge_graph_reasoning` | Transitive closure, path finding | Find paths in knowledge graphs |
| `constraint_satisfaction` | Solve CSP problems | Solve x != y with domains |

### Specialized Logic (3 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `modal_logic` | Alethic, temporal, epistemic logic | Work with necessity/possibility |
| `fuzzy_logic` | Union, intersection, complement | Compute fuzzy set operations |
| `generate_proof` | Generate mathematical proofs | Prove commutativity of addition |

## Example Usage

Once configured, you can ask Claude:

- "Find the pattern in the sequence [2, 6, 12, 20, 30] and give me the formula"
- "Convert the formula (A OR B) AND (C OR D) to conjunctive normal form"
- "Prove that if all humans are mortal and Socrates is human, then Socrates is mortal"
- "Find the fuzzy union of sets {a: 0.7, b: 0.3} and {a: 0.4, b: 0.8}"

## Dependencies

- Python >= 3.10
- mcp >= 1.0.0
- sympy >= 1.12
- reasonforge (core library)

## Running Tests

```bash
# Unit tests
pytest packages/reasonforge-logic/tests/ -v

# Benchmark tests
cd benchmarks
python benchmark_runner.py --config config_logic_test.yaml
```

## License

MIT License
