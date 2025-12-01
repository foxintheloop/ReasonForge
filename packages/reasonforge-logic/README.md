# ReasonForge Logic

**Formal reasoning and logic MCP server - 13 tools**

An MCP (Model Context Protocol) server that provides Claude with formal reasoning and logic capabilities using SymPy's symbolic engine.

## Capabilities

- **SAT Solving** - Boolean satisfiability with SymPy's DPLL algorithm
- **Constraint Satisfaction** - CSP solver with MRV heuristic, handles 8-Queens, cryptarithmetic, graph coloring
- **Knowledge Graphs** - Transitive closure with multi-relation support
- **Propositional Logic** - CNF/DNF conversion, simplification, satisfiability
- **First-Order Logic** - Parsing, normalization, unification
- **Modal/Fuzzy Logic** - Alethic, temporal, epistemic, and fuzzy set operations

## Test Status

| Category | Tests | Status |
|----------|-------|--------|
| Pattern Recognition | 5 | ✅ Stable |
| Logic Systems | 5 | ✅ Stable |
| Specialized Logic | 3 | ✅ Stable |
| Demo Problems | 4 | ✅ Stable |
| Hard Suite (SAT/KG/CSP) | 19 | ✅ Stable |

**Overall: 36 tests covering 13 tools**

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
| `propositional_logic_advanced` | CNF, DNF, simplify, SAT solving | Check satisfiability of boolean formulas |
| `knowledge_graph_reasoning` | Transitive closure, multi-relation graphs | Infer relationships in knowledge graphs |
| `constraint_satisfaction` | CSP with MRV heuristic | Solve 8-Queens, Sudoku, cryptarithmetic |

### Specialized Logic (3 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `modal_logic` | Alethic, temporal, epistemic logic | Work with necessity/possibility |
| `fuzzy_logic` | Union, intersection, complement | Compute fuzzy set operations |
| `generate_proof` | Generate mathematical proofs | Prove commutativity of addition |

## Example Usage

Once configured, you can ask Claude:

**Pattern Recognition:**
- "Find the pattern in [2, 6, 12, 20, 30] and give me the formula"

**SAT Solving:**
- "Is this formula satisfiable: (A | B) & (~A | C) & (~B | ~C)?"
- "Can 4 pigeons fit in 3 holes with one pigeon per hole?"

**Constraint Satisfaction:**
- "Solve the 8-Queens problem"
- "Solve SEND + MORE = MONEY where each letter is a unique digit"
- "3-color the Petersen graph"

**Knowledge Graphs:**
- "Find all transitive relationships: Alice manages Bob, Bob manages Carol"

**Propositional Logic:**
- "Convert (A OR B) AND (C OR D) to conjunctive normal form"

**Proofs:**
- "Prove that if all humans are mortal and Socrates is human, then Socrates is mortal"

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
