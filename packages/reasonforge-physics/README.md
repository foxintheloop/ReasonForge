# ReasonForge Physics

**Classical mechanics, electromagnetism, and quantum computing - 16 tools**

An MCP (Model Context Protocol) server that provides Claude with physics simulation and quantum computing capabilities using SymPy's symbolic engine.

> **Beta Status**: This package is functional but still undergoing testing. Some edge cases may not be fully covered. Please report issues on [GitHub](https://github.com/foxintheloop/ReasonForge/issues).

## Capabilities

- **Classical Mechanics** - Lagrangian and Hamiltonian formulations, Noether's theorem
- **Electromagnetism** - Maxwell's equations in differential form
- **Special Relativity** - Lorentz transformations and relativistic dynamics
- **Quantum Computing** - Quantum states, gates, circuits, and measurements

## Installation

```bash
pip install reasonforge-physics
```

Or install from source:

```bash
git clone https://github.com/foxintheloop/ReasonForge.git
cd ReasonForge
pip install -e packages/reasonforge -e packages/reasonforge-physics
```

## Claude Desktop Configuration

Add to your Claude Desktop config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reasonforge-physics": {
      "command": "python",
      "args": ["-m", "reasonforge_physics"]
    }
  }
}
```

## Tools

### Classical Mechanics (3 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `lagrangian_mechanics` | Derive equations of motion from Lagrangian | Pendulum, double pendulum |
| `hamiltonian_mechanics` | Hamiltonian formulation and phase space | Canonical transformations |
| `noether_theorem` | Find conserved quantities from symmetries | Energy, momentum conservation |

### Electromagnetism & Relativity (2 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `maxwell_equations` | Work with Maxwell's equations | Field calculations |
| `special_relativity` | Lorentz transformations and 4-vectors | Time dilation, length contraction |

### Quantum Computing (10 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `create_quantum_state` | Create quantum state vectors | \|0>, \|1>, \|+>, \|-> states |
| `quantum_gate_operations` | Apply quantum gates | X, Y, Z, H, CNOT, etc. |
| `tensor_product_states` | Compose multi-qubit states | \|00>, \|01>, Bell states |
| `quantum_entanglement_measure` | Measure entanglement | Concurrence, entanglement entropy |
| `quantum_circuit_symbolic` | Build symbolic quantum circuits | Gate sequences |
| `quantum_measurement` | Perform measurements | Projective measurements |
| `quantum_fidelity` | Calculate state fidelity | Compare quantum states |
| `pauli_matrices` | Work with Pauli operators | Spin-1/2 systems |
| `commutator_anticommutator` | Compute [A,B] and {A,B} | Operator algebra |
| `quantum_evolution` | Time evolution of states | Schrodinger dynamics |

### Optimization (1 tool)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `symbolic_optimization_setup` | Set up optimization problems | Variational methods |

## Example Usage

Once configured, you can ask Claude:

**Classical Mechanics:**
- "Derive the equations of motion for a simple pendulum using Lagrangian mechanics"
- "Find the Hamiltonian for a harmonic oscillator"
- "What quantity is conserved due to time translation symmetry?"

**Electromagnetism:**
- "Write out Maxwell's equations in differential form"
- "Calculate the Lorentz transformation for velocity v = 0.8c"

**Quantum Computing:**
- "Create the Bell state (|00> + |11>)/sqrt(2)"
- "Apply a Hadamard gate followed by CNOT to |00>"
- "Calculate the entanglement entropy of this two-qubit state"
- "What is the fidelity between |+> and |0>?"

## Dependencies

- Python >= 3.10
- mcp >= 1.0.0
- sympy >= 1.12
- numpy >= 1.24.0
- reasonforge (core library)

## Running Tests

```bash
pytest packages/reasonforge-physics/tests/ -v
```

## License

MIT License - See [LICENSE](../../LICENSE) for details.

## Related Packages

- [reasonforge](https://pypi.org/project/reasonforge/) - Core symbolic computation library
- [reasonforge-geometry](https://pypi.org/project/reasonforge-geometry/) - Tensor calculus for general relativity
- [reasonforge-analysis](https://pypi.org/project/reasonforge-analysis/) - Differential equations for physics
- [reasonforge-algebra](https://pypi.org/project/reasonforge-algebra/) - Matrix operations for quantum mechanics
