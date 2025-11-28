# reasonforge-physics (16 tools)

**Classical mechanics, quantum computing, and electromagnetism**

**Installation:** `pip install reasonforge-physics`

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-statistics](./reasonforge-statistics-examples.md) | [Next: reasonforge-logic](./reasonforge-logic-examples.md)

---

## Classical Mechanics (3 tools)

### Example 82: Lagrangian Mechanics

**User:** "Derive equations of motion for a simple pendulum"

**Claude with ReasonForge:**
```
Using lagrangian_mechanics tool from reasonforge-physics...

Result:
{
  "lagrangian": "m*l^2*theta_dot^2/2 - m*g*l*(1 - cos(theta))",
  "generalized_coordinates": ["theta"],
  "euler_lagrange": "m*l^2*theta_ddot + m*g*l*sin(theta) = 0",
  "equation_of_motion": "theta_ddot + (g/l)*sin(theta) = 0",
  "interpretation": "Nonlinear pendulum equation"
}
```

### Example 83: Hamiltonian Mechanics

**User:** "Derive Hamilton's equations for a harmonic oscillator"

**Claude with ReasonForge:**
```
Using hamiltonian_mechanics tool from reasonforge-physics...

Result:
{
  "operation": "hamilton_equations",
  "hamiltonian": "p^2/(2*m) + k*x^2/2",
  "coordinates": ["x"],
  "momenta": ["p"],
  "equations": {
    "dx/dt": "p/m",
    "dp/dt": "-k*x"
  },
  "poisson_brackets": "{x, p} = 1"
}
```

### Example 84: Noether Theorem

**User:** "Find the conserved quantity from time translation symmetry"

**Claude with ReasonForge:**
```
Using noether_theorem tool from reasonforge-physics...

Result:
{
  "lagrangian": "L(q, q_dot)",
  "symmetry_transformation": "t -> t + epsilon",
  "conserved_quantity": "H = Sum(p_i*q_dot_i) - L",
  "interpretation": "Energy conservation (Hamiltonian)",
  "noether_current": "dL/dt = 0 => H = const"
}
```

## Electromagnetism & Relativity (2 tools)

### Example 85: Maxwell Equations

**User:** "Derive electromagnetic wave equation from Maxwell's equations"

**Claude with ReasonForge:**
```
Using maxwell_equations tool from reasonforge-physics...

Result:
{
  "operation": "derive_wave",
  "maxwell_equations": {
    "gauss": "div(E) = rho/epsilon_0",
    "gauss_magnetism": "div(B) = 0",
    "faraday": "curl(E) = -dB/dt",
    "ampere": "curl(B) = mu_0*J + mu_0*epsilon_0*dE/dt"
  },
  "wave_equation": "nabla^2 E - mu_0*epsilon_0*d^2E/dt^2 = 0",
  "wave_speed": "c = 1/sqrt(mu_0*epsilon_0)"
}
```

### Example 86: Special Relativity

**User:** "Calculate Lorentz transformation for velocity v=0.6c"

**Claude with ReasonForge:**
```
Using special_relativity tool from reasonforge-physics...

Result:
{
  "operation": "lorentz_transform",
  "velocity": "0.6*c",
  "gamma": "1.25",
  "transformation": {
    "t'": "gamma*(t - v*x/c^2)",
    "x'": "gamma*(x - v*t)",
    "y'": "y",
    "z'": "z"
  },
  "time_dilation": "Δt' = γ*Δt",
  "length_contraction": "L' = L/γ"
}
```

## Quantum Computing (10 tools)

### Example 87: Create Quantum State

**User:** "Create a pure quantum state |ψ⟩ = (|0⟩ + |1⟩)/√2"

**Claude with ReasonForge:**
```
Using create_quantum_state tool from reasonforge-physics...

Result:
{
  "state_type": "pure",
  "num_qubits": 1,
  "amplitudes": ["1/sqrt(2)", "1/sqrt(2)"],
  "state_key": "state_1",
  "normalization": "1.0",
  "latex": "\\frac{1}{\\sqrt{2}}(|0\\rangle + |1\\rangle)"
}
```

### Example 88: Quantum Gate Operations

**User:** "Apply Hadamard gate to |0⟩"

**Claude with ReasonForge:**
```
Using quantum_gate_operations tool from reasonforge-physics...

Result:
{
  "state_key": "state_0",
  "gates": ["Hadamard"],
  "initial_state": "|0⟩",
  "final_state": "(|0⟩ + |1⟩)/sqrt(2)",
  "latex": "H|0\\rangle = \\frac{1}{\\sqrt{2}}(|0\\rangle + |1\\rangle)"
}
```

### Example 89: Tensor Product States

**User:** "Create tensor product of |0⟩ and |1⟩"

**Claude with ReasonForge:**
```
Using tensor_product_states tool from reasonforge-physics...

Result:
{
  "state_keys": ["state_0", "state_1"],
  "states": ["|0⟩", "|1⟩"],
  "tensor_product": "|01⟩",
  "num_qubits": 2,
  "latex": "|0\\rangle \\otimes |1\\rangle = |01\\rangle"
}
```

### Example 90: Quantum Entanglement Measure

**User:** "Calculate entanglement entropy of Bell state (|00⟩ + |11⟩)/√2"

**Claude with ReasonForge:**
```
Using quantum_entanglement_measure tool from reasonforge-physics...

Result:
{
  "state_key": "bell_state",
  "measure_type": "entropy",
  "subsystem_partition": [[0], [1]],
  "entanglement_entropy": "1.0",
  "interpretation": "Maximally entangled state",
  "latex": "S = 1 \\text{ bit}"
}
```

### Example 91: Quantum Circuit Symbolic

**User:** "Build a quantum circuit with H on qubit 0, then CNOT(0,1)"

**Claude with ReasonForge:**
```
Using quantum_circuit_symbolic tool from reasonforge-physics...

Result:
{
  "num_qubits": 2,
  "gate_sequence": [
    {"gate": "H", "target": 0},
    {"gate": "CNOT", "control": 0, "target": 1}
  ],
  "initial_state": "|00⟩",
  "final_state": "(|00⟩ + |11⟩)/sqrt(2)",
  "circuit_description": "Bell state preparation"
}
```

### Example 92: Quantum Measurement

**User:** "Measure Bell state in computational basis"

**Claude with ReasonForge:**
```
Using quantum_measurement tool from reasonforge-physics...

Result:
{
  "state_key": "bell_state",
  "measurement_basis": "computational",
  "probabilities": {
    "|00⟩": "1/2",
    "|11⟩": "1/2"
  },
  "post_measurement_states": {
    "|00⟩": "|00⟩",
    "|11⟩": "|11⟩"
  }
}
```

### Example 93: Quantum Fidelity

**User:** "Calculate fidelity between |0⟩ and (|0⟩ + |1⟩)/√2"

**Claude with ReasonForge:**
```
Using quantum_fidelity tool from reasonforge-physics...

Result:
{
  "state_key_1": "state_0",
  "state_key_2": "state_plus",
  "fidelity": "1/sqrt(2)",
  "numerical": "0.7071",
  "latex": "F = \\frac{1}{\\sqrt{2}}"
}
```

### Example 94: Pauli Matrices

**User:** "Generate Pauli X matrix"

**Claude with ReasonForge:**
```
Using pauli_matrices tool from reasonforge-physics...

Result:
{
  "operation": "single",
  "matrix": "X",
  "result": "[[0, 1], [1, 0]]",
  "latex": "\\sigma_x = \\begin{bmatrix} 0 & 1 \\\\ 1 & 0 \\end{bmatrix}"
}
```

### Example 95: Commutator Anticommutator

**User:** "Calculate [X, Y] for Pauli matrices"

**Claude with ReasonForge:**
```
Using commutator_anticommutator tool from reasonforge-physics...

Result:
{
  "operator_1": "X",
  "operator_2": "Y",
  "operation": "commutator",
  "result": "2*I*Z",
  "latex": "[X, Y] = 2iZ"
}
```

### Example 96: Quantum Evolution

**User:** "Evolve |0⟩ under Hamiltonian H = σ_x for time t"

**Claude with ReasonForge:**
```
Using quantum_evolution tool from reasonforge-physics...

Result:
{
  "state_key": "state_0",
  "hamiltonian": "sigma_x",
  "time": "t",
  "method": "schrodinger",
  "evolved_state": "cos(t)|0⟩ - I*sin(t)|1⟩",
  "latex": "e^{-i\\sigma_x t}|0\\rangle = \\cos(t)|0\\rangle - i\\sin(t)|1\\rangle"
}
```

## Optimization (1 tool)

### Example 97: Symbolic Optimization Setup

**User:** "Set up optimization problem: minimize x^2 + y^2 subject to x + y = 1"

**Claude with ReasonForge:**
```
Using symbolic_optimization_setup tool from reasonforge-physics...

Result:
{
  "objective": "x**2 + y**2",
  "equality_constraints": ["x + y - 1"],
  "inequality_constraints": [],
  "variables": ["x", "y"],
  "lagrangian": "x**2 + y**2 - lambda*(x + y - 1)",
  "kkt_conditions": [
    "2*x - lambda = 0",
    "2*y - lambda = 0",
    "x + y - 1 = 0"
  ],
  "stationary_points": [{"x": "1/2", "y": "1/2", "lambda": "1"}]
}
```

---

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-statistics](./reasonforge-statistics-examples.md) | [Next: reasonforge-logic](./reasonforge-logic-examples.md)
