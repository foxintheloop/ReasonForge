"""ReasonForge Physics - Physics and Quantum Computing

16 tools for physics, quantum mechanics, and quantum computing.
"""

from typing import Dict, Any

import sympy as sp
from sympy import (
    symbols, Matrix, latex, simplify, diff, exp, I, sqrt,
    conjugate, eye, cos, sin
)
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.quantum import TensorProduct, Dagger

from reasonforge import (
    BaseReasonForgeServer,
    SymbolicAI,
    create_input_schema,
    safe_sympify,
    validate_variable_name,
    validate_expression_string,
    validate_list_input,
    ValidationError,
)


class PhysicsServer(BaseReasonForgeServer):
    """MCP server for physics and quantum computing calculations."""

    def __init__(self):
        super().__init__("reasonforge-physics")
        self.ai = SymbolicAI()

    def register_tools(self):
        """Register all physics tools."""

        # Classical Mechanics
        self.add_tool(
            name="lagrangian_mechanics",
            description="Derive equations of motion from Lagrangian.",
            handler=self.handle_lagrangian_mechanics,
            input_schema=create_input_schema(
                properties={
                    "lagrangian": {
                        "type": "string",
                        "description": "Lagrangian expression L = T - V"
                    },
                    "generalized_coordinates": {
                        "type": "array",
                        "description": "List of generalized coordinates"
                    }
                },
                required=["lagrangian"]
            )
        )

        self.add_tool(
            name="hamiltonian_mechanics",
            description="Work with Hamiltonian mechanics.",
            handler=self.handle_hamiltonian_mechanics,
            input_schema=create_input_schema(
                properties={
                    "operation": {
                        "type": "string",
                        "description": "Operation (derive, poisson_bracket)"
                    },
                    "hamiltonian": {
                        "type": "string",
                        "description": "Hamiltonian expression H = T + V"
                    }
                },
                required=["hamiltonian"]
            )
        )

        self.add_tool(
            name="noether_theorem",
            description="Find conserved quantities from symmetries.",
            handler=self.handle_noether_theorem,
            input_schema=create_input_schema(
                properties={
                    "lagrangian": {
                        "type": "string",
                        "description": "Lagrangian expression"
                    },
                    "symmetry_transformation": {
                        "type": "string",
                        "description": "Symmetry transformation type"
                    }
                },
                required=["lagrangian"]
            )
        )

        self.add_tool(
            name="maxwell_equations",
            description="Work with Maxwell's equations.",
            handler=self.handle_maxwell_equations,
            input_schema=create_input_schema(
                properties={
                    "operation": {
                        "type": "string",
                        "description": "Operation (vacuum, wave_equation, energy_momentum)"
                    }
                },
                required=["operation"]
            )
        )

        self.add_tool(
            name="special_relativity",
            description="Special relativity calculations.",
            handler=self.handle_special_relativity,
            input_schema=create_input_schema(
                properties={
                    "operation": {
                        "type": "string",
                        "description": "Operation (lorentz_factor, time_dilation, length_contraction, relativistic_energy)"
                    },
                    "velocity": {
                        "type": "string",
                        "description": "Velocity expression"
                    }
                },
                required=["operation"]
            )
        )

        # Quantum Computing
        self.add_tool(
            name="create_quantum_state",
            description="Create quantum state vector.",
            handler=self.handle_create_quantum_state,
            input_schema=create_input_schema(
                properties={
                    "state_type": {
                        "type": "string",
                        "description": "State type (pure)"
                    },
                    "num_qubits": {
                        "type": "integer",
                        "description": "Number of qubits"
                    },
                    "amplitudes": {
                        "type": "array",
                        "description": "State amplitudes"
                    }
                },
                required=["state_type", "num_qubits"]
            )
        )

        self.add_tool(
            name="quantum_gate_operations",
            description="Apply quantum gates to states.",
            handler=self.handle_quantum_gate_operations,
            input_schema=create_input_schema(
                properties={
                    "state_key": {
                        "type": "string",
                        "description": "State key to apply gates to"
                    },
                    "gates": {
                        "type": "array",
                        "description": "List of gate names (H, X, Y, Z, CNOT)"
                    }
                },
                required=["state_key", "gates"]
            )
        )

        self.add_tool(
            name="tensor_product_states",
            description="Compute tensor product of quantum states.",
            handler=self.handle_tensor_product_states,
            input_schema=create_input_schema(
                properties={
                    "state_keys": {
                        "type": "array",
                        "description": "List of state keys to tensor"
                    }
                },
                required=["state_keys"]
            )
        )

        self.add_tool(
            name="quantum_entanglement_measure",
            description="Measure quantum entanglement.",
            handler=self.handle_quantum_entanglement_measure,
            input_schema=create_input_schema(
                properties={
                    "state_key": {
                        "type": "string",
                        "description": "State key"
                    },
                    "measure_type": {
                        "type": "string",
                        "description": "Measure type (schmidt, entropy)"
                    }
                },
                required=["state_key", "measure_type"]
            )
        )

        self.add_tool(
            name="quantum_circuit_symbolic",
            description="Build symbolic quantum circuits.",
            handler=self.handle_quantum_circuit_symbolic,
            input_schema=create_input_schema(
                properties={
                    "num_qubits": {
                        "type": "integer",
                        "description": "Number of qubits"
                    },
                    "gate_sequence": {
                        "type": "array",
                        "description": "Sequence of gates"
                    }
                },
                required=["num_qubits"]
            )
        )

        self.add_tool(
            name="quantum_measurement",
            description="Perform quantum measurements.",
            handler=self.handle_quantum_measurement,
            input_schema=create_input_schema(
                properties={
                    "state_key": {
                        "type": "string",
                        "description": "State key to measure"
                    },
                    "measurement_basis": {
                        "type": "string",
                        "description": "Measurement basis"
                    }
                },
                required=["state_key"]
            )
        )

        self.add_tool(
            name="quantum_fidelity",
            description="Calculate fidelity between quantum states.",
            handler=self.handle_quantum_fidelity,
            input_schema=create_input_schema(
                properties={
                    "state_key_1": {
                        "type": "string",
                        "description": "First state key"
                    },
                    "state_key_2": {
                        "type": "string",
                        "description": "Second state key"
                    }
                },
                required=["state_key_1", "state_key_2"]
            )
        )

        self.add_tool(
            name="pauli_matrices",
            description="Work with Pauli matrices.",
            handler=self.handle_pauli_matrices,
            input_schema=create_input_schema(
                properties={
                    "operation": {
                        "type": "string",
                        "description": "Operation (get, properties)"
                    },
                    "matrix": {
                        "type": "string",
                        "description": "Matrix name (X, Y, Z, I)"
                    }
                },
                required=["operation"]
            )
        )

        self.add_tool(
            name="commutator_anticommutator",
            description="Calculate commutators and anticommutators.",
            handler=self.handle_commutator_anticommutator,
            input_schema=create_input_schema(
                properties={
                    "operator_1": {
                        "type": "string",
                        "description": "First operator"
                    },
                    "operator_2": {
                        "type": "string",
                        "description": "Second operator"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation (commutator, anticommutator)"
                    }
                },
                required=["operator_1", "operator_2", "operation"]
            )
        )

        self.add_tool(
            name="quantum_evolution",
            description="Evolve quantum state under Hamiltonian.",
            handler=self.handle_quantum_evolution,
            input_schema=create_input_schema(
                properties={
                    "state_key": {
                        "type": "string",
                        "description": "State key"
                    },
                    "hamiltonian": {
                        "type": "string",
                        "description": "Hamiltonian operator"
                    },
                    "time": {
                        "type": "string",
                        "description": "Evolution time"
                    }
                },
                required=["state_key", "hamiltonian"]
            )
        )

        self.add_tool(
            name="symbolic_optimization_setup",
            description="Set up physics optimization problems.",
            handler=self.handle_symbolic_optimization_setup,
            input_schema=create_input_schema(
                properties={
                    "objective": {
                        "type": "string",
                        "description": "Objective function"
                    },
                    "variables": {
                        "type": "array",
                        "description": "Optimization variables"
                    }
                },
                required=["objective"]
            )
        )

    def handle_lagrangian_mechanics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle lagrangian_mechanics."""
        lagrangian_str = validate_expression_string(arguments["lagrangian"])
        generalized_coords = arguments.get("generalized_coordinates", ["q"])

        # Validate coordinate names
        generalized_coords = [validate_variable_name(coord) for coord in generalized_coords]

        # Parse Lagrangian - use sp.sympify directly as we need time derivatives
        L = sp.sympify(lagrangian_str)

        # Create time-dependent generalized coordinates
        t = symbols('t')
        q_syms = [dynamicsymbols(coord) for coord in generalized_coords]

        # Euler-Lagrange equations: d/dt(∂L/∂q̇) - ∂L/∂q = 0
        equations_of_motion = []
        for q in q_syms:
            q_dot = diff(q, t)
            # ∂L/∂q̇
            dL_dqdot = diff(L, q_dot)
            # d/dt(∂L/∂q̇)
            d_dt_dL_dqdot = diff(dL_dqdot, t)
            # ∂L/∂q
            dL_dq = diff(L, q)
            # Euler-Lagrange equation
            eom = d_dt_dL_dqdot - dL_dq
            equations_of_motion.append(eom)

        return {
            "lagrangian": lagrangian_str,
            "generalized_coordinates": generalized_coords,
            "equations_of_motion": [str(eq) for eq in equations_of_motion],
            "latex": [latex(eq) for eq in equations_of_motion],
            "formula": "d/dt(∂L/∂q̇) - ∂L/∂q = 0",
            "result": [str(eq) for eq in equations_of_motion]
        }

    def handle_hamiltonian_mechanics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hamiltonian_mechanics."""
        operation = arguments.get("operation", "derive")
        hamiltonian_str = validate_expression_string(arguments["hamiltonian"])

        H = safe_sympify(hamiltonian_str)

        if operation == "derive":
            # Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q
            q = symbols('q', real=True)
            p = symbols('p', real=True)

            q_dot = diff(H, p)
            p_dot = -diff(H, q)

            return {
                "hamiltonian": hamiltonian_str,
                "operation": operation,
                "hamiltons_equations": {
                    "q_dot": str(q_dot),
                    "p_dot": str(p_dot)
                },
                "latex": {
                    "q_dot": latex(q_dot),
                    "p_dot": latex(p_dot)
                },
                "formula": "q̇ = ∂H/∂p, ṗ = -∂H/∂q",
                "result": {"q_dot": str(q_dot), "p_dot": str(p_dot)}
            }

        elif operation == "poisson_bracket":
            return {
                "hamiltonian": hamiltonian_str,
                "operation": operation,
                "formula": "{f, g} = Σᵢ(∂f/∂qᵢ ∂g/∂pᵢ - ∂f/∂pᵢ ∂g/∂qᵢ)",
                "note": "Poisson bracket structure for canonical transformations"
            }

        else:
            raise ValidationError(f"Unknown operation: {operation}")

    def handle_noether_theorem(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle noether_theorem."""
        lagrangian_str = validate_expression_string(arguments["lagrangian"])
        symmetry_transformation = arguments.get("symmetry_transformation", "time")

        return {
            "lagrangian": lagrangian_str,
            "symmetry_transformation": symmetry_transformation,
            "noether_theorem": "Every continuous symmetry corresponds to a conserved quantity",
            "examples": {
                "time_translation": "Energy conservation (if L doesn't depend explicitly on t)",
                "space_translation": "Momentum conservation (if L is translation invariant)",
                "rotation": "Angular momentum conservation (if L is rotationally invariant)"
            },
            "conserved_quantity": f"For {symmetry_transformation} symmetry, find corresponding conserved quantity"
        }

    def handle_maxwell_equations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle maxwell_equations."""
        operation = arguments["operation"]

        if operation == "vacuum":
            return {
                "operation": operation,
                "equations": {
                    "gauss_law": "∇·E = 0",
                    "no_magnetic_monopoles": "∇·B = 0",
                    "faraday_law": "∇×E = -∂B/∂t",
                    "ampere_maxwell_law": "∇×B = μ₀ε₀∂E/∂t"
                },
                "constants": {
                    "μ₀": "Permeability of free space",
                    "ε₀": "Permittivity of free space",
                    "c": "Speed of light = 1/√(μ₀ε₀)"
                }
            }

        elif operation == "wave_equation":
            return {
                "operation": operation,
                "wave_equations": {
                    "electric_field": "∇²E - (1/c²)∂²E/∂t² = 0",
                    "magnetic_field": "∇²B - (1/c²)∂²B/∂t² = 0"
                },
                "solution": "Plane wave: E = E₀ exp(i(k·r - ωt))",
                "dispersion_relation": "ω = c|k|"
            }

        elif operation == "energy_momentum":
            return {
                "operation": operation,
                "energy_density": "u = (ε₀/2)E² + (1/2μ₀)B²",
                "poynting_vector": "S = (1/μ₀)E×B",
                "momentum_density": "g = ε₀E×B"
            }

        else:
            raise ValidationError(f"Unknown operation: {operation}")

    def handle_special_relativity(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle special_relativity."""
        operation = arguments["operation"]
        velocity_str = arguments.get("velocity", "v")

        v_val = validate_expression_string(velocity_str) if velocity_str != "v" else "v"
        v = safe_sympify(v_val)
        c = symbols('c', positive=True)

        if operation == "lorentz_factor":
            gamma = 1 / sqrt(1 - v**2/c**2)

            return {
                "operation": operation,
                "velocity": velocity_str,
                "lorentz_factor": str(gamma),
                "latex": latex(gamma),
                "formula": "γ = 1/√(1 - v²/c²)",
                "result": str(gamma)
            }

        elif operation == "time_dilation":
            gamma = 1 / sqrt(1 - v**2/c**2)
            t_0 = symbols('t_0', positive=True)
            t = gamma * t_0

            return {
                "operation": operation,
                "proper_time": "t₀",
                "dilated_time": str(t),
                "latex": latex(t),
                "formula": "t = γt₀",
                "result": str(t)
            }

        elif operation == "length_contraction":
            gamma = 1 / sqrt(1 - v**2/c**2)
            L_0 = symbols('L_0', positive=True)
            L = L_0 / gamma

            return {
                "operation": operation,
                "proper_length": "L₀",
                "contracted_length": str(L),
                "latex": latex(L),
                "formula": "L = L₀/γ",
                "result": str(L)
            }

        elif operation == "relativistic_energy":
            m = symbols('m', positive=True)
            gamma = 1 / sqrt(1 - v**2/c**2)
            E = gamma * m * c**2

            return {
                "operation": operation,
                "total_energy": str(E),
                "rest_energy": str(m*c**2),
                "kinetic_energy": str(E - m*c**2),
                "latex": latex(E),
                "formula": "E = γmc²",
                "result": str(E)
            }

        else:
            raise ValidationError(f"Unknown operation: {operation}")

    def handle_create_quantum_state(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_quantum_state."""
        num_qubits = arguments["num_qubits"]
        amplitudes = arguments.get("amplitudes", [1] + [0]*(2**num_qubits-1))
        state = Matrix(amplitudes)
        key = self.ai._get_next_key("quantum_state")
        self.ai.quantum_states[key] = state

        return {
            "state_type": "pure",
            "num_qubits": num_qubits,
            "key": key,
            "state": str(state),
            "latex": latex(state),
            "normalized": bool(abs(state.norm() - 1) < 1e-10),
            "result": key
        }

    def handle_quantum_gate_operations(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum_gate_operations."""
        state_key = arguments["state_key"]
        gates = validate_list_input(arguments["gates"])

        if state_key not in self.ai.quantum_states:
            raise ValidationError(f"Quantum state '{state_key}' not found.")

        state = self.ai.quantum_states[state_key]

        # Define common quantum gates
        gate_matrices = {
            "H": Matrix([[1, 1], [1, -1]]) / sqrt(2),  # Hadamard
            "X": Matrix([[0, 1], [1, 0]]),  # Pauli-X (NOT)
            "Y": Matrix([[0, -I], [I, 0]]),  # Pauli-Y
            "Z": Matrix([[1, 0], [0, -1]]),  # Pauli-Z
            "CNOT": Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),  # CNOT
            "I": eye(2)  # Identity
        }

        result_state = state
        for gate_name in gates:
            if gate_name not in gate_matrices:
                raise ValidationError(f"Unknown gate: {gate_name}")
            gate = gate_matrices[gate_name]
            result_state = gate * result_state

        # Store result
        result_key = self.ai._get_next_key("quantum_state")
        self.ai.quantum_states[result_key] = result_state

        return {
            "original_state_key": state_key,
            "gates_applied": gates,
            "result_key": result_key,
            "result_state": str(result_state),
            "latex": latex(result_state),
            "result": result_key
        }

    def handle_tensor_product_states(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tensor_product_states."""
        state_keys = validate_list_input(arguments["state_keys"])

        # Get all states
        states = []
        for key in state_keys:
            if key not in self.ai.quantum_states:
                raise ValidationError(f"Quantum state '{key}' not found.")
            states.append(self.ai.quantum_states[key])

        # Compute tensor product
        result_state = states[0]
        for state in states[1:]:
            result_state = TensorProduct(result_state, state)

        # Store result
        result_key = self.ai._get_next_key("quantum_state")
        self.ai.quantum_states[result_key] = Matrix(result_state)

        return {
            "state_keys": state_keys,
            "result_key": result_key,
            "result_state": str(result_state),
            "dimension": result_state.shape[0] if hasattr(result_state, 'shape') else len(result_state),
            "result": result_key
        }

    def handle_quantum_entanglement_measure(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum_entanglement_measure."""
        state_key = arguments["state_key"]
        measure_type = arguments["measure_type"]

        if state_key not in self.ai.quantum_states:
            raise ValidationError(f"Quantum state '{state_key}' not found.")

        state = self.ai.quantum_states[state_key]

        if measure_type == "schmidt":
            return {
                "state_key": state_key,
                "measure_type": measure_type,
                "note": "Schmidt decomposition reveals entanglement structure",
                "formula": "|ψ⟩ = Σᵢ √λᵢ |iₐ⟩⊗|iᵦ⟩"
            }

        elif measure_type == "entropy":
            return {
                "state_key": state_key,
                "measure_type": measure_type,
                "note": "Von Neumann entropy S = -Tr(ρ log ρ) measures entanglement",
                "formula": "For pure states, entanglement entropy of subsystem"
            }

        else:
            raise ValidationError(f"Unknown measure type: {measure_type}")

    def handle_quantum_circuit_symbolic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum_circuit_symbolic."""
        num_qubits = arguments["num_qubits"]
        gate_sequence = arguments.get("gate_sequence", [])

        return {
            "num_qubits": num_qubits,
            "gate_sequence": gate_sequence,
            "initial_state": "|0⟩^⊗" + str(num_qubits),
            "note": "Symbolic quantum circuit representation",
            "gates_available": ["H", "X", "Y", "Z", "CNOT", "T", "S"]
        }

    def handle_quantum_measurement(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum_measurement."""
        state_key = arguments["state_key"]
        measurement_basis = arguments.get("measurement_basis", "computational")

        if state_key not in self.ai.quantum_states:
            raise ValidationError(f"Quantum state '{state_key}' not found.")

        state = self.ai.quantum_states[state_key]

        # Calculate probabilities
        probabilities = [abs(complex(amp))**2 for amp in state]

        return {
            "state_key": state_key,
            "measurement_basis": measurement_basis,
            "probabilities": [str(p) for p in probabilities],
            "note": "Measurement collapses state to eigenstate with given probabilities",
            "born_rule": "|⟨ψ|φ⟩|²"
        }

    def handle_quantum_fidelity(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum_fidelity."""
        state_key_1 = arguments["state_key_1"]
        state_key_2 = arguments["state_key_2"]

        if state_key_1 not in self.ai.quantum_states:
            raise ValidationError(f"Quantum state '{state_key_1}' not found.")
        if state_key_2 not in self.ai.quantum_states:
            raise ValidationError(f"Quantum state '{state_key_2}' not found.")

        state1 = self.ai.quantum_states[state_key_1]
        state2 = self.ai.quantum_states[state_key_2]

        # Fidelity F = |⟨ψ₁|ψ₂⟩|²
        inner_product = (Dagger(state1) * state2)[0, 0] if state1.shape[1] == 1 else (state1.H * state2)[0, 0]
        fidelity = abs(inner_product)**2

        return {
            "state_key_1": state_key_1,
            "state_key_2": state_key_2,
            "fidelity": str(simplify(fidelity)),
            "latex": latex(fidelity),
            "formula": "F = |⟨ψ₁|ψ₂⟩|²",
            "result": str(simplify(fidelity))
        }

    def handle_pauli_matrices(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pauli_matrices."""
        operation = arguments["operation"]
        matrix_name = arguments.get("matrix", "X")

        # Define Pauli matrices
        pauli_matrices = {
            "X": Matrix([[0, 1], [1, 0]]),
            "Y": Matrix([[0, -I], [I, 0]]),
            "Z": Matrix([[1, 0], [0, -1]]),
            "I": eye(2)
        }

        if operation == "get":
            if matrix_name not in pauli_matrices:
                raise ValidationError(f"Unknown Pauli matrix: {matrix_name}")

            matrix = pauli_matrices[matrix_name]

            return {
                "operation": operation,
                "matrix": matrix_name,
                "matrix_elements": str(matrix),
                "latex": latex(matrix),
                "result": str(matrix)
            }

        elif operation == "properties":
            return {
                "operation": operation,
                "properties": {
                    "hermitian": "σᵢ† = σᵢ",
                    "unitary": "σᵢ² = I",
                    "traceless": "Tr(σᵢ) = 0 for i=X,Y,Z",
                    "anticommutation": "{σᵢ, σⱼ} = 2δᵢⱼI",
                    "commutation": "[σᵢ, σⱼ] = 2iεᵢⱼₖσₖ"
                }
            }

        else:
            raise ValidationError(f"Unknown operation: {operation}")

    def handle_commutator_anticommutator(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle commutator_anticommutator."""
        operator_1_str = validate_expression_string(arguments["operator_1"])
        operator_2_str = validate_expression_string(arguments["operator_2"])
        operation = arguments["operation"]

        # Parse operators as matrices - use sp.sympify as we need matrix operations
        op1_parsed = sp.sympify(operator_1_str)
        op2_parsed = sp.sympify(operator_2_str)

        op1 = op1_parsed if isinstance(op1_parsed, Matrix) else Matrix([[op1_parsed]])
        op2 = op2_parsed if isinstance(op2_parsed, Matrix) else Matrix([[op2_parsed]])

        if operation == "commutator":
            # [A, B] = AB - BA
            result_op = op1 * op2 - op2 * op1

            return {
                "operation": "commutator",
                "operator_1": operator_1_str,
                "operator_2": operator_2_str,
                "commutator": str(result_op),
                "latex": latex(result_op),
                "formula": "[A, B] = AB - BA",
                "result": str(result_op)
            }

        elif operation == "anticommutator":
            # {A, B} = AB + BA
            result_op = op1 * op2 + op2 * op1

            return {
                "operation": "anticommutator",
                "operator_1": operator_1_str,
                "operator_2": operator_2_str,
                "anticommutator": str(result_op),
                "latex": latex(result_op),
                "formula": "{A, B} = AB + BA",
                "result": str(result_op)
            }

        else:
            raise ValidationError(f"Unknown operation: {operation}")

    def handle_quantum_evolution(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum_evolution."""
        state_key = arguments["state_key"]
        hamiltonian_str = validate_expression_string(arguments["hamiltonian"])
        time_str = arguments.get("time", "t")

        if state_key not in self.ai.quantum_states:
            raise ValidationError(f"Quantum state '{state_key}' not found.")

        t_val = validate_expression_string(time_str) if time_str != "t" else "t"
        t = safe_sympify(t_val)
        H = safe_sympify(hamiltonian_str)

        # Time evolution operator: U(t) = exp(-iHt/ℏ)
        hbar = symbols('hbar', positive=True)
        U_symbolic = exp(-I * H * t / hbar)

        return {
            "state_key": state_key,
            "hamiltonian": hamiltonian_str,
            "time": time_str,
            "evolution_operator": str(U_symbolic),
            "latex": latex(U_symbolic),
            "formula": "U(t) = exp(-iHt/ℏ)",
            "schrodinger_equation": "iℏ ∂|ψ⟩/∂t = H|ψ⟩",
            "result": str(U_symbolic)
        }

    def handle_symbolic_optimization_setup(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle symbolic_optimization_setup."""
        objective_str = validate_expression_string(arguments["objective"])
        variables = arguments.get("variables", [])

        # Validate variable names
        if isinstance(variables, list):
            variables = [validate_variable_name(v) for v in variables]

        objective = safe_sympify(objective_str)

        # Create symbolic optimization problem
        var_symbols = symbols(' '.join(variables), real=True)

        gradient = [str(diff(objective, v)) for v in (var_symbols if isinstance(var_symbols, tuple) else [var_symbols])]

        return {
            "objective": objective_str,
            "variables": variables,
            "optimization_type": "minimize or maximize",
            "gradient": gradient,
            "critical_points": "Solve ∇f = 0",
            "note": "Use ai.optimize_function() for full optimization",
            "result": {"gradient": gradient}
        }


# Entry point
server = PhysicsServer()

if __name__ == "__main__":
    server.run()
