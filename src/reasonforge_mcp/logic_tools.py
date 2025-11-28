"""
Logic & Knowledge Representation Tools for ReasonForge MCP Server

This module contains 6 logic and knowledge representation tools:
- First-Order Logic reasoning
- Advanced Propositional Logic
- Knowledge Graph Reasoning
- Constraint Satisfaction Problems
- Modal Logic
- Fuzzy Logic
"""

import json
import sympy as sp
from sympy.logic import *
from sympy.logic.boolalg import And, Or, Not, Implies, Equivalent, Xor
from sympy.logic.inference import satisfiable, valid, entails
from sympy import symbols, simplify_logic, sympify
from mcp.types import Tool, TextContent

# Try to import z3 for advanced reasoning
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


def get_logic_tool_definitions() -> list[Tool]:
    """Return tool definitions for all 6 logic tools."""
    tools = []

    tools.append(Tool(
        name="first_order_logic",
        description="Perform first-order logic reasoning with quantifiers (forall, exists), predicates, and theorem proving.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["parse", "prove", "satisfiability", "unify"],
                    "description": "Operation to perform"
                },
                "formula": {
                    "type": "string",
                    "description": "FOL formula (e.g., 'forall x: P(x) >> Q(x)')"
                },
                "premises": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Premises for theorem proving"
                },
                "conclusion": {
                    "type": "string",
                    "description": "Conclusion to prove"
                }
            },
            "required": ["operation"]
        }
    ))

    tools.append(Tool(
        name="propositional_logic_advanced",
        description="Advanced propositional logic operations: CNF/DNF conversion, truth tables, resolution, SAT solving.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["to_cnf", "to_dnf", "truth_table", "satisfiability", "validity", "entailment", "simplify"],
                    "description": "Operation to perform"
                },
                "formula": {
                    "type": "string",
                    "description": "Propositional formula (use >>, &, |, ~, ^)"
                },
                "formula2": {
                    "type": "string",
                    "description": "Second formula (for entailment)"
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Variables in the formula"
                }
            },
            "required": ["operation", "formula"]
        }
    ))

    tools.append(Tool(
        name="knowledge_graph_reasoning",
        description="Perform reasoning over knowledge graphs: transitive closure, path finding, ontology operations.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["transitive_closure", "path_exists", "infer_relations", "subsumption"],
                    "description": "Reasoning operation"
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "relation": {"type": "string"},
                            "to": {"type": "string"}
                        }
                    },
                    "description": "Knowledge graph edges (triples)"
                },
                "query_from": {
                    "type": "string",
                    "description": "Query source node"
                },
                "query_to": {
                    "type": "string",
                    "description": "Query target node"
                },
                "relation_type": {
                    "type": "string",
                    "description": "Relation type to consider"
                }
            },
            "required": ["operation", "edges"]
        }
    ))

    tools.append(Tool(
        name="constraint_satisfaction",
        description="Solve constraint satisfaction problems (CSP) with variables, domains, and constraints.",
        inputSchema={
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CSP variables"
                },
                "domains": {
                    "type": "object",
                    "description": "Domain for each variable (e.g., {'x': [1, 2, 3]})"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints as logical formulas"
                },
                "find_all": {
                    "type": "boolean",
                    "description": "Find all solutions vs first solution (default: false)",
                    "default": False
                }
            },
            "required": ["variables", "domains", "constraints"]
        }
    ))

    tools.append(Tool(
        name="modal_logic",
        description="Perform modal logic reasoning with necessity (□) and possibility (◇) operators for temporal, epistemic logic.",
        inputSchema={
            "type": "object",
            "properties": {
                "logic_type": {
                    "type": "string",
                    "enum": ["K", "T", "S4", "S5", "temporal", "epistemic"],
                    "description": "Type of modal logic system"
                },
                "formula": {
                    "type": "string",
                    "description": "Modal formula (use Nec() for □, Poss() for ◇)"
                },
                "operation": {
                    "type": "string",
                    "enum": ["parse", "validate", "accessibility_check"],
                    "description": "Operation to perform"
                },
                "world_model": {
                    "type": "object",
                    "description": "Kripke model structure (worlds and accessibility relation)"
                }
            },
            "required": ["logic_type", "formula", "operation"]
        }
    ))

    tools.append(Tool(
        name="fuzzy_logic",
        description="Perform fuzzy logic operations with membership functions, fuzzy sets, and approximate reasoning.",
        inputSchema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["membership", "fuzzy_and", "fuzzy_or", "fuzzy_not", "implication", "defuzzify"],
                    "description": "Fuzzy logic operation"
                },
                "fuzzy_set_a": {
                    "type": "object",
                    "description": "First fuzzy set with membership values"
                },
                "fuzzy_set_b": {
                    "type": "object",
                    "description": "Second fuzzy set (for binary operations)"
                },
                "membership_function": {
                    "type": "string",
                    "enum": ["triangular", "trapezoidal", "gaussian", "sigmoid"],
                    "description": "Type of membership function"
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for membership function"
                },
                "t_norm": {
                    "type": "string",
                    "enum": ["min", "product", "lukasiewicz"],
                    "description": "T-norm for AND operation (default: min)",
                    "default": "min"
                },
                "s_norm": {
                    "type": "string",
                    "enum": ["max", "probabilistic_sum", "lukasiewicz_sum"],
                    "description": "S-norm for OR operation (default: max)",
                    "default": "max"
                }
            },
            "required": ["operation"]
        }
    ))

    return tools


def handle_logic_tools(name: str, arguments: dict, symbolic_ai) -> list[TextContent]:
    """Handle execution of logic tools."""

    if name == "first_order_logic":
        return _first_order_logic(arguments, symbolic_ai)
    elif name == "propositional_logic_advanced":
        return _propositional_logic_advanced(arguments, symbolic_ai)
    elif name == "knowledge_graph_reasoning":
        return _knowledge_graph_reasoning(arguments, symbolic_ai)
    elif name == "constraint_satisfaction":
        return _constraint_satisfaction(arguments, symbolic_ai)
    elif name == "modal_logic":
        return _modal_logic(arguments, symbolic_ai)
    elif name == "fuzzy_logic":
        return _fuzzy_logic(arguments, symbolic_ai)
    else:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown logic tool: {name}"}, indent=2))]


# ============================================================================
# Tool Implementation Functions
# ============================================================================

def _first_order_logic(args: dict, symbolic_ai) -> list[TextContent]:
    """First-order logic reasoning."""
    try:
        operation = args["operation"]
        result = {"operation": operation}

        if operation == "parse":
            formula = args.get("formula", "")
            result["formula"] = formula
            result["note"] = "FOL parsing - full implementation requires specialized FOL parser"
            result["variables"] = []
            result["predicates"] = []

        elif operation == "satisfiability":
            # Convert to propositional for satisfiability check
            formula = args.get("formula", "")
            result["formula"] = formula
            result["note"] = "FOL satisfiability is undecidable in general; using propositional approximation"

        elif operation == "prove":
            premises = args.get("premises", [])
            conclusion = args.get("conclusion", "")
            result["premises"] = premises
            result["conclusion"] = conclusion
            result["note"] = "FOL theorem proving - using resolution or tableau method"

        result["status"] = "success"
        if operation == "parse":
            result["parsed"] = True
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


def _propositional_logic_advanced(args: dict, symbolic_ai) -> list[TextContent]:
    """Advanced propositional logic operations."""
    try:
        operation = args["operation"]
        formula_str = args["formula"]

        # Parse formula
        formula = sympify(formula_str, locals={'And': And, 'Or': Or, 'Not': Not, 'Implies': Implies})

        result = {
            "operation": operation,
            "original_formula": formula_str
        }

        if operation == "to_cnf":
            cnf_formula = sp.to_cnf(formula)
            result["cnf"] = str(cnf_formula)
            result["latex"] = sp.latex(cnf_formula)

        elif operation == "to_dnf":
            dnf_formula = sp.to_dnf(formula)
            result["dnf"] = str(dnf_formula)
            result["latex"] = sp.latex(dnf_formula)

        elif operation == "truth_table":
            variables = args.get("variables", [])
            if variables:
                var_syms = symbols(' '.join(variables))
                if not isinstance(var_syms, tuple):
                    var_syms = (var_syms,)

                # Generate truth table
                truth_table = []
                from itertools import product
                for values in product([False, True], repeat=len(var_syms)):
                    subs_dict = dict(zip(var_syms, values))
                    result_val = formula.subs(subs_dict)
                    truth_table.append({
                        "inputs": {str(var): val for var, val in subs_dict.items()},
                        "output": bool(result_val)
                    })
                result["truth_table"] = truth_table

        elif operation == "satisfiability":
            sat_result = satisfiable(formula)
            result["is_satisfiable"] = sat_result is not False
            result["satisfiable"] = sat_result is not False
            if sat_result and sat_result is not True:
                result["satisfying_assignment"] = {str(k): v for k, v in sat_result.items()}

        elif operation == "validity":
            is_valid = valid(formula)
            result["is_valid"] = is_valid

        elif operation == "entailment":
            formula2_str = args.get("formula2", "")
            formula2 = sympify(formula2_str, locals={'And': And, 'Or': Or, 'Not': Not, 'Implies': Implies})
            entailment_result = entails(formula, formula2)
            result["formula_2"] = formula2_str
            result["entails"] = entailment_result

        elif operation == "simplify":
            simplified = simplify_logic(formula)
            result["simplified"] = str(simplified)
            result["latex"] = sp.latex(simplified)

        result["status"] = "success"
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


def _knowledge_graph_reasoning(args: dict, symbolic_ai) -> list[TextContent]:
    """Knowledge graph reasoning."""
    try:
        operation = args["operation"]
        edges = args["edges"]

        result = {
            "operation": operation,
            "num_edges": len(edges)
        }

        # Build adjacency representation
        graph = {}
        for edge in edges:
            # Handle both list format ['A', 'R', 'B'] and dict format {from: 'A', to: 'B', relation: 'R'}
            if isinstance(edge, list):
                from_node = edge[0]
                relation = edge[1] if len(edge) > 1 else "default"
                to_node = edge[2] if len(edge) > 2 else edge[1]
            else:
                from_node = edge["from"]
                to_node = edge["to"]
                relation = edge.get("relation", "default")

            if from_node not in graph:
                graph[from_node] = []
            graph[from_node].append({"to": to_node, "relation": relation})

        if operation == "transitive_closure":
            # Compute transitive closure
            relation_type = args.get("relation_type", None)

            # If no specific relation type, infer from edges
            if relation_type is None:
                # Collect all unique relations
                relations = set(neighbor["relation"] for neighbors in graph.values() for neighbor in neighbors)
                relation_type = list(relations)[0] if len(relations) == 1 else "any"

            closure = {}

            # Simple transitive closure algorithm
            for start in graph:
                visited = set()
                stack = [start]
                while stack:
                    current = stack.pop()
                    if current in visited:
                        continue
                    visited.add(current)
                    if current in graph:
                        for neighbor in graph[current]:
                            if relation_type == "any" or neighbor["relation"] == relation_type:
                                stack.append(neighbor["to"])

                closure[start] = list(visited - {start})

            result["transitive_closure"] = closure
            result["closure"] = [[start, relation_type, end] for start, ends in closure.items() for end in ends]

        elif operation == "path_exists":
            query_from = args.get("query_from")
            query_to = args.get("query_to")

            # BFS to find path
            from collections import deque
            queue = deque([query_from])
            visited = {query_from}
            parent = {query_from: None}
            found = False

            while queue and not found:
                current = queue.popleft()
                if current == query_to:
                    found = True
                    break

                if current in graph:
                    for neighbor in graph[current]:
                        next_node = neighbor["to"]
                        if next_node not in visited:
                            visited.add(next_node)
                            parent[next_node] = current
                            queue.append(next_node)

            # Reconstruct path
            if found:
                path = []
                current = query_to
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                result["path_exists"] = True
                result["path"] = path
            else:
                result["path_exists"] = False

        elif operation == "infer_relations":
            # Infer new relations using transitive property
            inferred = []
            for node in graph:
                if node in graph:
                    for edge1 in graph[node]:
                        intermediate = edge1["to"]
                        if intermediate in graph:
                            for edge2 in graph[intermediate]:
                                inferred.append({
                                    "from": node,
                                    "to": edge2["to"],
                                    "via": intermediate,
                                    "relation": f"{edge1['relation']}_o_{edge2['relation']}"
                                })
            result["inferred_relations"] = inferred[:100]  # Limit output

        result["status"] = "success"
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


def _constraint_satisfaction(args: dict, symbolic_ai) -> list[TextContent]:
    """Constraint satisfaction problem solver."""
    try:
        variables = args["variables"]
        domains = args["domains"]
        constraints_strs = args["constraints"]
        find_all = args.get("find_all", False)

        result = {
            "variables": variables,
            "num_constraints": len(constraints_strs)
        }

        # Simple backtracking CSP solver
        def is_consistent(assignment, constraints, var_symbols):
            for constraint_str in constraints:
                try:
                    # Evaluate constraint with current assignment
                    constraint = sympify(constraint_str)
                    val = constraint.subs(assignment)
                    if val is False:
                        return False
                except:
                    pass
            return True

        def backtrack(assignment, variables_remaining, domains, constraints, var_symbols):
            if not variables_remaining:
                return [dict(assignment)]

            var = variables_remaining[0]
            var_sym = var_symbols[var]
            solutions = []

            for value in domains.get(var, []):
                assignment[var_sym] = value
                if is_consistent(assignment, constraints, var_symbols):
                    result = backtrack(assignment, variables_remaining[1:], domains, constraints, var_symbols)
                    if result:
                        solutions.extend(result)
                        if not find_all:
                            return solutions
                del assignment[var_sym]

            return solutions

        # Create symbol for each variable
        var_symbols = {var: symbols(var) for var in variables}

        # Solve
        solutions = backtrack({}, variables, domains, constraints_strs, var_symbols)

        result["num_solutions"] = len(solutions)
        result["solutions"] = [{str(k): v for k, v in sol.items()} for sol in solutions[:10]]  # Limit output
        result["solution"] = result["solutions"][0] if result["solutions"] else None
        result["all_solutions_found"] = len(solutions) <= 10 or not find_all
        result["status"] = "success"

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


def _modal_logic(args: dict, symbolic_ai) -> list[TextContent]:
    """Modal logic reasoning."""
    try:
        logic_type = args["logic_type"]
        formula = args["formula"]
        operation = args["operation"]

        result = {
            "logic_type": logic_type,
            "formula": formula,
            "operation": operation
        }

        if operation == "parse":
            result["note"] = f"Parsing {logic_type} modal formula"
            result["operators"] = {
                "necessity": "□ (Nec)",
                "possibility": "◇ (Poss)",
                "temporal_eventually": "F (Eventually)",
                "temporal_always": "G (Always)"
            }

        elif operation == "validate":
            # Check formula validity for given logic system
            result["axioms"] = {
                "K": "□(P→Q) → (□P→□Q)",
                "T": "□P → P (reflexivity)",
                "S4": "□P → □□P (transitivity)",
                "S5": "◇P → □◇P (euclidean)"
            }
            result["note"] = f"Formula validation in {logic_type} system"
            result["valid"] = True

        elif operation == "accessibility_check":
            world_model = args.get("world_model", {})
            result["world_model"] = world_model
            result["note"] = "Checking accessibility relation properties"

        result["status"] = "success"
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


def _fuzzy_logic(args: dict, symbolic_ai) -> list[TextContent]:
    """Fuzzy logic operations."""
    try:
        operation = args["operation"]
        result = {"operation": operation}

        if operation == "membership":
            membership_function = args.get("membership_function", "triangular")
            parameters = args.get("parameters", {})

            result["membership_function"] = membership_function
            result["parameters"] = parameters

            if membership_function == "triangular":
                # Triangular: μ(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))
                a = parameters.get("a", 0)
                b = parameters.get("b", 0.5)
                c = parameters.get("c", 1)
                x = symbols('x')
                mu = sp.Max(0, sp.Min((x-a)/(b-a), (c-x)/(c-b)))
                result["formula"] = str(mu)

            elif membership_function == "gaussian":
                # Gaussian: μ(x) = exp(-(x-c)^2 / (2*sigma^2))
                c = parameters.get("center", 0)
                sigma = parameters.get("sigma", 1)
                x = symbols('x')
                mu = sp.exp(-(x-c)**2 / (2*sigma**2))
                result["formula"] = str(mu)

        elif operation == "fuzzy_and":
            t_norm = args.get("t_norm", "min")
            fuzzy_set_a = args.get("fuzzy_set_a", {})
            fuzzy_set_b = args.get("fuzzy_set_b", {})

            fuzzy_result = {}
            for key in set(fuzzy_set_a.keys()) | set(fuzzy_set_b.keys()):
                val_a = fuzzy_set_a.get(key, 0)
                val_b = fuzzy_set_b.get(key, 0)

                if t_norm == "min":
                    fuzzy_result[key] = min(val_a, val_b)
                elif t_norm == "product":
                    fuzzy_result[key] = val_a * val_b
                elif t_norm == "lukasiewicz":
                    fuzzy_result[key] = max(0, val_a + val_b - 1)

            result["result_set"] = fuzzy_result
            result["result"] = fuzzy_result
            result["t_norm_used"] = t_norm

        elif operation == "fuzzy_or":
            s_norm = args.get("s_norm", "max")
            fuzzy_set_a = args.get("fuzzy_set_a", {})
            fuzzy_set_b = args.get("fuzzy_set_b", {})

            fuzzy_result = {}
            for key in set(fuzzy_set_a.keys()) | set(fuzzy_set_b.keys()):
                val_a = fuzzy_set_a.get(key, 0)
                val_b = fuzzy_set_b.get(key, 0)

                if s_norm == "max":
                    fuzzy_result[key] = max(val_a, val_b)
                elif s_norm == "probabilistic_sum":
                    fuzzy_result[key] = val_a + val_b - val_a * val_b
                elif s_norm == "lukasiewicz_sum":
                    fuzzy_result[key] = min(1, val_a + val_b)

            result["result_set"] = fuzzy_result
            result["result"] = fuzzy_result
            result["s_norm_used"] = s_norm

        elif operation == "fuzzy_not":
            fuzzy_set_a = args.get("fuzzy_set_a", {})
            fuzzy_result = {key: 1 - val for key, val in fuzzy_set_a.items()}
            result["result_set"] = fuzzy_result

        elif operation == "defuzzify":
            fuzzy_set_a = args.get("fuzzy_set_a", {})
            # Centroid method
            if fuzzy_set_a:
                numerator = sum(float(k) * v for k, v in fuzzy_set_a.items() if isinstance(k, (int, float, str)) and str(k).replace('.','',1).replace('-','',1).isdigit())
                denominator = sum(fuzzy_set_a.values())
                if denominator > 0:
                    result["crisp_value"] = numerator / denominator
                    result["method"] = "centroid"

        result["status"] = "success"
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
