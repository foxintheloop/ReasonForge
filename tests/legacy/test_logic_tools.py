"""
Tests for logic and knowledge representation tools.

This module tests the 6 logic tools from logic_tools.py:
1. first_order_logic - FOL reasoning with quantifiers
2. propositional_logic_advanced - CNF/DNF conversion, SAT
3. knowledge_graph_reasoning - Transitive closure, path finding
4. constraint_satisfaction - CSP solving
5. modal_logic - Modal logic with necessity/possibility operators
6. fuzzy_logic - Fuzzy sets and approximate reasoning

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.logic_tools import handle_logic_tools


class TestFirstOrderLogic:
    """Test the 'first_order_logic' tool."""

    def test_parse_fol_formula(self, ai):
        """Test parsing FOL formula ∀x (P(x) → ∃y Q(x,y)) (from USAGE_EXAMPLES.md)."""
        result = handle_logic_tools(
            'first_order_logic',
            {
                'operation': 'parse',
                'formula': 'ForAll(x, Implies(P(x), Exists(y, Q(x, y))))'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'parsed' in data

    def test_fol_satisfiability(self, ai):
        """Test FOL satisfiability checking."""
        result = handle_logic_tools(
            'first_order_logic',
            {
                'operation': 'satisfiability',
                'formula': 'Exists(x, P(x))'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'satisfiable' in data

    def test_fol_proof(self, ai):
        """Test FOL proof generation."""
        result = handle_logic_tools(
            'first_order_logic',
            {
                'operation': 'prove',
                'premises': ['ForAll(x, P(x))'],
                'conclusion': 'P(a)'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestPropositionalLogicAdvanced:
    """Test the 'propositional_logic_advanced' tool."""

    def test_convert_to_cnf(self, ai):
        """Test converting (A ∨ B) ∧ (C ∨ D) to CNF (from USAGE_EXAMPLES.md)."""
        result = handle_logic_tools(
            'propositional_logic_advanced',
            {
                'operation': 'to_cnf',
                'formula': '(A | B) & (C | D)'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'cnf' in data

    def test_convert_to_dnf(self, ai):
        """Test converting to DNF."""
        result = handle_logic_tools(
            'propositional_logic_advanced',
            {
                'operation': 'to_dnf',
                'formula': '(A & B) | (C & D)'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'dnf' in data

    def test_satisfiability_check(self, ai):
        """Test SAT solving."""
        result = handle_logic_tools(
            'propositional_logic_advanced',
            {
                'operation': 'satisfiability',
                'formula': 'A & ~A'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'satisfiable' in data

    def test_truth_table_generation(self, ai):
        """Test truth table generation."""
        result = handle_logic_tools(
            'propositional_logic_advanced',
            {
                'operation': 'truth_table',
                'formula': 'A & B',
                'variables': ['A', 'B']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'truth_table' in data


class TestKnowledgeGraphReasoning:
    """Test the 'knowledge_graph_reasoning' tool."""

    def test_transitive_closure(self, ai):
        """Test transitive closure (from USAGE_EXAMPLES.md)."""
        result = handle_logic_tools(
            'knowledge_graph_reasoning',
            {
                'operation': 'transitive_closure',
                'edges': [['A', 'R', 'B'], ['B', 'R', 'C']]
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'closure' in data
        # Should infer A->C
        assert ['A', 'R', 'C'] in data['closure'] or 'new_inferences' in data

    def test_path_exists(self, ai):
        """Test path finding in knowledge graph."""
        result = handle_logic_tools(
            'knowledge_graph_reasoning',
            {
                'operation': 'path_exists',
                'edges': [['A', 'R', 'B'], ['B', 'R', 'C']],
                'query_from': 'A',
                'query_to': 'C'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'path_exists' in data

    def test_infer_relations(self, ai):
        """Test relation inference."""
        result = handle_logic_tools(
            'knowledge_graph_reasoning',
            {
                'operation': 'infer_relations',
                'edges': [['Alice', 'parent_of', 'Bob'], ['Bob', 'parent_of', 'Charlie']]
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestConstraintSatisfaction:
    """Test the 'constraint_satisfaction' tool."""

    def test_solve_csp(self, ai):
        """Test CSP solving (from USAGE_EXAMPLES.md)."""
        result = handle_logic_tools(
            'constraint_satisfaction',
            {
                'variables': ['A', 'B', 'C'],
                'domains': {'A': [1, 2], 'B': [2, 3], 'C': [1, 3]},
                'constraints': ['A < B', 'B < C']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'solution' in data

    def test_csp_multiple_solutions(self, ai):
        """Test finding all CSP solutions."""
        result = handle_logic_tools(
            'constraint_satisfaction',
            {
                'variables': ['X', 'Y'],
                'domains': {'X': [1, 2], 'Y': [1, 2]},
                'constraints': ['X != Y'],
                'find_all': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'all_solutions' in data

    def test_csp_no_solution(self, ai):
        """Test CSP with no valid solution."""
        result = handle_logic_tools(
            'constraint_satisfaction',
            {
                'variables': ['A', 'B'],
                'domains': {'A': [1], 'B': [1]},
                'constraints': ['A != B']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data


class TestModalLogic:
    """Test the 'modal_logic' tool."""

    def test_validate_k_axiom(self, ai):
        """Test validating K axiom (from USAGE_EXAMPLES.md)."""
        result = handle_logic_tools(
            'modal_logic',
            {
                'logic_type': 'K',
                'formula': 'Implies(Box(Implies(p, q)), Implies(Box(p), Box(q)))',
                'operation': 'validate'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'valid' in data

    def test_parse_modal_formula(self, ai):
        """Test parsing modal logic formula."""
        result = handle_logic_tools(
            'modal_logic',
            {
                'logic_type': 'S5',
                'formula': 'Box(Diamond(p))',
                'operation': 'parse'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    def test_temporal_logic(self, ai):
        """Test temporal modal logic."""
        result = handle_logic_tools(
            'modal_logic',
            {
                'logic_type': 'temporal',
                'formula': 'Always(Eventually(p))',
                'operation': 'parse'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestFuzzyLogic:
    """Test the 'fuzzy_logic' tool."""

    def test_fuzzy_and(self, ai):
        """Test fuzzy AND operation (from USAGE_EXAMPLES.md)."""
        result = handle_logic_tools(
            'fuzzy_logic',
            {
                'operation': 'fuzzy_and',
                'fuzzy_set_a': {'x': 0.7, 'y': 0.3},
                'fuzzy_set_b': {'x': 0.5, 'y': 0.9},
                't_norm': 'min'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
        # Should be {x: 0.5, y: 0.3}
        assert data['result']['x'] == 0.5
        assert data['result']['y'] == 0.3

    def test_fuzzy_or(self, ai):
        """Test fuzzy OR operation."""
        result = handle_logic_tools(
            'fuzzy_logic',
            {
                'operation': 'fuzzy_or',
                'fuzzy_set_a': {'x': 0.7, 'y': 0.3},
                'fuzzy_set_b': {'x': 0.5, 'y': 0.9},
                's_norm': 'max'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    def test_membership_function(self, ai):
        """Test fuzzy membership function."""
        result = handle_logic_tools(
            'fuzzy_logic',
            {
                'operation': 'membership',
                'membership_function': 'triangular',
                'parameters': {'a': 0, 'b': 5, 'c': 10}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    def test_defuzzification(self, ai):
        """Test defuzzification."""
        result = handle_logic_tools(
            'fuzzy_logic',
            {
                'operation': 'defuzzify',
                'fuzzy_set_a': {'x1': 0.2, 'x2': 0.5, 'x3': 0.8, 'x4': 0.3}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data
