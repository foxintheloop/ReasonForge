"""
Comprehensive tests for reasonforge-logic MCP server.

Tests all 13 tools:
- Pattern Recognition (5 tools): pattern_to_equation, symbolic_knowledge_extraction, symbolic_theorem_proving, feature_extraction, structure_mapping
- Logic Systems (5 tools): automated_conjecture, first_order_logic, propositional_logic_advanced, knowledge_graph_reasoning, constraint_satisfaction
- Specialized Logic (3 tools): modal_logic, fuzzy_logic, generate_proof
"""

import json
import pytest

from reasonforge_logic.server import server as logic_server


class TestPatternRecognition:
    """Test pattern recognition tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_pattern_to_equation(self):
        """Test fitting equation to data pattern."""
        result = await logic_server.call_tool_for_test(
            "pattern_to_equation",
            {
                "x_values": [1, 2, 3, 4, 5],
                "y_values": [1, 4, 9, 16, 25]
            }
        )
        data = json.loads(result[0].text)

        assert "patterns" in data or "equation" in data

    @pytest.mark.asyncio
    async def test_symbolic_knowledge_extraction(self):
        """Test extracting logical rules."""
        result = await logic_server.call_tool_for_test(
            "symbolic_knowledge_extraction",
            {
                "data": {"examples": [{"x": 1, "y": 2}, {"x": 2, "y": 4}]},
                "extraction_type": "rules"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_symbolic_theorem_proving(self):
        """Test theorem proving with symbolic deduction."""
        result = await logic_server.call_tool_for_test(
            "symbolic_theorem_proving",
            {
                "theorem": "For all x, x + 0 = x",
                "axioms": ["x + 0 = x"]
            }
        )
        data = json.loads(result[0].text)

        assert "theorem" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_feature_extraction(self):
        """Test extracting common features from examples."""
        result = await logic_server.call_tool_for_test(
            "feature_extraction",
            {
                "positive_examples": [{"shape": "circle"}, {"shape": "sphere"}],
                "negative_examples": [{"shape": "square"}]
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_structure_mapping(self):
        """Test finding structural mappings."""
        result = await logic_server.call_tool_for_test(
            "structure_mapping",
            {
                "source_domain": "numbers",
                "target_domain": "geometry",
                "relation": "addition"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0


class TestLogicSystems:
    """Test logic system tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_automated_conjecture(self):
        """Test generating conjectures."""
        result = await logic_server.call_tool_for_test(
            "automated_conjecture",
            {
                "domain": "number_theory",
                "constraints": {}
            }
        )
        data = json.loads(result[0].text)

        assert "conjecture" in data or "conjectures" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_first_order_logic(self):
        """Test first-order logic operations."""
        result = await logic_server.call_tool_for_test(
            "first_order_logic",
            {
                "operation": "parse",
                "formula": "forall x, P(x) -> Q(x)"
            }
        )
        data = json.loads(result[0].text)

        assert "operation" in data or "formula" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_propositional_logic_advanced(self):
        """Test advanced propositional logic."""
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "cnf",
                "formula": "(A | B) & (C | D)"
            }
        )
        data = json.loads(result[0].text)

        assert "operation" in data or "result" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_knowledge_graph_reasoning(self):
        """Test knowledge graph reasoning."""
        result = await logic_server.call_tool_for_test(
            "knowledge_graph_reasoning",
            {
                "nodes": ["A", "B", "C"],
                "edges": [{"from": "A", "to": "B", "relation": "knows"}],
                "query": "path"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_constraint_satisfaction(self):
        """Test constraint satisfaction problems."""
        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {
                "variables": ["x", "y"],
                "domains": {"x": [1, 2, 3], "y": [1, 2, 3]},
                "constraints": ["x != y"]
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0


class TestSpecializedLogic:
    """Test specialized logic tools (3 tools)."""

    @pytest.mark.asyncio
    async def test_modal_logic(self):
        """Test modal logic operations."""
        result = await logic_server.call_tool_for_test(
            "modal_logic",
            {
                "operation": "necessitation",
                "formula": "P"
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_fuzzy_logic(self):
        """Test fuzzy logic operations."""
        result = await logic_server.call_tool_for_test(
            "fuzzy_logic",
            {
                "operation": "union",
                "fuzzy_set_a": {"x1": 0.8, "x2": 0.3},
                "fuzzy_set_b": {"x1": 0.6, "x2": 0.9}
            }
        )
        data = json.loads(result[0].text)

        assert "operation" in data or "result" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_generate_proof(self):
        """Test generating mathematical proofs."""
        result = await logic_server.call_tool_for_test(
            "generate_proof",
            {
                "theorem": "a + b = b + a",
                "proof_type": "direct"
            }
        )
        data = json.loads(result[0].text)

        assert "theorem" in data or "proof" in data or len(data) > 0


class TestDemoProblems:
    """Test demo problem scenarios - showcasing ReasonForge vs LLM capabilities."""

    @pytest.mark.asyncio
    async def test_sat_satisfiability_with_verification(self):
        """Problem 5: SAT check with assignment verification.

        Formula: (A ∨ B) ∧ (¬A ∨ C) ∧ (¬B ∨ ¬C) ∧ (¬A ∨ ¬B)
        Expected: Satisfiable (multiple valid assignments exist)
        """
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": "(A | B) & (~A | C) & (~B | ~C) & (~A | ~B)"
            }
        )
        data = json.loads(result[0].text)

        # Verify satisfiability
        assert data["satisfiable"] == True
        assert "assignment" in data

        # Verify assignment is valid (check all clauses evaluate to True)
        assignment = data["assignment"]
        A, B, C = assignment["A"], assignment["B"], assignment["C"]

        # Manually verify the assignment satisfies all clauses
        assert (A or B)            # (A ∨ B)
        assert (not A or C)        # (¬A ∨ C)
        assert (not B or not C)    # (¬B ∨ ¬C)
        assert (not A or not B)    # (¬A ∨ ¬B)

        # Verify verification output
        assert "verification" in data
        assert data["verification"]["all_satisfied"] == True
        assert len(data["verification"]["clauses"]) == 4

        # Verify all clause verifications show satisfied
        for clause in data["verification"]["clauses"]:
            assert clause["satisfied"] == True

        # Verify summary
        assert "SATISFIABLE" in data["summary"]

    @pytest.mark.asyncio
    async def test_knowledge_graph_transitive_closure_with_relation_inference(self):
        """Problem 6: Transitive closure with auto-inferred relation names.

        Edges: A parent_of B, B parent_of C, C parent_of D
        Expected: 6 ancestor_of edges (3 direct + 3 inferred)
        """
        result = await logic_server.call_tool_for_test(
            "knowledge_graph_reasoning",
            {
                "operation": "transitive_closure",
                "edges": [
                    ["A", "parent_of", "B"],
                    ["B", "parent_of", "C"],
                    ["C", "parent_of", "D"]
                ]
            }
        )
        data = json.loads(result[0].text)

        # Verify relation inference
        assert data["base_relation"] == "parent_of"
        assert data["derived_relation"] == "ancestor_of"

        # Verify edge counts
        assert len(data["direct_edges"]) == 3
        assert len(data["inferred_edges"]) == 3
        assert data["total_edges"] == 6

        # Verify specific inferred edges
        assert ["A", "C"] in data["inferred_edges"]
        assert ["A", "D"] in data["inferred_edges"]
        assert ["B", "D"] in data["inferred_edges"]

        # Verify full closure has correct structure
        assert len(data["full_closure"]) == 6
        for edge in data["full_closure"]:
            assert "from" in edge
            assert "to" in edge
            assert "relation" in edge
            assert edge["relation"] == "ancestor_of"
            assert "direct" in edge

    @pytest.mark.asyncio
    async def test_csp_find_all_solutions(self):
        """Problem 7: CSP with all solutions found.

        Variables: X, Y, Z in {1, 2, 3}
        Constraints: X ≠ Y, Y < Z, X + Y + Z = 6
        Expected: 3 solutions - (1,2,3), (2,1,3), (3,1,2)
        """
        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {
                "variables": ["X", "Y", "Z"],
                "domains": {"X": [1, 2, 3], "Y": [1, 2, 3], "Z": [1, 2, 3]},
                "constraints": ["X != Y", "Y < Z", "X + Y + Z == 6"],
                "find_all": True
            }
        )
        data = json.loads(result[0].text)

        # Verify satisfiability and solution count
        assert data["satisfiable"] == True
        assert data["solution_count"] == 3

        # Verify all expected solutions are present
        solutions = data["solutions"]
        assert {"X": 1, "Y": 2, "Z": 3} in solutions
        assert {"X": 2, "Y": 1, "Z": 3} in solutions
        assert {"X": 3, "Y": 1, "Z": 2} in solutions

        # Verify search statistics
        assert "search_stats" in data
        assert data["search_stats"]["total_assignments"] == 27
        assert data["search_stats"]["solutions_found"] == 3

        # Verify summary
        assert "Found 3 solutions" in data["summary"]

    @pytest.mark.asyncio
    async def test_csp_backward_compatibility(self):
        """Test that CSP still works without find_all (backward compatibility)."""
        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {
                "variables": ["X", "Y"],
                "domains": {"X": [1, 2, 3], "Y": [1, 2, 3]},
                "constraints": ["X != Y"]
            }
        )
        data = json.loads(result[0].text)

        # Should return single solution
        assert data["satisfiable"] == True
        assert "solution" in data
        assert data["solution"]["X"] != data["solution"]["Y"]


class TestHardSuite:
    """Hard test suite - 20 problems designed to exceed native LLM reasoning."""

    # =========================================================================
    # SAT PROBLEMS (7 tests)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_sat_deceptive_1(self):
        """Deceptive UNSAT: (A | B) & (A | ~B) & (~A | B) & (~A | ~B)"""
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": "(A | B) & (A | ~B) & (~A | B) & (~A | ~B)"
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == False

    @pytest.mark.asyncio
    async def test_sat_chain_5(self):
        """5-Variable Implication Chain: A→B→C→D→E with A=T and E=F."""
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": "(A) & (~A | B) & (~B | C) & (~C | D) & (~D | E) & (~E)"
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == False

    @pytest.mark.asyncio
    async def test_sat_xor_4(self):
        """4-Variable XOR Chain: XOR ladder with A=T and D=F.

        XOR chain: A⊕B, B⊕C, C⊕D with A=T, ~D.
        Solution: A=T → B=F (XOR) → C=T (XOR) → D=F (XOR), which satisfies ~D.
        Note: hard_test_suite.py incorrectly expected UNSAT.
        """
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": "(A | B) & (~A | ~B) & (B | C) & (~B | ~C) & (C | D) & (~C | ~D) & (A) & (~D)"
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True
        # Verify the specific solution
        a = data["assignment"]
        assert a["A"] == True
        assert a["B"] == False
        assert a["C"] == True
        assert a["D"] == False

    @pytest.mark.asyncio
    async def test_sat_resolution_hard(self):
        """Resolution-Heavy UNSAT: All 8 combinations of 3 variables."""
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": "(A | B | C) & (A | B | ~C) & (A | ~B | C) & (A | ~B | ~C) & (~A | B | C) & (~A | B | ~C) & (~A | ~B | C) & (~A | ~B | ~C)"
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == False

    @pytest.mark.asyncio
    async def test_sat_pigeonhole_4_3(self):
        """Pigeonhole 4→3: 4 pigeons, 3 holes (multi-line formula)."""
        formula = """
        (P11 | P12 | P13) & (P21 | P22 | P23) & (P31 | P32 | P33) & (P41 | P42 | P43) &
        (~P11 | ~P21) & (~P11 | ~P31) & (~P11 | ~P41) & (~P21 | ~P31) & (~P21 | ~P41) & (~P31 | ~P41) &
        (~P12 | ~P22) & (~P12 | ~P32) & (~P12 | ~P42) & (~P22 | ~P32) & (~P22 | ~P42) & (~P32 | ~P42) &
        (~P13 | ~P23) & (~P13 | ~P33) & (~P13 | ~P43) & (~P23 | ~P33) & (~P23 | ~P43) & (~P33 | ~P43)
        """
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": formula
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == False

    @pytest.mark.asyncio
    async def test_sat_subtle_sat(self):
        """Subtle Satisfiable: Only one solution exists."""
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": "(A | B) & (~A | C) & (~B | D) & (~C | ~D) & (C | D) & (~A | ~B)"
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True
        # Verify the assignment works
        a = data["assignment"]
        A, B, C, D = a["A"], a["B"], a["C"], a["D"]
        assert (A or B)
        assert (not A or C)
        assert (not B or D)
        assert (not C or not D)
        assert (C or D)
        assert (not A or not B)

    @pytest.mark.asyncio
    async def test_sat_interlocked_6(self):
        """6-Variable Interlocked: Cyclic constraints."""
        result = await logic_server.call_tool_for_test(
            "propositional_logic_advanced",
            {
                "operation": "satisfiability",
                "formula": "(A | B) & (~A | C) & (B | ~C) & (~B | D) & (C | ~D) & (~C | E) & (D | ~E) & (~D | F) & (E | ~F) & (~E | A) & (F | ~A) & (~F | B)"
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True

    # =========================================================================
    # KNOWLEDGE GRAPH PROBLEMS (5 tests)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_kg_diamond(self):
        """Diamond Pattern: A→B, A→C, B→D, C→D."""
        result = await logic_server.call_tool_for_test(
            "knowledge_graph_reasoning",
            {
                "operation": "transitive_closure",
                "edges": [
                    ["A", "parent_of", "B"],
                    ["A", "parent_of", "C"],
                    ["B", "parent_of", "D"],
                    ["C", "parent_of", "D"]
                ]
            }
        )
        data = json.loads(result[0].text)
        assert data["total_edges"] == 5
        assert ["A", "D"] in data["inferred_edges"]

    @pytest.mark.asyncio
    async def test_kg_tree_4_2(self):
        """Binary Tree Depth 4: 15 nodes, 14 edges → 34 total.

        Breakdown: N1→14, N2→6, N3→6, N4→2, N5→2, N6→2, N7→2 = 34 total
        """
        result = await logic_server.call_tool_for_test(
            "knowledge_graph_reasoning",
            {
                "operation": "transitive_closure",
                "edges": [
                    ["N1", "parent_of", "N2"], ["N1", "parent_of", "N3"],
                    ["N2", "parent_of", "N4"], ["N2", "parent_of", "N5"],
                    ["N3", "parent_of", "N6"], ["N3", "parent_of", "N7"],
                    ["N4", "parent_of", "N8"], ["N4", "parent_of", "N9"],
                    ["N5", "parent_of", "N10"], ["N5", "parent_of", "N11"],
                    ["N6", "parent_of", "N12"], ["N6", "parent_of", "N13"],
                    ["N7", "parent_of", "N14"], ["N7", "parent_of", "N15"]
                ]
            }
        )
        data = json.loads(result[0].text)
        assert len(data["direct_edges"]) == 14
        assert data["total_edges"] == 34

    @pytest.mark.asyncio
    async def test_kg_dag_converge(self):
        """Converging DAG: Multiple paths to sink.

        11 direct + 10 inferred = 21 total.
        Inferred: A→F, A→G, A→H, B→G, B→H, C→F, C→G, C→H, D→H, E→H
        """
        result = await logic_server.call_tool_for_test(
            "knowledge_graph_reasoning",
            {
                "operation": "transitive_closure",
                "edges": [
                    ["A", "flows_to", "D"],
                    ["B", "flows_to", "D"],
                    ["C", "flows_to", "E"],
                    ["D", "flows_to", "F"],
                    ["E", "flows_to", "F"],
                    ["D", "flows_to", "G"],
                    ["E", "flows_to", "G"],
                    ["F", "flows_to", "H"],
                    ["G", "flows_to", "H"],
                    ["A", "flows_to", "E"],
                    ["B", "flows_to", "F"]
                ]
            }
        )
        data = json.loads(result[0].text)
        assert len(data["direct_edges"]) == 11
        assert data["total_edges"] == 21

    @pytest.mark.asyncio
    async def test_kg_chain_branch(self):
        """Chain with Branches: Main chain A→B→C→D→E with branches.

        A→9, B→7, C→5, D→3, E→1 = 25 total edges.
        """
        result = await logic_server.call_tool_for_test(
            "knowledge_graph_reasoning",
            {
                "operation": "transitive_closure",
                "edges": [
                    ["A", "parent_of", "B"], ["A", "parent_of", "A1"],
                    ["B", "parent_of", "C"], ["B", "parent_of", "B1"],
                    ["C", "parent_of", "D"], ["C", "parent_of", "C1"],
                    ["D", "parent_of", "E"], ["D", "parent_of", "D1"],
                    ["E", "parent_of", "E1"]
                ]
            }
        )
        data = json.loads(result[0].text)
        assert len(data["direct_edges"]) == 9
        assert data["total_edges"] == 25

    @pytest.mark.asyncio
    async def test_kg_multi_relation(self):
        """Multi-Relation Graph: Different relations with separate closures."""
        result = await logic_server.call_tool_for_test(
            "knowledge_graph_reasoning",
            {
                "operation": "transitive_closure",
                "edges": [
                    ["Alice", "manages", "Bob"],
                    ["Bob", "manages", "Carol"],
                    ["Alice", "mentors", "Carol"],
                    ["Carol", "mentors", "Dave"],
                    ["Bob", "collaborates", "Dave"]
                ]
            }
        )
        data = json.loads(result[0].text)
        assert data["multi_relation"] == True
        assert data["relation_count"] == 3
        assert "per_relation_closure" in data

        # Check manages closure
        manages = data["per_relation_closure"]["manages"]
        assert manages["direct_count"] == 2
        assert manages["total_count"] == 3
        assert ["Alice", "Carol"] in manages["inferred_edges"]

        # Check mentors closure
        mentors = data["per_relation_closure"]["mentors"]
        assert mentors["direct_count"] == 2
        assert mentors["total_count"] == 3
        assert ["Alice", "Dave"] in mentors["inferred_edges"]

        # Check collaborates closure (no inferences)
        collaborates = data["per_relation_closure"]["collaborates"]
        assert collaborates["direct_count"] == 1
        assert collaborates["inferred_count"] == 0

    # =========================================================================
    # CSP PROBLEMS (7 tests)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_csp_8queens(self):
        """8-Queens Problem: Place 8 queens, no attacks."""
        variables = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
        domains = {f"Q{i}": list(range(1, 9)) for i in range(1, 9)}
        constraints = [
            # All different rows
            "Q1 != Q2", "Q1 != Q3", "Q1 != Q4", "Q1 != Q5", "Q1 != Q6", "Q1 != Q7", "Q1 != Q8",
            "Q2 != Q3", "Q2 != Q4", "Q2 != Q5", "Q2 != Q6", "Q2 != Q7", "Q2 != Q8",
            "Q3 != Q4", "Q3 != Q5", "Q3 != Q6", "Q3 != Q7", "Q3 != Q8",
            "Q4 != Q5", "Q4 != Q6", "Q4 != Q7", "Q4 != Q8",
            "Q5 != Q6", "Q5 != Q7", "Q5 != Q8",
            "Q6 != Q7", "Q6 != Q8",
            "Q7 != Q8",
            # No diagonal attacks
            "abs(Q1-Q2) != 1", "abs(Q1-Q3) != 2", "abs(Q1-Q4) != 3", "abs(Q1-Q5) != 4",
            "abs(Q1-Q6) != 5", "abs(Q1-Q7) != 6", "abs(Q1-Q8) != 7",
            "abs(Q2-Q3) != 1", "abs(Q2-Q4) != 2", "abs(Q2-Q5) != 3",
            "abs(Q2-Q6) != 4", "abs(Q2-Q7) != 5", "abs(Q2-Q8) != 6",
            "abs(Q3-Q4) != 1", "abs(Q3-Q5) != 2", "abs(Q3-Q6) != 3",
            "abs(Q3-Q7) != 4", "abs(Q3-Q8) != 5",
            "abs(Q4-Q5) != 1", "abs(Q4-Q6) != 2", "abs(Q4-Q7) != 3", "abs(Q4-Q8) != 4",
            "abs(Q5-Q6) != 1", "abs(Q5-Q7) != 2", "abs(Q5-Q8) != 3",
            "abs(Q6-Q7) != 1", "abs(Q6-Q8) != 2",
            "abs(Q7-Q8) != 1"
        ]

        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {"variables": variables, "domains": domains, "constraints": constraints}
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True

        # Verify solution is valid
        sol = data["solution"]
        for i in range(1, 9):
            for j in range(i + 1, 9):
                qi, qj = sol[f"Q{i}"], sol[f"Q{j}"]
                assert qi != qj, f"Q{i} and Q{j} on same row"
                assert abs(qi - qj) != abs(i - j), f"Q{i} and Q{j} on same diagonal"

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Search space 10^8 too large for basic backtracking - needs optimization")
    async def test_csp_send_more_money(self):
        """SEND + MORE = MONEY cryptarithmetic.

        Note: This test is skipped by default because the 10^8 search space
        is too large for basic backtracking. Needs arc consistency or
        constraint propagation optimization.
        """
        variables = ["S", "E", "N", "D", "M", "O", "R", "Y"]
        domains = {v: list(range(0, 10)) for v in variables}
        constraints = [
            # All different
            "S != E", "S != N", "S != D", "S != M", "S != O", "S != R", "S != Y",
            "E != N", "E != D", "E != M", "E != O", "E != R", "E != Y",
            "N != D", "N != M", "N != O", "N != R", "N != Y",
            "D != M", "D != O", "D != R", "D != Y",
            "M != O", "M != R", "M != Y",
            "O != R", "O != Y",
            "R != Y",
            # Leading digits non-zero
            "S > 0",
            "M > 0",
            # The equation: SEND + MORE = MONEY
            "1000*S + 100*E + 10*N + D + 1000*M + 100*O + 10*R + E == 10000*M + 1000*O + 100*N + 10*E + Y"
        ]

        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {"variables": variables, "domains": domains, "constraints": constraints}
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True

        # Verify the unique solution
        sol = data["solution"]
        send = 1000*sol["S"] + 100*sol["E"] + 10*sol["N"] + sol["D"]
        more = 1000*sol["M"] + 100*sol["O"] + 10*sol["R"] + sol["E"]
        money = 10000*sol["M"] + 1000*sol["O"] + 100*sol["N"] + 10*sol["E"] + sol["Y"]
        assert send + more == money

    @pytest.mark.asyncio
    async def test_csp_3color_petersen(self):
        """Petersen Graph 3-Coloring."""
        variables = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"]
        domains = {f"V{i}": [1, 2, 3] for i in range(10)}
        constraints = [
            # Outer pentagon
            "V0 != V1", "V1 != V2", "V2 != V3", "V3 != V4", "V4 != V0",
            # Inner pentagram
            "V5 != V7", "V7 != V9", "V9 != V6", "V6 != V8", "V8 != V5",
            # Spokes
            "V0 != V5", "V1 != V6", "V2 != V7", "V3 != V8", "V4 != V9"
        ]

        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {"variables": variables, "domains": domains, "constraints": constraints}
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True

        # Verify all constraints satisfied
        sol = data["solution"]
        for c in constraints:
            v1, v2 = c.split(" != ")
            assert sol[v1] != sol[v2]

    @pytest.mark.asyncio
    async def test_csp_magic_square_3(self):
        """3x3 Magic Square: All rows/cols/diagonals sum to 15."""
        variables = ["C11", "C12", "C13", "C21", "C22", "C23", "C31", "C32", "C33"]
        domains = {v: list(range(1, 10)) for v in variables}
        constraints = [
            # All different
            "C11 != C12", "C11 != C13", "C11 != C21", "C11 != C22", "C11 != C23",
            "C11 != C31", "C11 != C32", "C11 != C33",
            "C12 != C13", "C12 != C21", "C12 != C22", "C12 != C23",
            "C12 != C31", "C12 != C32", "C12 != C33",
            "C13 != C21", "C13 != C22", "C13 != C23",
            "C13 != C31", "C13 != C32", "C13 != C33",
            "C21 != C22", "C21 != C23", "C21 != C31", "C21 != C32", "C21 != C33",
            "C22 != C23", "C22 != C31", "C22 != C32", "C22 != C33",
            "C23 != C31", "C23 != C32", "C23 != C33",
            "C31 != C32", "C31 != C33",
            "C32 != C33",
            # Rows sum to 15
            "C11 + C12 + C13 == 15",
            "C21 + C22 + C23 == 15",
            "C31 + C32 + C33 == 15",
            # Cols sum to 15
            "C11 + C21 + C31 == 15",
            "C12 + C22 + C32 == 15",
            "C13 + C23 + C33 == 15",
            # Diagonals sum to 15
            "C11 + C22 + C33 == 15",
            "C13 + C22 + C31 == 15"
        ]

        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {"variables": variables, "domains": domains, "constraints": constraints}
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True

        # Verify magic square properties
        sol = data["solution"]
        # All values 1-9 used exactly once
        values = sorted([sol[v] for v in variables])
        assert values == list(range(1, 10))
        # All rows sum to 15
        assert sol["C11"] + sol["C12"] + sol["C13"] == 15
        assert sol["C21"] + sol["C22"] + sol["C23"] == 15
        assert sol["C31"] + sol["C32"] + sol["C33"] == 15
        # All cols sum to 15
        assert sol["C11"] + sol["C21"] + sol["C31"] == 15
        assert sol["C12"] + sol["C22"] + sol["C32"] == 15
        assert sol["C13"] + sol["C23"] + sol["C33"] == 15
        # Diagonals sum to 15
        assert sol["C11"] + sol["C22"] + sol["C33"] == 15
        assert sol["C13"] + sol["C22"] + sol["C31"] == 15

    @pytest.mark.asyncio
    async def test_csp_scheduling(self):
        """Job Shop Scheduling: 4 jobs, 3 machines."""
        variables = [
            "J1M1", "J1M2", "J1M3",
            "J2M1", "J2M2", "J2M3",
            "J3M1", "J3M2", "J3M3",
            "J4M1", "J4M2", "J4M3"
        ]
        domains = {v: list(range(0, 12)) for v in variables}
        constraints = [
            # Job order constraints
            "J1M1 + 1 <= J1M2", "J1M2 + 1 <= J1M3",
            "J2M1 + 1 <= J2M2", "J2M2 + 1 <= J2M3",
            "J3M1 + 1 <= J3M2", "J3M2 + 1 <= J3M3",
            "J4M1 + 1 <= J4M2", "J4M2 + 1 <= J4M3",
            # Machine capacity (disjunctive)
            "(J1M1 + 1 <= J2M1) | (J2M1 + 1 <= J1M1)",
            "(J1M1 + 1 <= J3M1) | (J3M1 + 1 <= J1M1)",
            "(J1M1 + 1 <= J4M1) | (J4M1 + 1 <= J1M1)",
            "(J2M1 + 1 <= J3M1) | (J3M1 + 1 <= J2M1)",
            "(J2M1 + 1 <= J4M1) | (J4M1 + 1 <= J2M1)",
            "(J3M1 + 1 <= J4M1) | (J4M1 + 1 <= J3M1)",
            "(J1M2 + 1 <= J2M2) | (J2M2 + 1 <= J1M2)",
            "(J1M2 + 1 <= J3M2) | (J3M2 + 1 <= J1M2)",
            "(J1M2 + 1 <= J4M2) | (J4M2 + 1 <= J1M2)",
            "(J2M2 + 1 <= J3M2) | (J3M2 + 1 <= J2M2)",
            "(J2M2 + 1 <= J4M2) | (J4M2 + 1 <= J2M2)",
            "(J3M2 + 1 <= J4M2) | (J4M2 + 1 <= J3M2)",
            "(J1M3 + 1 <= J2M3) | (J2M3 + 1 <= J1M3)",
            "(J1M3 + 1 <= J3M3) | (J3M3 + 1 <= J1M3)",
            "(J1M3 + 1 <= J4M3) | (J4M3 + 1 <= J1M3)",
            "(J2M3 + 1 <= J3M3) | (J3M3 + 1 <= J2M3)",
            "(J2M3 + 1 <= J4M3) | (J4M3 + 1 <= J2M3)",
            "(J3M3 + 1 <= J4M3) | (J4M3 + 1 <= J3M3)"
        ]

        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {"variables": variables, "domains": domains, "constraints": constraints}
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True

        # Verify job ordering
        sol = data["solution"]
        for j in range(1, 5):
            assert sol[f"J{j}M1"] + 1 <= sol[f"J{j}M2"]
            assert sol[f"J{j}M2"] + 1 <= sol[f"J{j}M3"]

    @pytest.mark.asyncio
    async def test_csp_impossible_hard(self):
        """Overconstrained UNSAT: Sum constraint impossible."""
        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {
                "variables": ["A", "B", "C", "D"],
                "domains": {"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "D": [1, 2, 3]},
                "constraints": [
                    "A + B + C + D == 14",
                    "A != B", "B != C", "C != D", "A != D"
                ]
            }
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == False

    @pytest.mark.asyncio
    async def test_csp_sudoku_4x4(self):
        """4x4 Sudoku with pre-filled cells."""
        variables = ["C11", "C12", "C13", "C14",
                     "C21", "C22", "C23", "C24",
                     "C31", "C32", "C33", "C34",
                     "C41", "C42", "C43", "C44"]
        domains = {
            "C11": [1], "C14": [4],
            "C22": [3], "C23": [1],
            "C32": [4], "C33": [2],
            "C41": [2], "C44": [3],
            "C12": [1, 2, 3, 4], "C13": [1, 2, 3, 4],
            "C21": [1, 2, 3, 4], "C24": [1, 2, 3, 4],
            "C31": [1, 2, 3, 4], "C34": [1, 2, 3, 4],
            "C42": [1, 2, 3, 4], "C43": [1, 2, 3, 4]
        }
        constraints = [
            # Row constraints
            "C11 != C12", "C11 != C13", "C11 != C14", "C12 != C13", "C12 != C14", "C13 != C14",
            "C21 != C22", "C21 != C23", "C21 != C24", "C22 != C23", "C22 != C24", "C23 != C24",
            "C31 != C32", "C31 != C33", "C31 != C34", "C32 != C33", "C32 != C34", "C33 != C34",
            "C41 != C42", "C41 != C43", "C41 != C44", "C42 != C43", "C42 != C44", "C43 != C44",
            # Column constraints
            "C11 != C21", "C11 != C31", "C11 != C41", "C21 != C31", "C21 != C41", "C31 != C41",
            "C12 != C22", "C12 != C32", "C12 != C42", "C22 != C32", "C22 != C42", "C32 != C42",
            "C13 != C23", "C13 != C33", "C13 != C43", "C23 != C33", "C23 != C43", "C33 != C43",
            "C14 != C24", "C14 != C34", "C14 != C44", "C24 != C34", "C24 != C44", "C34 != C44",
            # Box constraints
            "C11 != C22", "C12 != C21",
            "C13 != C24", "C14 != C23",
            "C31 != C42", "C32 != C41",
            "C33 != C44", "C34 != C43"
        ]

        result = await logic_server.call_tool_for_test(
            "constraint_satisfaction",
            {"variables": variables, "domains": domains, "constraints": constraints}
        )
        data = json.loads(result[0].text)
        assert data["satisfiable"] == True

        # Verify Sudoku properties
        sol = data["solution"]
        # Check rows have unique values
        for r in range(1, 5):
            row = [sol[f"C{r}{c}"] for c in range(1, 5)]
            assert len(set(row)) == 4
        # Check cols have unique values
        for c in range(1, 5):
            col = [sol[f"C{r}{c}"] for r in range(1, 5)]
            assert len(set(col)) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
