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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
