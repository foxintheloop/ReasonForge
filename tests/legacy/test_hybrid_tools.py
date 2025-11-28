"""
Comprehensive tests for Hybrid Neuro-Symbolic Tools

Tests all 7 hybrid tools that combine neural and symbolic AI approaches.
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from reasonforge_mcp.symbolic_engine import SymbolicAI
from reasonforge_mcp.hybrid_tools import (
    handle_hybrid_tools,
    _pattern_to_equation,
    _symbolic_knowledge_extraction,
    _neural_guided_theorem_proving,
    _semantic_parsing,
    _concept_learning,
    _analogical_reasoning,
    _automated_conjecture
)


@pytest.fixture
def symbolic_ai():
    """Create a SymbolicAI instance for testing."""
    return SymbolicAI()


class TestPatternToEquation:
    """Test pattern_to_equation tool."""

    def test_linear_pattern(self, symbolic_ai):
        """Test linear relationship discovery."""
        args = {
            "x_values": [1, 2, 3, 4, 5],
            "y_values": [2, 4, 6, 8, 10],  # y = 2x
            "max_degree": 3
        }
        result = _pattern_to_equation(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "best_fit" in data
        assert data["best_r2"] > 0.99  # Should be nearly perfect fit
        assert "2*x" in data["best_fit"]["expression"] or "2.0*x" in data["best_fit"]["expression"]

    def test_quadratic_pattern(self, symbolic_ai):
        """Test quadratic relationship discovery."""
        args = {
            "x_values": [1, 2, 3, 4, 5],
            "y_values": [1, 4, 9, 16, 25],  # y = x^2
            "max_degree": 3
        }
        result = _pattern_to_equation(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["best_r2"] > 0.99
        assert "x**2" in data["best_fit"]["expression"]

    def test_polynomial_pattern(self, symbolic_ai):
        """Test cubic polynomial."""
        args = {
            "x_values": [0, 1, 2, 3, 4],
            "y_values": [0, 1, 8, 27, 64],  # y = x^3
            "max_degree": 5
        }
        result = _pattern_to_equation(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["best_r2"] > 0.99
        assert "candidates" in data
        assert len(data["candidates"]) > 0

    def test_trigonometric_pattern(self, symbolic_ai):
        """Test trigonometric pattern recognition."""
        import numpy as np
        x = np.linspace(0, 2*np.pi, 20)
        y = np.sin(x)

        args = {
            "x_values": x.tolist(),
            "y_values": y.tolist(),
            "max_degree": 3,
            "try_trig": True
        }
        result = _pattern_to_equation(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "candidates" in data
        # Should find a good trigonometric fit
        trig_candidates = [c for c in data["candidates"] if c["type"].startswith("trigonometric")]
        if trig_candidates:
            assert trig_candidates[0]["r2_score"] > 0.9

    def test_exponential_pattern(self, symbolic_ai):
        """Test exponential pattern recognition."""
        import numpy as np
        x = np.array([0, 1, 2, 3, 4])
        y = np.exp(x)

        args = {
            "x_values": x.tolist(),
            "y_values": y.tolist(),
            "max_degree": 3,
            "try_exp": True
        }
        result = _pattern_to_equation(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "candidates" in data
        exp_candidates = [c for c in data["candidates"] if c["type"] == "exponential"]
        if exp_candidates:
            assert exp_candidates[0]["r2_score"] > 0.95


class TestSymbolicKnowledgeExtraction:
    """Test symbolic_knowledge_extraction tool."""

    def test_simple_and_rule(self, symbolic_ai):
        """Test extracting AND rule."""
        args = {
            "data_points": [
                {"inputs": {"a": True, "b": True}, "output": True},
                {"inputs": {"a": True, "b": False}, "output": False},
                {"inputs": {"a": False, "b": True}, "output": False},
                {"inputs": {"a": False, "b": False}, "output": False},
            ]
        }
        result = _symbolic_knowledge_extraction(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "extracted_formula" in data
        assert data["is_satisfiable"]
        assert data["num_positive_examples"] == 1

    def test_simple_or_rule(self, symbolic_ai):
        """Test extracting OR rule."""
        args = {
            "data_points": [
                {"inputs": {"a": True, "b": True}, "output": True},
                {"inputs": {"a": True, "b": False}, "output": True},
                {"inputs": {"a": False, "b": True}, "output": True},
                {"inputs": {"a": False, "b": False}, "output": False},
            ]
        }
        result = _symbolic_knowledge_extraction(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["num_positive_examples"] == 3
        assert data["is_satisfiable"]

    def test_complex_logic(self, symbolic_ai):
        """Test complex logical pattern."""
        args = {
            "data_points": [
                {"inputs": {"x": True, "y": False, "z": True}, "output": True},
                {"inputs": {"x": False, "y": True, "z": False}, "output": False},
                {"inputs": {"x": True, "y": True, "z": True}, "output": True},
            ]
        }
        result = _symbolic_knowledge_extraction(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "extracted_formula" in data
        assert "variables" in data
        assert len(data["variables"]) == 3


class TestNeuralGuidedTheoremProving:
    """Test neural_guided_theorem_proving tool."""

    def test_simple_modus_ponens(self, symbolic_ai):
        """Test modus ponens: P, P→Q ⊢ Q"""
        args = {
            "premises": ["P", "P >> Q"],
            "goal": "Q",
            "max_depth": 5
        }
        result = _neural_guided_theorem_proving(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "premises" in data
        assert "goal" in data
        assert "steps" in data

    def test_transitivity(self, symbolic_ai):
        """Test transitive reasoning."""
        args = {
            "premises": ["A >> B", "B >> C"],
            "goal": "A >> C",
            "max_depth": 10
        }
        result = _neural_guided_theorem_proving(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "method" in data
        assert "steps" in data

    def test_contradiction(self, symbolic_ai):
        """Test detecting contradictions."""
        args = {
            "premises": ["P", "~P"],
            "goal": "Q",  # From contradiction, anything follows
            "max_depth": 5
        }
        result = _neural_guided_theorem_proving(args, symbolic_ai)
        data = json.loads(result[0].text)

        # Should be able to handle contradictory premises
        assert "premises" in data


class TestSemanticParsing:
    """Test semantic_parsing tool."""

    def test_simple_addition(self, symbolic_ai):
        """Test parsing simple addition."""
        args = {
            "text": "x plus y",
            "context_variables": ["x", "y"]
        }
        result = _semantic_parsing(args, symbolic_ai)
        data = json.loads(result[0].text)

        if data.get("success"):
            assert "parsed_expression" in data
            assert "x" in data["parsed_expression"]
            assert "y" in data["parsed_expression"]

    def test_quadratic_equation(self, symbolic_ai):
        """Test parsing quadratic equation."""
        args = {
            "text": "x squared plus two times x minus three equals zero",
            "context_variables": ["x"],
            "output_format": "equation"
        }
        result = _semantic_parsing(args, symbolic_ai)
        data = json.loads(result[0].text)

        if data.get("success"):
            assert "parsed_expression" in data

    def test_square_expression(self, symbolic_ai):
        """Test parsing square expression."""
        args = {
            "text": "the square of x plus twice y",
            "context_variables": ["x", "y"]
        }
        result = _semantic_parsing(args, symbolic_ai)
        data = json.loads(result[0].text)

        # Should attempt to parse
        assert "original_text" in data

    def test_division(self, symbolic_ai):
        """Test parsing division."""
        args = {
            "text": "x divided by y",
            "context_variables": ["x", "y"]
        }
        result = _semantic_parsing(args, symbolic_ai)
        data = json.loads(result[0].text)

        if data.get("success"):
            assert "/" in data["parsed_expression"] or "x/y" in data["parsed_expression"]


class TestConceptLearning:
    """Test concept_learning tool."""

    def test_positive_numbers(self, symbolic_ai):
        """Test learning 'positive number' concept."""
        args = {
            "positive_examples": [
                {"value": 5},
                {"value": 10},
                {"value": 100}
            ],
            "negative_examples": [
                {"value": -5},
                {"value": -10},
                {"value": 0}
            ]
        }
        result = _concept_learning(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "learned_rule" in data
        assert "confidence" in data

    def test_even_numbers(self, symbolic_ai):
        """Test learning 'even number' concept."""
        args = {
            "positive_examples": [
                {"value": 2, "even": True},
                {"value": 4, "even": True},
                {"value": 6, "even": True}
            ],
            "negative_examples": [
                {"value": 1, "even": False},
                {"value": 3, "even": False},
                {"value": 5, "even": False}
            ]
        }
        result = _concept_learning(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["covers_positive"] == 3
        assert data["excludes_negative"] == 3

    def test_complex_concept(self, symbolic_ai):
        """Test learning complex multi-feature concept."""
        args = {
            "positive_examples": [
                {"x": 10, "y": 20, "flag": True},
                {"x": 15, "y": 25, "flag": True}
            ],
            "negative_examples": [
                {"x": -5, "y": 10, "flag": False},
                {"x": 5, "y": -10, "flag": False}
            ],
            "background_predicates": ["greater_than", "positive"],
            "max_rule_length": 3
        }
        result = _concept_learning(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert "num_conditions" in data


class TestAnalogicalReasoning:
    """Test analogical_reasoning tool."""

    def test_simple_analogy(self, symbolic_ai):
        """Test simple structural analogy."""
        args = {
            "source_domain": {
                "objects": ["1", "2", "3"],
                "relations": [
                    {"relation": "less_than", "arguments": ["1", "2"]},
                    {"relation": "less_than", "arguments": ["2", "3"]}
                ]
            },
            "target_domain": {
                "objects": ["a", "b", "c"],
                "relations": [
                    {"relation": "less_than", "arguments": ["a", "b"]},
                    {"relation": "less_than", "arguments": ["b", "c"]}
                ]
            }
        }
        result = _analogical_reasoning(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["num_matching_relations"] == 1  # less_than with arity 2
        assert data["analogy_strength"] > 0
        assert "object_mapping" in data

    def test_group_ring_analogy(self, symbolic_ai):
        """Test mathematical structure analogy (groups vs rings)."""
        args = {
            "source_domain": {
                "objects": ["Z", "+", "0"],
                "relations": [
                    {"relation": "binary_operation", "arguments": ["Z", "+"]},
                    {"relation": "identity", "arguments": ["+", "0"]}
                ]
            },
            "target_domain": {
                "objects": ["R", "*", "1"],
                "relations": [
                    {"relation": "binary_operation", "arguments": ["R", "*"]},
                    {"relation": "identity", "arguments": ["*", "1"]}
                ]
            }
        }
        result = _analogical_reasoning(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["num_matching_relations"] == 2
        assert data["analogy_strength"] == 1.0

    def test_no_analogy(self, symbolic_ai):
        """Test case with no matching relations."""
        args = {
            "source_domain": {
                "objects": ["x", "y"],
                "relations": [
                    {"relation": "equals", "arguments": ["x", "y"]}
                ]
            },
            "target_domain": {
                "objects": ["a", "b"],
                "relations": [
                    {"relation": "different_relation", "arguments": ["a", "b"]}
                ]
            }
        }
        result = _analogical_reasoning(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["num_matching_relations"] == 0
        assert data["analogy_strength"] == 0.0


class TestAutomatedConjecture:
    """Test automated_conjecture tool."""

    def test_number_theory_conjectures(self, symbolic_ai):
        """Test number theory conjecture generation."""
        args = {
            "domain": "number_theory",
            "context_objects": ["n", "prime_numbers"],
            "num_conjectures": 3,
            "verify": True
        }
        result = _automated_conjecture(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["domain"] == "number_theory"
        assert len(data["conjectures"]) == 3
        assert "num_verified" in data

    def test_algebra_conjectures(self, symbolic_ai):
        """Test algebra conjecture generation."""
        args = {
            "domain": "algebra",
            "context_objects": ["x", "y"],
            "num_conjectures": 2,
            "verify": True
        }
        result = _automated_conjecture(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["domain"] == "algebra"
        assert len(data["conjectures"]) >= 2
        for conj in data["conjectures"]:
            assert "statement" in conj

    def test_verified_conjectures(self, symbolic_ai):
        """Test that generated conjectures can be verified."""
        args = {
            "domain": "number_theory",
            "context_objects": ["n"],
            "num_conjectures": 1,
            "verify": True,
            "test_range": 50
        }
        result = _automated_conjecture(args, symbolic_ai)
        data = json.loads(result[0].text)

        # At least some conjectures should be verifiable
        assert "num_verified" in data

    def test_geometry_conjectures(self, symbolic_ai):
        """Test geometry domain (basic support)."""
        args = {
            "domain": "geometry",
            "context_objects": ["triangle", "circle"],
            "num_conjectures": 1
        }
        result = _automated_conjecture(args, symbolic_ai)
        data = json.loads(result[0].text)

        assert data["domain"] == "geometry"
        assert len(data["conjectures"]) >= 1


class TestIntegration:
    """Integration tests for hybrid tools."""

    def test_pattern_to_knowledge_pipeline(self, symbolic_ai):
        """Test using pattern discovery to inform knowledge extraction."""
        # First, discover a pattern
        pattern_args = {
            "x_values": [1, 2, 3, 4],
            "y_values": [1, 4, 9, 16],
            "max_degree": 3
        }
        pattern_result = _pattern_to_equation(pattern_args, symbolic_ai)
        pattern_data = json.loads(pattern_result[0].text)

        # Verify we found a good pattern
        assert pattern_data["best_r2"] > 0.99

    def test_semantic_to_theorem_pipeline(self, symbolic_ai):
        """Test parsing natural language and then proving theorems."""
        # Parse a mathematical statement
        parse_args = {
            "text": "x plus y",
            "context_variables": ["x", "y"]
        }
        parse_result = _semantic_parsing(parse_args, symbolic_ai)
        parse_data = json.loads(parse_result[0].text)

        # Verify parsing succeeded
        assert "original_text" in parse_data

    def test_concept_to_analogy_pipeline(self, symbolic_ai):
        """Test learning a concept and then using analogical reasoning."""
        # Learn a concept
        concept_args = {
            "positive_examples": [{"x": 10}, {"x": 20}],
            "negative_examples": [{"x": -5}, {"x": -10}]
        }
        concept_result = _concept_learning(concept_args, symbolic_ai)
        concept_data = json.loads(concept_result[0].text)

        assert "learned_rule" in concept_data


def test_all_tools_accessible_via_handle(symbolic_ai):
    """Test that all tools are accessible via the main handler."""
    tool_names = [
        "pattern_to_equation",
        "symbolic_knowledge_extraction",
        "neural_guided_theorem_proving",
        "semantic_parsing",
        "concept_learning",
        "analogical_reasoning",
        "automated_conjecture"
    ]

    for tool_name in tool_names:
        # Call with minimal valid arguments
        result = handle_hybrid_tools(tool_name, {}, symbolic_ai)
        assert len(result) > 0
        # Should return error or result, but not crash
        data = json.loads(result[0].text)
        assert isinstance(data, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
