"""
Metrics and validation for comparing LLM answers to expected results.

Uses SymPy for symbolic equivalence checking to accurately determine
if an LLM's answer matches the expected answer mathematically,
not just as a string comparison.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp
from sympy import symbols, simplify, sympify, Matrix
from sympy.logic.boolalg import Boolean


def extract_mathematical_expression(text: str) -> str:
    """
    Extract mathematical expression from natural language response.

    LLMs often provide explanations. This attempts to find the actual answer.

    Args:
        text: The LLM's response text

    Returns:
        Extracted expression string
    """
    if not text:
        return ""

    # Clean up the text
    text = text.strip()

    # Common patterns for answers
    patterns = [
        r"(?:answer|result|solution)(?:\s+is)?[:\s]+(.+?)(?:\.|$|\n)",
        r"^(.+?)(?:\.|$|\n)",  # First line/sentence
        r"=\s*(.+?)(?:\.|$|\n)",  # After equals sign
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).strip()
            # Remove common trailing phrases
            extracted = re.sub(r"\s*(where|when|for|given).*$", "", extracted, flags=re.IGNORECASE)
            return extracted

    # If no pattern matches, return the original text
    return text


def parse_to_sympy(expr_str: str) -> Optional[Any]:
    """
    Parse a string expression to SymPy format.

    Handles common LLM output formats and converts to SymPy.

    Args:
        expr_str: Expression string

    Returns:
        SymPy expression or None if parsing fails
    """
    if not expr_str:
        return None

    try:
        # Clean the string
        expr_str = expr_str.strip()

        # Remove LaTeX formatting if present
        expr_str = expr_str.replace("\\", "")

        # Common replacements for LLM outputs
        replacements = {
            "^": "**",
            "√": "sqrt",
            "π": "pi",
            "∞": "oo",
            "×": "*",
            "÷": "/",
        }

        for old, new in replacements.items():
            expr_str = expr_str.replace(old, new)

        # Try to parse as SymPy expression
        return sympify(expr_str, evaluate=False)

    except (SyntaxError, ValueError, TypeError, AttributeError):
        return None


def validate_symbolic_equivalence(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if LLM answer is symbolically equivalent to expected answer.

    Args:
        llm_answer: The LLM's answer (may be in natural language)
        expected: Expected answer in SymPy format

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract and parse LLM answer
        extracted = extract_mathematical_expression(llm_answer)
        llm_expr = parse_to_sympy(extracted)

        if llm_expr is None:
            return False, f"Could not parse LLM answer: '{extracted}'"

        # Parse expected answer
        expected_expr = sympify(expected)

        # Simplify both expressions
        llm_simplified = simplify(llm_expr)
        expected_simplified = simplify(expected_expr)

        # Check equality
        difference = simplify(llm_simplified - expected_simplified)

        if difference == 0:
            return True, "Symbolically equivalent"
        else:
            return False, f"LLM: {llm_simplified}, Expected: {expected_simplified}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_symbolic_equivalence_ignore_constant(
    llm_answer: str, expected: str
) -> Tuple[bool, str]:
    """
    Check symbolic equivalence ignoring additive constants (for indefinite integrals).

    Args:
        llm_answer: The LLM's answer
        expected: Expected answer

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        extracted = extract_mathematical_expression(llm_answer)
        llm_expr = parse_to_sympy(extracted)

        if llm_expr is None:
            return False, f"Could not parse LLM answer: '{extracted}'"

        expected_expr = sympify(expected)

        # Get free symbols
        all_symbols = llm_expr.free_symbols.union(expected_expr.free_symbols)

        if not all_symbols:
            return validate_symbolic_equivalence(llm_answer, expected)

        # Pick a symbol to differentiate with respect to
        var = list(all_symbols)[0]

        # Take derivatives (eliminates constants)
        llm_derivative = sp.diff(llm_expr, var)
        expected_derivative = sp.diff(expected_expr, var)

        # Simplify and compare
        difference = simplify(llm_derivative - expected_derivative)

        if difference == 0:
            return True, "Symbolically equivalent (ignoring constant)"
        else:
            return False, f"Derivatives differ: LLM: {llm_derivative}, Expected: {expected_derivative}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_list_equivalence(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if LLM answer matches expected list (e.g., equation solutions).

    Args:
        llm_answer: The LLM's answer
        expected: Expected list as string (e.g., "[1, 2, 3]")

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract answer
        extracted = extract_mathematical_expression(llm_answer)

        # Try to parse as list
        # Handle different formats: [1, 2], {1, 2}, "1, 2", "x=1, x=2"
        extracted = extracted.replace("{", "[").replace("}", "]")

        # Extract numbers/expressions from the string
        import ast
        try:
            llm_list = ast.literal_eval(extracted)
            if not isinstance(llm_list, list):
                llm_list = [llm_list]
        except (ValueError, SyntaxError):
            # Try to extract numbers with regex
            numbers = re.findall(r"-?\d+\.?\d*", extracted)
            if numbers:
                llm_list = [float(n) if "." in n else int(n) for n in numbers]
            else:
                return False, f"Could not parse list from: '{extracted}'"

        # Parse expected list
        expected_list = ast.literal_eval(expected)

        # Convert to SymPy and sort
        llm_sympy = sorted([sympify(x) for x in llm_list], key=str)
        expected_sympy = sorted([sympify(x) for x in expected_list], key=str)

        # Compare
        if len(llm_sympy) != len(expected_sympy):
            return False, f"Different lengths: LLM has {len(llm_sympy)}, expected {len(expected_sympy)}"

        for llm_val, exp_val in zip(llm_sympy, expected_sympy):
            if simplify(llm_val - exp_val) != 0:
                return False, f"Mismatch: {llm_sympy} vs {expected_sympy}"

        return True, "Lists match"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_dict_equivalence(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if LLM answer matches expected dictionary (e.g., system solutions).

    Args:
        llm_answer: The LLM's answer
        expected: Expected dict as string (e.g., "{x: 1, y: 2}")

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract answer
        extracted = extract_mathematical_expression(llm_answer)

        # Try to parse expected dict
        import ast
        expected_dict = ast.literal_eval(expected.replace("{", "{'").replace(":", "':").replace(",", ",'").replace(" ", ""))

        # Extract key-value pairs from LLM response
        # Handle formats like "x=1, y=2" or "{x: 1, y: 2}"
        pattern = r"([a-zA-Z_]\w*)\s*[=:]\s*(-?\d+\.?\d*)"
        matches = re.findall(pattern, extracted)

        if not matches:
            return False, f"Could not parse dict from: '{extracted}'"

        llm_dict = {k: sympify(v) for k, v in matches}

        # Compare
        if set(llm_dict.keys()) != set(expected_dict.keys()):
            return False, f"Different keys: {set(llm_dict.keys())} vs {set(expected_dict.keys())}"

        for key in expected_dict:
            expected_val = sympify(expected_dict[key])
            llm_val = llm_dict.get(key)
            if llm_val is None or simplify(llm_val - expected_val) != 0:
                return False, f"Mismatch at {key}: {llm_val} vs {expected_val}"

        return True, "Dicts match"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_matrix_equivalence(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if LLM answer matches expected matrix.

    Args:
        llm_answer: The LLM's answer
        expected: Expected matrix as string

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        import ast

        # Extract answer
        extracted = extract_mathematical_expression(llm_answer)

        # Parse both matrices
        llm_matrix_data = ast.literal_eval(extracted)
        expected_matrix_data = ast.literal_eval(expected)

        # Convert to SymPy matrices
        llm_matrix = Matrix(llm_matrix_data)
        expected_matrix = Matrix(expected_matrix_data)

        # Compare dimensions
        if llm_matrix.shape != expected_matrix.shape:
            return False, f"Different dimensions: {llm_matrix.shape} vs {expected_matrix.shape}"

        # Compare elements
        diff = simplify(llm_matrix - expected_matrix)
        if diff == Matrix.zeros(*llm_matrix.shape):
            return True, "Matrices match"
        else:
            return False, f"Matrices differ"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_logic_equivalence(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if logical expressions are equivalent.

    Args:
        llm_answer: The LLM's answer
        expected: Expected logical expression

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        from sympy.logic.boolalg import simplify_logic, to_cnf

        # Extract answer
        extracted = extract_mathematical_expression(llm_answer)

        # Parse logical expressions
        llm_expr = sympify(extracted)
        expected_expr = sympify(expected)

        # Simplify both
        llm_simplified = simplify_logic(llm_expr)
        expected_simplified = simplify_logic(expected_expr)

        # Check equivalence using CNF
        llm_cnf = to_cnf(llm_simplified)
        expected_cnf = to_cnf(expected_simplified)

        if llm_cnf == expected_cnf:
            return True, "Logically equivalent"
        else:
            return False, f"LLM: {llm_cnf}, Expected: {expected_cnf}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_boolean_equivalence(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if boolean values match.

    Args:
        llm_answer: The LLM's answer
        expected: Expected boolean value

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract answer
        extracted = extract_mathematical_expression(llm_answer).lower()

        # Parse boolean
        llm_bool = extracted in ["true", "t", "1", "yes"]
        expected_bool = expected.lower() in ["true", "t", "1", "yes"]

        if llm_bool == expected_bool:
            return True, "Boolean match"
        else:
            return False, f"LLM: {llm_bool}, Expected: {expected_bool}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_numerical_close(llm_answer: str, expected: str, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Check if numerical values are close within tolerance.

    Args:
        llm_answer: The LLM's answer
        expected: Expected numerical value
        tolerance: Acceptable difference

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract answer
        extracted = extract_mathematical_expression(llm_answer)

        # Parse numbers
        llm_num = float(extracted)
        expected_num = float(expected)

        # Check closeness
        if abs(llm_num - expected_num) <= tolerance:
            return True, f"Numerically close (diff: {abs(llm_num - expected_num):.2e})"
        else:
            return False, f"LLM: {llm_num}, Expected: {expected_num}, Diff: {abs(llm_num - expected_num):.2e}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_field_exists(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if a specific field exists in JSON response.

    Args:
        llm_answer: The LLM's answer (should be JSON)
        expected: Field name to check for

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        import json
        parsed = json.loads(llm_answer)

        if expected in parsed:
            return True, f"Field '{expected}' exists"
        else:
            return False, f"Field '{expected}' not found in response"

    except json.JSONDecodeError:
        return False, "Response is not valid JSON"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_string_match(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if strings match exactly (case-sensitive).

    Args:
        llm_answer: The LLM's answer
        expected: Expected string

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract answer
        extracted = extract_mathematical_expression(llm_answer).strip()
        expected_clean = expected.strip()

        if extracted == expected_clean:
            return True, "Exact string match"
        else:
            return False, f"LLM: '{extracted}', Expected: '{expected_clean}'"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_contains_pattern(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if answer contains expected substring/pattern.

    Args:
        llm_answer: The LLM's answer
        expected: Expected substring or pattern

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        if expected in llm_answer:
            return True, f"Contains '{expected}'"
        else:
            return False, f"Does not contain '{expected}'"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_numeric_exact(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if numeric values match exactly (no tolerance).

    Args:
        llm_answer: The LLM's answer
        expected: Expected numeric value

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract answer
        extracted = extract_mathematical_expression(llm_answer)

        # Parse numbers
        llm_num = float(extracted)
        expected_num = float(expected)

        if llm_num == expected_num:
            return True, f"Exact numeric match: {llm_num}"
        else:
            return False, f"LLM: {llm_num}, Expected: {expected_num}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_numeric_ge(llm_answer: str, expected: str) -> Tuple[bool, str]:
    """
    Check if numeric value is greater than or equal to expected.

    Args:
        llm_answer: The LLM's answer
        expected: Minimum expected value

    Returns:
        Tuple of (is_correct, explanation)
    """
    try:
        # Extract answer
        extracted = extract_mathematical_expression(llm_answer)

        # Parse numbers
        llm_num = float(extracted)
        expected_num = float(expected)

        if llm_num >= expected_num:
            return True, f"Value {llm_num} >= {expected_num}"
        else:
            return False, f"LLM: {llm_num} < Expected minimum: {expected_num}"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_answer(llm_answer: str, expected: str, validation_type: str) -> Tuple[bool, str]:
    """
    Validate an LLM answer against expected result using the specified method.

    Args:
        llm_answer: The LLM's answer
        expected: Expected answer
        validation_type: Type of validation to perform

    Returns:
        Tuple of (is_correct, explanation)
    """
    validators = {
        "symbolic_equivalence": validate_symbolic_equivalence,
        "symbolic_equivalence_ignore_constant": validate_symbolic_equivalence_ignore_constant,
        "list_equivalence": validate_list_equivalence,
        "dict_equivalence": validate_dict_equivalence,
        "matrix_equivalence": validate_matrix_equivalence,
        "logic_equivalence": validate_logic_equivalence,
        "boolean_equivalence": validate_boolean_equivalence,
        "numerical_close": validate_numerical_close,
        "field_exists": validate_field_exists,
        "string_match": validate_string_match,
        "contains_pattern": validate_contains_pattern,
        "numeric_exact": validate_numeric_exact,
        "numeric_ge": validate_numeric_ge,
    }

    validator = validators.get(validation_type)
    if not validator:
        return False, f"Unknown validation type: {validation_type}"

    return validator(llm_answer, expected)


class BenchmarkMetrics:
    """Track and calculate benchmark metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        test_id: str,
        provider: str,
        correct: bool,
        latency_ms: int,
        cost: float,
        explanation: str
    ):
        """Add a test result."""
        self.results.append({
            "test_id": test_id,
            "provider": provider,
            "correct": correct,
            "latency_ms": latency_ms,
            "cost": cost,
            "explanation": explanation
        })

    def get_accuracy(self, provider: Optional[str] = None) -> float:
        """Calculate accuracy percentage."""
        results = self.results
        if provider:
            results = [r for r in results if r["provider"] == provider]

        if not results:
            return 0.0

        correct = sum(1 for r in results if r["correct"])
        return (correct / len(results)) * 100

    def get_average_latency(self, provider: Optional[str] = None) -> float:
        """Calculate average latency in milliseconds."""
        results = self.results
        if provider:
            results = [r for r in results if r["provider"] == provider]

        if not results:
            return 0.0

        return sum(r["latency_ms"] for r in results) / len(results)

    def get_total_cost(self, provider: Optional[str] = None) -> float:
        """Calculate total cost in USD."""
        results = self.results
        if provider:
            results = [r for r in results if r["provider"] == provider]

        return sum(r["cost"] for r in results)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        providers = list(set(r["provider"] for r in self.results))

        summary = {
            "total_tests": len(self.results),
            "providers": {}
        }

        for provider in providers:
            provider_results = [r for r in self.results if r["provider"] == provider]
            correct = sum(1 for r in provider_results if r["correct"])

            summary["providers"][provider] = {
                "accuracy": round((correct / len(provider_results)) * 100, 2),
                "correct": correct,
                "total": len(provider_results),
                "avg_latency_ms": round(self.get_average_latency(provider), 2),
                "total_cost": round(self.get_total_cost(provider), 4)
            }

        return summary
