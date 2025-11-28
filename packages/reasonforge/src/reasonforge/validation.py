"""
Input validation utilities for ReasonForge MCP Server.

This module provides safe wrappers around SymPy's sympify() and other input parsing
functions to prevent code execution and other security vulnerabilities.
"""

import re
from typing import Any, Optional, Union, List
import sympy as sp
from sympy.core.sympify import SympifyError


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


# Dangerous patterns that could lead to code execution
DANGEROUS_PATTERNS = [
    r'__\w+__',          # Dunder methods
    r'\beval\b',         # eval() function
    r'\bexec\b',         # exec() function
    r'\bcompile\b',      # compile() function
    r'\bimport\b',       # import statements
    r'\b__import__\b',   # __import__() function
    r'\bopen\b',         # file operations
    r'\bgetattr\b',      # getattr() function
    r'\bsetattr\b',      # setattr() function
    r'\bdelattr\b',      # delattr() function
    r'\bglobals\b',      # globals() function
    r'\blocals\b',       # locals() function
    r'\bvars\b',         # vars() function
    r'\bdir\b',          # dir() function
    r'\blambda\b',       # lambda expressions (can be dangerous)
]

# Maximum input lengths to prevent DoS
MAX_EXPRESSION_LENGTH = 10000
MAX_VARIABLE_NAME_LENGTH = 100
MAX_LIST_LENGTH = 1000


def validate_variable_name(name: str) -> str:
    """
    Validate a variable name.

    Args:
        name: The variable name to validate

    Returns:
        The validated variable name

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(name, str):
        raise ValidationError(f"Variable name must be a string, got {type(name)}")

    if len(name) > MAX_VARIABLE_NAME_LENGTH:
        raise ValidationError(
            f"Variable name too long (max {MAX_VARIABLE_NAME_LENGTH} characters)"
        )

    if not name:
        raise ValidationError("Variable name cannot be empty")

    # Check for valid Python identifier
    if not name.isidentifier():
        raise ValidationError(
            f"Invalid variable name '{name}': must be a valid Python identifier"
        )

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            raise ValidationError(
                f"Variable name '{name}' contains potentially dangerous pattern: {pattern}"
            )

    return name


def validate_expression_string(expr: str) -> str:
    """
    Validate an expression string before parsing.

    Args:
        expr: The expression string to validate

    Returns:
        The validated expression string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(expr, str):
        raise ValidationError(f"Expression must be a string, got {type(expr)}")

    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise ValidationError(
            f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)"
        )

    if not expr.strip():
        raise ValidationError("Expression cannot be empty")

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, expr, re.IGNORECASE):
            raise ValidationError(
                f"Expression contains potentially dangerous pattern: {pattern}"
            )

    return expr


def safe_sympify(
    expr: Union[str, Any],
    locals_dict: Optional[dict] = None,
    strict: bool = True
) -> sp.Basic:
    """
    Safely parse an expression using SymPy's sympify with validation.

    Args:
        expr: Expression to parse (string or SymPy object)
        locals_dict: Dictionary of local variables for parsing
        strict: If True, perform strict validation

    Returns:
        Parsed SymPy expression

    Raises:
        ValidationError: If validation fails
        SympifyError: If SymPy parsing fails
    """
    # If already a SymPy object, return it
    if isinstance(expr, sp.Basic):
        return expr

    # If numeric, convert safely
    if isinstance(expr, (int, float, complex)):
        return sp.sympify(expr)

    # Validate string expressions
    if isinstance(expr, str):
        if strict:
            validate_expression_string(expr)

        try:
            # Use sympify with evaluate=False to prevent auto-simplification
            # This is safer as it doesn't execute arbitrary code
            result = sp.sympify(
                expr,
                locals=locals_dict,
                evaluate=False,
                rational=False,  # Don't automatically convert floats to rationals
            )
            return result
        except (SympifyError, TypeError, ValueError) as e:
            raise ValidationError(f"Failed to parse expression '{expr}': {str(e)}")

    raise ValidationError(f"Unsupported expression type: {type(expr)}")


def validate_list_input(data: Any, max_length: int = MAX_LIST_LENGTH) -> List:
    """
    Validate a list input.

    Args:
        data: The data to validate as a list
        max_length: Maximum allowed list length

    Returns:
        The validated list

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, list):
        raise ValidationError(f"Expected list, got {type(data)}")

    if len(data) > max_length:
        raise ValidationError(f"List too long (max {max_length} elements)")

    return data


def validate_dict_input(data: Any, required_keys: Optional[List[str]] = None) -> dict:
    """
    Validate a dictionary input.

    Args:
        data: The data to validate as a dict
        required_keys: List of required keys (optional)

    Returns:
        The validated dictionary

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError(f"Expected dict, got {type(data)}")

    if required_keys:
        missing_keys = set(required_keys) - set(data.keys())
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}")

    return data


def validate_numeric_input(value: Any, allow_symbolic: bool = False) -> Union[int, float, sp.Basic]:
    """
    Validate a numeric input.

    Args:
        value: The value to validate
        allow_symbolic: If True, allow SymPy symbolic expressions

    Returns:
        The validated numeric value

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(value, (int, float)):
        return value

    if allow_symbolic and isinstance(value, sp.Basic):
        return value

    if isinstance(value, str):
        try:
            # Try to parse as float first
            return float(value)
        except ValueError:
            if allow_symbolic:
                return safe_sympify(value)
            raise ValidationError(f"Invalid numeric value: '{value}'")

    raise ValidationError(f"Expected numeric value, got {type(value)}")


def sanitize_latex_input(latex_str: str) -> str:
    """
    Sanitize LaTeX input to prevent injection attacks.

    Args:
        latex_str: LaTeX string to sanitize

    Returns:
        Sanitized LaTeX string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(latex_str, str):
        raise ValidationError(f"LaTeX input must be a string, got {type(latex_str)}")

    if len(latex_str) > MAX_EXPRESSION_LENGTH:
        raise ValidationError(
            f"LaTeX string too long (max {MAX_EXPRESSION_LENGTH} characters)"
        )

    # Check for dangerous LaTeX commands that could execute code
    dangerous_latex = [
        r'\\write',
        r'\\input',
        r'\\include',
        r'\\def',
        r'\\let',
        r'\\expandafter',
        r'\\csname',
    ]

    for pattern in dangerous_latex:
        if pattern in latex_str:
            raise ValidationError(
                f"LaTeX string contains potentially dangerous command: {pattern}"
            )

    return latex_str


def validate_matrix_dimensions(rows: int, cols: int, max_size: int = 100) -> tuple:
    """
    Validate matrix dimensions.

    Args:
        rows: Number of rows
        cols: Number of columns
        max_size: Maximum dimension size

    Returns:
        Tuple of (rows, cols)

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(rows, int) or not isinstance(cols, int):
        raise ValidationError("Matrix dimensions must be integers")

    if rows <= 0 or cols <= 0:
        raise ValidationError("Matrix dimensions must be positive")

    if rows > max_size or cols > max_size:
        raise ValidationError(f"Matrix dimensions too large (max {max_size}x{max_size})")

    return rows, cols


def validate_key_format(key: str, prefix: str = "") -> str:
    """
    Validate a key format (e.g., expr_0, matrix_1).

    Args:
        key: The key to validate
        prefix: Expected prefix (optional)

    Returns:
        The validated key

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(key, str):
        raise ValidationError(f"Key must be a string, got {type(key)}")

    if prefix and not key.startswith(prefix):
        raise ValidationError(f"Key must start with '{prefix}'")

    # Validate format
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
        raise ValidationError(f"Invalid key format: '{key}'")

    return key
