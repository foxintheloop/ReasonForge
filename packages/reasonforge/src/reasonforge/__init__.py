"""
ReasonForge Core Library

A standalone Python library for symbolic mathematical computation.
This is the core computation engine used by all ReasonForge MCP servers.

Usage:
    from reasonforge import SymbolicAI

    ai = SymbolicAI()
    x, y = ai.define_variables(['x', 'y'])
    result = ai.solve_equation_system([x**2 + y**2 - 25, x + y - 7])
"""

from .symbolic_engine import SymbolicAI, SymbolicApplications
from .validation import (
    ValidationError,
    safe_sympify,
    validate_variable_name,
    validate_expression_string,
    validate_list_input,
    validate_dict_input,
    validate_numeric_input,
    sanitize_latex_input,
    validate_matrix_dimensions,
    validate_key_format,
)
from .mcp_base import (
    BaseReasonForgeServer,
    ToolHandler,
    create_input_schema,
)

__version__ = "0.1.0"
__all__ = [
    "SymbolicAI",
    "SymbolicApplications",
    "ValidationError",
    "safe_sympify",
    "validate_variable_name",
    "validate_expression_string",
    "validate_list_input",
    "validate_dict_input",
    "validate_numeric_input",
    "sanitize_latex_input",
    "validate_matrix_dimensions",
    "validate_key_format",
    "BaseReasonForgeServer",
    "ToolHandler",
    "create_input_schema",
]
