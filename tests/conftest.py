"""
Project-level pytest configuration for ReasonForge MCP Server.

NOTE: This file contains legacy fixtures from earlier development phases.
The current test suite (packages/*/tests/) uses package-specific conftest.py files.

All active tests are located in:
- packages/reasonforge-expressions/tests/
- packages/reasonforge-algebra/tests/
- packages/reasonforge-analysis/tests/
- packages/reasonforge-geometry/tests/
- packages/reasonforge-statistics/tests/
- packages/reasonforge-physics/tests/
- packages/reasonforge-logic/tests/

Run tests with: pytest packages/

Legacy fixtures below are kept for reference only.
"""

import pytest
import json


# Legacy fixtures (not used by current test suite)
# These were used by tests now archived in tests/legacy/

@pytest.fixture
def sample_expressions():
    """Provide common mathematical expressions for testing."""
    return {
        'polynomial': 'x**2 + 2*x + 1',
        'factored_polynomial': '(x + 1)**2',
        'trigonometric': 'sin(x)**2 + cos(x)**2',
        'exponential': 'exp(x) * exp(-x)',
        'rational': '(x**2 - 1)/(x - 1)',
        'complex': 'x**3 - 3*x**2 + 3*x - 1'
    }


@pytest.fixture
def sample_matrices():
    """Provide common matrices for testing."""
    return {
        '2x2_identity': [[1, 0], [0, 1]],
        '2x2_simple': [[1, 2], [3, 4]],
        '3x3_identity': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        '3x3_singular': [[1, 2, 3], [2, 4, 6], [7, 8, 9]],
        '2x2_symbolic': [['a', 'b'], ['c', 'd']]
    }
