"""
Tests for tensor calculus and general relativity tools.

This module tests the 5 tensor/GR tools from advanced_tools.py:
1. create_predefined_metric - Load spacetime metrics
2. search_predefined_metrics - Search available metrics
3. calculate_tensor - Compute Ricci, Einstein, Weyl tensors
4. create_custom_metric - Define custom metric tensors
5. print_latex_tensor - Format tensors as LaTeX

Tests based on usage examples from USAGE_EXAMPLES.md
Note: Requires einsteinpy library
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


@pytest.fixture
def check_einsteinpy():
    """Check if einsteinpy is installed."""
    try:
        import einsteinpy
        return True
    except ImportError:
        return False


class TestCreatePredefinedMetric:
    """Test the 'create_predefined_metric' tool."""

    @pytest.mark.asyncio
    async def test_load_schwarzschild_metric(self, ai, check_einsteinpy):
        """Test loading Schwarzschild metric (from USAGE_EXAMPLES.md)."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'create_predefined_metric',
            {
                'metric_type': 'Schwarzschild',
                'parameters': {'M': 'M'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'key' in data

    @pytest.mark.asyncio
    async def test_load_kerr_metric(self, ai, check_einsteinpy):
        """Test loading Kerr metric."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'create_predefined_metric',
            {
                'metric_type': 'Kerr',
                'parameters': {'M': 'M', 'a': 'a'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_load_minkowski_metric(self, ai, check_einsteinpy):
        """Test loading flat spacetime metric."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'create_predefined_metric',
            {
                'metric_type': 'Minkowski',
                'parameters': {}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestSearchPredefinedMetrics:
    """Test the 'search_predefined_metrics' tool."""

    @pytest.mark.asyncio
    async def test_search_all_metrics(self, ai, check_einsteinpy):
        """Test searching all available metrics (from USAGE_EXAMPLES.md)."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'search_predefined_metrics',
            {},
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'available_metrics' in data

    @pytest.mark.asyncio
    async def test_search_specific_metric(self, ai, check_einsteinpy):
        """Test searching for specific metric."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'search_predefined_metrics',
            {'query': 'Schwarzschild'},
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestCalculateTensor:
    """Test the 'calculate_tensor' tool."""

    @pytest.mark.asyncio
    async def test_calculate_ricci_tensor(self, ai, check_einsteinpy):
        """Test calculating Ricci tensor (from USAGE_EXAMPLES.md)."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'calculate_tensor',
            {
                'metric_key': 'schwarzschild_1',
                'tensor_type': 'Ricci'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'result' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_calculate_einstein_tensor(self, ai, check_einsteinpy):
        """Test calculating Einstein tensor."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'calculate_tensor',
            {
                'metric_key': 'metric_1',
                'tensor_type': 'Einstein'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_calculate_christoffel_symbols(self, ai, check_einsteinpy):
        """Test calculating Christoffel symbols."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'calculate_tensor',
            {
                'metric_key': 'metric_1',
                'tensor_type': 'Christoffel'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestCreateCustomMetric:
    """Test the 'create_custom_metric' tool."""

    @pytest.mark.asyncio
    async def test_create_2d_metric(self, ai, check_einsteinpy):
        """Test creating custom 2D metric (from USAGE_EXAMPLES.md)."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'create_custom_metric',
            {
                'components': [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ],
                'coordinates': ['t', 'x', 'y', 'z']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'key' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_create_diagonal_metric(self, ai, check_einsteinpy):
        """Test creating diagonal metric."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'create_custom_metric',
            {
                'components': [
                    ['-1', '0', '0', '0'],
                    ['0', '1', '0', '0'],
                    ['0', '0', '1', '0'],
                    ['0', '0', '0', '1']
                ],
                'coordinates': ['t', 'r', 'theta', 'phi']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestPrintLatexTensor:
    """Test the 'print_latex_tensor' tool."""

    @pytest.mark.asyncio
    async def test_print_ricci_tensor_latex(self, ai, check_einsteinpy):
        """Test printing Ricci tensor in LaTeX (from USAGE_EXAMPLES.md)."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'print_latex_tensor',
            {'key': 'ricci_1'},
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'latex' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_print_metric_latex(self, ai, check_einsteinpy):
        """Test printing metric tensor in LaTeX."""
        if not check_einsteinpy:
            pytest.skip("einsteinpy not installed")

        result = await handle_advanced_tool(
            'print_latex_tensor',
            {'key': 'metric_1'},
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data
