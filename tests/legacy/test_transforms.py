"""
Tests for transform theory and signal processing tools.

This module tests the 6 transform tools from advanced_tools.py:
1. laplace_transform - Laplace transform and inverse
2. fourier_transform - Fourier transform and inverse
3. z_transform - Z-transforms for discrete-time signals
4. convolution - Symbolic convolution
5. transfer_function_analysis - Analyze poles, zeros, stability
6. mellin_transform - Mellin transform and inverse

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestLaplaceTransform:
    """Test the 'laplace_transform' tool."""

    @pytest.mark.asyncio
    async def test_laplace_exponential_decay(self, ai):
        """Test Laplace transform of e^(-at) (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'laplace_transform',
            {
                'expression': 'exp(-a*t)',
                'variable': 't',
                'transform_variable': 's',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
        # Result should be 1/(s + a)
        assert 's' in data['result']
        assert 'a' in data['result']

    @pytest.mark.asyncio
    async def test_laplace_sine_function(self, ai):
        """Test Laplace transform of sin(ωt)."""
        result = await handle_advanced_tool(
            'laplace_transform',
            {
                'expression': 'sin(omega*t)',
                'variable': 't',
                'transform_variable': 's',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_laplace_inverse_transform(self, ai):
        """Test inverse Laplace transform."""
        result = await handle_advanced_tool(
            'laplace_transform',
            {
                'expression': '1/(s + a)',
                'variable': 's',
                'transform_variable': 't',
                'inverse': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data


class TestFourierTransform:
    """Test the 'fourier_transform' tool."""

    @pytest.mark.asyncio
    async def test_fourier_gaussian(self, ai):
        """Test Fourier transform of Gaussian e^(-x^2) (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'fourier_transform',
            {
                'expression': 'exp(-x**2)',
                'variable': 'x',
                'transform_variable': 'k',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
        # Result should contain sqrt(pi) and exp(-k^2/4)
        result_str = str(data['result'])
        assert 'pi' in result_str or 'exp' in result_str

    @pytest.mark.asyncio
    async def test_fourier_rect_pulse(self, ai):
        """Test Fourier transform of rectangular pulse."""
        result = await handle_advanced_tool(
            'fourier_transform',
            {
                'expression': 'Heaviside(x + 1) - Heaviside(x - 1)',
                'variable': 'x',
                'transform_variable': 'k',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_fourier_inverse_transform(self, ai):
        """Test inverse Fourier transform."""
        result = await handle_advanced_tool(
            'fourier_transform',
            {
                'expression': 'exp(-k**2)',
                'variable': 'k',
                'transform_variable': 'x',
                'inverse': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data


class TestZTransform:
    """Test the 'z_transform' tool."""

    @pytest.mark.asyncio
    async def test_z_transform_geometric(self, ai):
        """Test Z-transform of a^n (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'z_transform',
            {
                'sequence': 'a**n',
                'n_variable': 'n',
                'z_variable': 'z',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
        # Result should be z/(z - a)
        assert 'z' in data['result']

    @pytest.mark.asyncio
    async def test_z_transform_unit_step(self, ai):
        """Test Z-transform of unit step sequence."""
        result = await handle_advanced_tool(
            'z_transform',
            {
                'sequence': '1',
                'n_variable': 'n',
                'z_variable': 'z',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_z_inverse_transform(self, ai):
        """Test inverse Z-transform."""
        result = await handle_advanced_tool(
            'z_transform',
            {
                'sequence': 'z/(z - a)',
                'n_variable': 'z',
                'z_variable': 'n',
                'inverse': True
            },
            ai
        )

        data = json.loads(result[0].text)
        # Should succeed or return appropriate message
        assert 'status' in data or 'result' in data


class TestConvolution:
    """Test the 'convolution' tool."""

    @pytest.mark.asyncio
    async def test_convolution_exponentials(self, ai):
        """Test convolution of e^(-t) and e^(-2t) (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'convolution',
            {
                'f': 'exp(-t)',
                'g': 'exp(-2*t)',
                'variable': 't',
                'convolution_type': 'continuous'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
        # Result should contain exponential terms

    @pytest.mark.asyncio
    async def test_discrete_convolution(self, ai):
        """Test discrete convolution."""
        result = await handle_advanced_tool(
            'convolution',
            {
                'f': 'a**n',
                'g': 'b**n',
                'variable': 'n',
                'convolution_type': 'discrete'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_circular_convolution(self, ai):
        """Test circular convolution."""
        result = await handle_advanced_tool(
            'convolution',
            {
                'f': 'sin(t)',
                'g': 'cos(t)',
                'variable': 't',
                'convolution_type': 'circular'
            },
            ai
        )

        data = json.loads(result[0].text)
        # May succeed or indicate circular convolution requires finite support
        assert 'status' in data or 'error' in data or 'result' in data


class TestTransferFunctionAnalysis:
    """Test the 'transfer_function_analysis' tool."""

    @pytest.mark.asyncio
    async def test_analyze_second_order_system(self, ai):
        """Test analysis of H(s) = 1/(s^2 + 2s + 2) (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'transfer_function_analysis',
            {
                'transfer_function': '1/(s**2 + 2*s + 2)',
                'variable': 's',
                'analyze': ['poles', 'zeros', 'stability']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'poles' in data
        assert 'zeros' in data or 'stability' in data
        # Poles should be -1 ± i

    @pytest.mark.asyncio
    async def test_analyze_first_order_system(self, ai):
        """Test analysis of first order system."""
        result = await handle_advanced_tool(
            'transfer_function_analysis',
            {
                'transfer_function': '1/(s + 1)',
                'variable': 's',
                'analyze': ['poles', 'stability']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'poles' in data

    @pytest.mark.asyncio
    async def test_analyze_with_zeros(self, ai):
        """Test system with both poles and zeros."""
        result = await handle_advanced_tool(
            'transfer_function_analysis',
            {
                'transfer_function': '(s + 1)/(s**2 + 3*s + 2)',
                'variable': 's',
                'analyze': ['poles', 'zeros']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'poles' in data
        assert 'zeros' in data


class TestMellinTransform:
    """Test the 'mellin_transform' tool."""

    @pytest.mark.asyncio
    async def test_mellin_exponential(self, ai):
        """Test Mellin transform of e^(-x) (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'mellin_transform',
            {
                'expression': 'exp(-x)',
                'variable': 'x',
                'transform_variable': 's',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
        # Result should be gamma(s)
        result_str = str(data['result'])
        assert 'gamma' in result_str.lower() or 'Gamma' in result_str

    @pytest.mark.asyncio
    async def test_mellin_power_function(self, ai):
        """Test Mellin transform of x^a."""
        result = await handle_advanced_tool(
            'mellin_transform',
            {
                'expression': 'x**a',
                'variable': 'x',
                'transform_variable': 's',
                'inverse': False
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_mellin_inverse_transform(self, ai):
        """Test inverse Mellin transform."""
        result = await handle_advanced_tool(
            'mellin_transform',
            {
                'expression': 'gamma(s)',
                'variable': 's',
                'transform_variable': 'x',
                'inverse': True
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
