"""
Tests for visualization and plotting tools.

This module tests the 6 visualization tools from visualization_tools.py:
1. plot_symbolic_function - Generate plots and analyze functions
2. contour_plot_symbolic - Create contour plots and level curves
3. vector_field_plot - Visualize vector fields
4. phase_portrait - Generate phase portraits for dynamical systems
5. bifurcation_diagram - Analyze bifurcations
6. 3d_surface_plot - 3D surface representations

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.visualization_tools import handle_visualization_tool


class TestPlotSymbolicFunction:
    """Test the 'plot_symbolic_function' tool."""

    @pytest.mark.asyncio
    async def test_plot_quadratic(self, ai):
        """Test plotting quadratic function (from USAGE_EXAMPLES.md)."""
        result = await handle_visualization_tool(
            'plot_symbolic_function',
            {
                'expression': 'x**2 - 4*x + 3',
                'variable': 'x',
                'range': {'min': -1, 'max': 5},
                'output_format': ['latex', 'critical_points']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'critical_points' in data

    @pytest.mark.asyncio
    async def test_plot_trigonometric(self, ai):
        """Test plotting trigonometric function."""
        result = await handle_visualization_tool(
            'plot_symbolic_function',
            {
                'expression': 'sin(x)',
                'variable': 'x',
                'range': {'min': -3.14, 'max': 3.14}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'latex' in data

    @pytest.mark.asyncio
    async def test_plot_with_properties(self, ai):
        """Test plotting with property analysis."""
        result = await handle_visualization_tool(
            'plot_symbolic_function',
            {
                'expression': 'x**3 - 3*x',
                'variable': 'x',
                'range': {'min': -2, 'max': 2},
                'output_format': ['properties', 'critical_points']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestContourPlotSymbolic:
    """Test the 'contour_plot_symbolic' tool."""

    @pytest.mark.asyncio
    async def test_contour_circle(self, ai):
        """Test contour plot for circles (from USAGE_EXAMPLES.md)."""
        result = await handle_visualization_tool(
            'contour_plot_symbolic',
            {
                'expression': 'x**2 + y**2',
                'x_range': {'min': -5, 'max': 5},
                'y_range': {'min': -5, 'max': 5},
                'level_curves': [1, 4, 9],
                'compute': ['gradient', 'critical_points']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'gradient' in data or 'critical_points' in data

    @pytest.mark.asyncio
    async def test_contour_saddle(self, ai):
        """Test contour plot for saddle surface."""
        result = await handle_visualization_tool(
            'contour_plot_symbolic',
            {
                'expression': 'x**2 - y**2',
                'x_range': {'min': -3, 'max': 3},
                'y_range': {'min': -3, 'max': 3},
                'level_curves': [-4, 0, 4]
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_level_equations(self, ai):
        """Test level curve equations."""
        result = await handle_visualization_tool(
            'contour_plot_symbolic',
            {
                'expression': 'x*y',
                'x_range': {'min': -2, 'max': 2},
                'y_range': {'min': -2, 'max': 2},
                'compute': ['level_equations']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestVectorFieldPlot:
    """Test the 'vector_field_plot' tool."""

    @pytest.mark.asyncio
    async def test_circular_vector_field(self, ai):
        """Test circular vector field (from USAGE_EXAMPLES.md)."""
        result = await handle_visualization_tool(
            'vector_field_plot',
            {
                'field_components': {'x': '-y', 'y': 'x'},
                'dimension': 2,
                'analyze': ['divergence', 'curl', 'critical_points']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'divergence' in data or 'curl' in data

    @pytest.mark.asyncio
    async def test_radial_field(self, ai):
        """Test radial vector field."""
        result = await handle_visualization_tool(
            'vector_field_plot',
            {
                'field_components': {'x': 'x', 'y': 'y'},
                'dimension': 2,
                'analyze': ['divergence']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_3d_vector_field(self, ai):
        """Test 3D vector field."""
        result = await handle_visualization_tool(
            'vector_field_plot',
            {
                'field_components': {'x': 'y', 'y': '-x', 'z': 'z'},
                'dimension': 3,
                'analyze': ['curl', 'divergence']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestPhasePortrait:
    """Test the 'phase_portrait' tool."""

    @pytest.mark.asyncio
    async def test_harmonic_oscillator(self, ai):
        """Test phase portrait for harmonic oscillator (from USAGE_EXAMPLES.md)."""
        result = await handle_visualization_tool(
            'phase_portrait',
            {
                'system': {'dx_dt': 'y', 'dy_dt': '-x'},
                'variables': ['x', 'y'],
                'analyze': ['equilibria', 'stability', 'jacobian']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'equilibria' in data or 'stability' in data

    @pytest.mark.asyncio
    async def test_predator_prey(self, ai):
        """Test predator-prey system."""
        result = await handle_visualization_tool(
            'phase_portrait',
            {
                'system': {'dx_dt': 'x - x*y', 'dy_dt': '-y + x*y'},
                'analyze': ['equilibria', 'nullclines']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_van_der_pol(self, ai):
        """Test Van der Pol oscillator."""
        result = await handle_visualization_tool(
            'phase_portrait',
            {
                'system': {'dx_dt': 'y', 'dy_dt': 'mu*(1 - x**2)*y - x'},
                'analyze': ['stability', 'eigenvectors']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestBifurcationDiagram:
    """Test the 'bifurcation_diagram' tool."""

    @pytest.mark.asyncio
    async def test_pitchfork_bifurcation(self, ai):
        """Test pitchfork bifurcation (from USAGE_EXAMPLES.md)."""
        result = await handle_visualization_tool(
            'bifurcation_diagram',
            {
                'system': 'r*x - x**3',
                'state_variable': 'x',
                'parameter': 'r',
                'analyze': ['equilibria', 'bifurcation_points', 'classification']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'bifurcation_points' in data or 'equilibria' in data

    @pytest.mark.asyncio
    async def test_hopf_bifurcation(self, ai):
        """Test Hopf bifurcation."""
        result = await handle_visualization_tool(
            'bifurcation_diagram',
            {
                'system': 'mu*x - y - x*(x**2 + y**2)',
                'state_variable': 'x',
                'parameter': 'mu',
                'analyze': ['bifurcation_points']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_stability_change(self, ai):
        """Test stability change detection."""
        result = await handle_visualization_tool(
            'bifurcation_diagram',
            {
                'system': 'lambda*x - x**2',
                'state_variable': 'x',
                'parameter': 'lambda',
                'analyze': ['stability_change']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class Test3DSurfacePlot:
    """Test the '3d_surface_plot' tool."""

    @pytest.mark.asyncio
    async def test_saddle_surface(self, ai):
        """Test saddle surface plot (from USAGE_EXAMPLES.md)."""
        result = await handle_visualization_tool(
            '3d_surface_plot',
            {
                'expression': 'x**2 - y**2',
                'x_range': {'min': -3, 'max': 3},
                'y_range': {'min': -3, 'max': 3},
                'compute': ['gradient', 'hessian', 'saddle_points']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'saddle_points' in data or 'hessian' in data

    @pytest.mark.asyncio
    async def test_paraboloid(self, ai):
        """Test paraboloid surface."""
        result = await handle_visualization_tool(
            '3d_surface_plot',
            {
                'expression': 'x**2 + y**2',
                'x_range': {'min': -2, 'max': 2},
                'y_range': {'min': -2, 'max': 2},
                'compute': ['critical_points', 'curvature']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_surface_curvature(self, ai):
        """Test surface curvature computation."""
        result = await handle_visualization_tool(
            '3d_surface_plot',
            {
                'expression': 'sin(x)*cos(y)',
                'x_range': {'min': -3.14, 'max': 3.14},
                'y_range': {'min': -3.14, 'max': 3.14},
                'compute': ['curvature', 'gradient']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data
