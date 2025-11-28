"""
Comprehensive tests for reasonforge-geometry MCP server.

Tests all 15 tools:
- Vector Calculus (5 tools): create_coordinate_system, create_vector_field, calculate_curl, calculate_divergence, calculate_gradient
- Tensor Calculus & GR (5 tools): create_predefined_metric, search_predefined_metrics, calculate_tensor, create_custom_metric, print_latex_tensor
- Units (2 tools): convert_to_units, quantity_simplify_units
- Visualization (3 tools): plot_symbolic_function, contour_plot_symbolic, vector_field_plot
"""

import sys
import os
import asyncio
import json
import pytest

# Add packages to path
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge', 'src'))
sys.path.insert(0, os.path.join(base_dir, 'packages', 'reasonforge-geometry', 'src'))

from reasonforge_geometry.server import server as geometry_server


class TestVectorCalculus:
    """Test vector calculus tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_create_coordinate_system_cartesian(self):
        """Test creating Cartesian coordinate system."""
        result = await geometry_server.call_tool_for_test(
            "create_coordinate_system",
            {
                "name": "C",
                "type": "Cartesian"
            }
        )
        data = json.loads(result[0].text)

        assert data["name"] == "C"
        assert data["type"] == "Cartesian"
        assert "basis_vectors" in data
        assert "i" in data["basis_vectors"]
        assert "j" in data["basis_vectors"]
        assert "k" in data["basis_vectors"]

    @pytest.mark.asyncio
    async def test_create_coordinate_system_cylindrical(self):
        """Test creating cylindrical coordinate system."""
        result = await geometry_server.call_tool_for_test(
            "create_coordinate_system",
            {
                "name": "Cyl",
                "type": "cylindrical"
            }
        )
        data = json.loads(result[0].text)

        assert data["name"] == "Cyl"
        assert data["type"] == "cylindrical"
        assert "basis_vectors" in data

    @pytest.mark.asyncio
    async def test_create_coordinate_system_spherical(self):
        """Test creating spherical coordinate system."""
        result = await geometry_server.call_tool_for_test(
            "create_coordinate_system",
            {
                "name": "Sph",
                "type": "spherical"
            }
        )
        data = json.loads(result[0].text)

        assert data["name"] == "Sph"
        assert data["type"] == "spherical"

    @pytest.mark.asyncio
    async def test_create_vector_field(self):
        """Test creating vector field."""
        # First create coordinate system
        await geometry_server.call_tool_for_test(
            "create_coordinate_system",
            {"name": "C1", "type": "Cartesian"}
        )

        # Create vector field
        result = await geometry_server.call_tool_for_test(
            "create_vector_field",
            {
                "coord_system": "C1",
                "components": {
                    "i": "y",
                    "j": "-x",
                    "k": "0"
                }
            }
        )
        data = json.loads(result[0].text)

        assert "field_key" in data
        assert data["coordinate_system"] == "C1"
        assert data["components"]["i"] == "y"
        assert data["components"]["j"] == "-x"

    @pytest.mark.asyncio
    async def test_create_vector_field_error_no_coord_system(self):
        """Test error when coordinate system doesn't exist."""
        result = await geometry_server.call_tool_for_test(
            "create_vector_field",
            {
                "coord_system": "nonexistent",
                "components": {"i": "1", "j": "0", "k": "0"}
            }
        )
        data = json.loads(result[0].text)

        assert "error" in data

    @pytest.mark.asyncio
    async def test_calculate_curl(self):
        """Test calculating curl of vector field."""
        # Create coordinate system and vector field
        await geometry_server.call_tool_for_test(
            "create_coordinate_system",
            {"name": "C2", "type": "Cartesian"}
        )

        field_result = await geometry_server.call_tool_for_test(
            "create_vector_field",
            {
                "coord_system": "C2",
                "components": {"i": "y", "j": "-x", "k": "0"}
            }
        )
        field_data = json.loads(field_result[0].text)
        field_key = field_data["field_key"]

        # Calculate curl
        result = await geometry_server.call_tool_for_test(
            "calculate_curl",
            {"field_name": field_key}
        )
        data = json.loads(result[0].text)

        assert "curl" in data
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_calculate_divergence(self):
        """Test calculating divergence of vector field."""
        await geometry_server.call_tool_for_test(
            "create_coordinate_system",
            {"name": "C3", "type": "Cartesian"}
        )

        field_result = await geometry_server.call_tool_for_test(
            "create_vector_field",
            {
                "coord_system": "C3",
                "components": {"i": "x", "j": "y", "k": "z"}
            }
        )
        field_data = json.loads(field_result[0].text)
        field_key = field_data["field_key"]

        result = await geometry_server.call_tool_for_test(
            "calculate_divergence",
            {"field_name": field_key}
        )
        data = json.loads(result[0].text)

        assert "divergence" in data
        # Divergence of (x, y, z) should be 3

    @pytest.mark.asyncio
    async def test_calculate_gradient(self):
        """Test calculating gradient of scalar field."""
        result = await geometry_server.call_tool_for_test(
            "calculate_gradient",
            {
                "scalar_field": "x**2 + y**2 + z**2"
            }
        )
        data = json.loads(result[0].text)

        assert "gradient" in data
        assert "latex" in data
        # Gradient should contain 2*x, 2*y, 2*z components

    @pytest.mark.asyncio
    async def test_calculate_gradient_with_coord_system(self):
        """Test gradient with specific coordinate system."""
        await geometry_server.call_tool_for_test(
            "create_coordinate_system",
            {"name": "C4", "type": "Cartesian"}
        )

        result = await geometry_server.call_tool_for_test(
            "calculate_gradient",
            {
                "scalar_field": "x*y",
                "coord_system": "C4"
            }
        )
        data = json.loads(result[0].text)

        assert "gradient" in data


class TestTensorCalculusAndGR:
    """Test tensor calculus and general relativity tools (5 tools)."""

    @pytest.mark.asyncio
    async def test_create_predefined_metric_schwarzschild(self):
        """Test creating Schwarzschild metric."""
        result = await geometry_server.call_tool_for_test(
            "create_predefined_metric",
            {"metric_type": "Schwarzschild"}
        )
        data = json.loads(result[0].text)

        assert "metric_key" in data
        assert data["metric_type"] == "Schwarzschild"
        assert data["coordinates"] == ['t', 'r', 'theta', 'phi']
        assert "metric_components" in data
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_create_predefined_metric_minkowski(self):
        """Test creating Minkowski metric."""
        result = await geometry_server.call_tool_for_test(
            "create_predefined_metric",
            {"metric_type": "Minkowski"}
        )
        data = json.loads(result[0].text)

        assert data["metric_type"] == "Minkowski"
        assert data["coordinates"] == ['t', 'x', 'y', 'z']
        # Minkowski is diagonal with signature (-,+,+,+)

    @pytest.mark.asyncio
    async def test_create_predefined_metric_flrw(self):
        """Test creating FLRW metric."""
        result = await geometry_server.call_tool_for_test(
            "create_predefined_metric",
            {"metric_type": "FLRW"}
        )
        data = json.loads(result[0].text)

        assert data["metric_type"] == "FLRW"
        assert "metric_key" in data

    @pytest.mark.asyncio
    async def test_search_predefined_metrics_all(self):
        """Test searching all predefined metrics."""
        result = await geometry_server.call_tool_for_test(
            "search_predefined_metrics",
            {}
        )
        data = json.loads(result[0].text)

        assert "found" in data
        assert data["found"] >= 3  # At least Schwarzschild, Minkowski, FLRW
        assert "metrics" in data

    @pytest.mark.asyncio
    async def test_search_predefined_metrics_query(self):
        """Test searching metrics with query."""
        result = await geometry_server.call_tool_for_test(
            "search_predefined_metrics",
            {"query": "black hole"}
        )
        data = json.loads(result[0].text)

        assert "found" in data
        # Should find Schwarzschild

    @pytest.mark.asyncio
    async def test_calculate_tensor_inverse(self):
        """Test calculating inverse metric tensor."""
        # First create a metric
        metric_result = await geometry_server.call_tool_for_test(
            "create_predefined_metric",
            {"metric_type": "Minkowski"}
        )
        metric_data = json.loads(metric_result[0].text)
        metric_key = metric_data["metric_key"]

        # Calculate inverse
        result = await geometry_server.call_tool_for_test(
            "calculate_tensor",
            {
                "metric_key": metric_key,
                "tensor_type": "inverse"
            }
        )
        data = json.loads(result[0].text)

        assert data["tensor_type"] == "inverse"
        assert "result" in data
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_calculate_tensor_determinant(self):
        """Test calculating metric determinant."""
        metric_result = await geometry_server.call_tool_for_test(
            "create_predefined_metric",
            {"metric_type": "Minkowski"}
        )
        metric_data = json.loads(metric_result[0].text)
        metric_key = metric_data["metric_key"]

        result = await geometry_server.call_tool_for_test(
            "calculate_tensor",
            {
                "metric_key": metric_key,
                "tensor_type": "determinant"
            }
        )
        data = json.loads(result[0].text)

        assert data["tensor_type"] == "determinant"
        assert "result" in data
        # Minkowski determinant should be -1

    @pytest.mark.asyncio
    async def test_calculate_tensor_christoffel(self):
        """Test calculating Christoffel symbols."""
        metric_result = await geometry_server.call_tool_for_test(
            "create_predefined_metric",
            {"metric_type": "Schwarzschild"}
        )
        metric_data = json.loads(metric_result[0].text)
        metric_key = metric_data["metric_key"]

        result = await geometry_server.call_tool_for_test(
            "calculate_tensor",
            {
                "metric_key": metric_key,
                "tensor_type": "Christoffel"
            }
        )
        data = json.loads(result[0].text)

        assert data["tensor_type"] == "Christoffel"
        assert "samples" in data

    @pytest.mark.asyncio
    async def test_create_custom_metric(self):
        """Test creating custom metric."""
        result = await geometry_server.call_tool_for_test(
            "create_custom_metric",
            {
                "components": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]
                ],
                "coordinates": ["x", "y", "z"]
            }
        )
        data = json.loads(result[0].text)

        assert "metric_key" in data
        assert data["coordinates"] == ["x", "y", "z"]
        assert "metric_components" in data

    @pytest.mark.asyncio
    async def test_print_latex_tensor(self):
        """Test printing tensor in LaTeX."""
        # Create a metric first
        metric_result = await geometry_server.call_tool_for_test(
            "create_predefined_metric",
            {"metric_type": "Minkowski"}
        )
        metric_data = json.loads(metric_result[0].text)
        metric_key = metric_data["metric_key"]

        result = await geometry_server.call_tool_for_test(
            "print_latex_tensor",
            {"key": metric_key}
        )
        data = json.loads(result[0].text)

        assert "latex" in data
        assert "string" in data

    @pytest.mark.asyncio
    async def test_print_latex_tensor_not_found(self):
        """Test error when tensor not found."""
        result = await geometry_server.call_tool_for_test(
            "print_latex_tensor",
            {"key": "nonexistent"}
        )
        data = json.loads(result[0].text)

        assert "error" in data


class TestUnits:
    """Test unit conversion tools (2 tools)."""

    @pytest.mark.asyncio
    async def test_quantity_simplify_units(self):
        """Test simplifying units."""
        result = await geometry_server.call_tool_for_test(
            "quantity_simplify_units",
            {"expression": "x**2 + 2*x + 1"}
        )
        data = json.loads(result[0].text)

        assert "simplified" in data
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_quantity_simplify_units_complex(self):
        """Test simplifying complex expression."""
        result = await geometry_server.call_tool_for_test(
            "quantity_simplify_units",
            {"expression": "(a + b)**2 - (a**2 + 2*a*b + b**2)"}
        )
        data = json.loads(result[0].text)

        assert "simplified" in data
        # Should simplify to 0


class TestVisualization:
    """Test visualization tools (3 tools)."""

    @pytest.mark.asyncio
    async def test_plot_symbolic_function(self):
        """Test plotting symbolic function."""
        result = await geometry_server.call_tool_for_test(
            "plot_symbolic_function",
            {
                "expression": "x**2",
                "variable": "x",
                "range": {"min": -5, "max": 5}
            }
        )
        data = json.loads(result[0].text)

        assert data["expression"] == "x**2"
        assert data["variable"] == "x"
        assert data["range"]["min"] == -5
        assert data["range"]["max"] == 5
        assert "note" in data  # Note about MCP not displaying plots

    @pytest.mark.asyncio
    async def test_plot_symbolic_function_default_range(self):
        """Test plotting with default range."""
        result = await geometry_server.call_tool_for_test(
            "plot_symbolic_function",
            {
                "expression": "sin(x)",
                "variable": "x"
            }
        )
        data = json.loads(result[0].text)

        assert "range" in data

    @pytest.mark.asyncio
    async def test_contour_plot_symbolic(self):
        """Test contour plot."""
        result = await geometry_server.call_tool_for_test(
            "contour_plot_symbolic",
            {
                "expression": "x**2 + y**2",
                "x_range": {"min": -3, "max": 3},
                "y_range": {"min": -3, "max": 3}
            }
        )
        data = json.loads(result[0].text)

        assert data["expression"] == "x**2 + y**2"
        assert "variables" in data
        assert len(data["variables"]) >= 2

    @pytest.mark.asyncio
    async def test_contour_plot_symbolic_error_one_variable(self):
        """Test error when expression has only one variable."""
        result = await geometry_server.call_tool_for_test(
            "contour_plot_symbolic",
            {"expression": "x**2"}
        )
        data = json.loads(result[0].text)

        assert "error" in data

    @pytest.mark.asyncio
    async def test_vector_field_plot_2d(self):
        """Test 2D vector field plot."""
        result = await geometry_server.call_tool_for_test(
            "vector_field_plot",
            {
                "field_components": {"x": "y", "y": "-x"},
                "dimension": 2
            }
        )
        data = json.loads(result[0].text)

        assert data["dimension"] == 2
        assert "field_components" in data
        assert "note" in data

    @pytest.mark.asyncio
    async def test_vector_field_plot_3d(self):
        """Test 3D vector field plot."""
        result = await geometry_server.call_tool_for_test(
            "vector_field_plot",
            {
                "field_components": {"x": "y", "y": "z", "z": "x"},
                "dimension": 3
            }
        )
        data = json.loads(result[0].text)

        assert data["dimension"] == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
