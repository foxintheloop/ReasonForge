"""
Tests for Vector Calculus Tools (5 tools).

Tools tested:
1. create_coordinate_system - Define 3D coordinate systems
2. create_vector_field - Define vector fields
3. calculate_curl - Compute curl of vector field
4. calculate_divergence - Compute divergence of vector field
5. calculate_gradient - Compute gradient of scalar field
"""

import pytest
import json
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestCreateCoordinateSystem:
    """Test the 'create_coordinate_system' tool."""

    @pytest.mark.asyncio
    async def test_create_cartesian_system(self, ai):
        """Test creating a Cartesian coordinate system."""
        result = await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'C', 'coord_type': 'cartesian'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['name'] == 'C'
        assert data['type'] == 'cartesian'
        assert 'C' in ai.coordinate_systems

    @pytest.mark.asyncio
    async def test_create_cylindrical_system(self, ai):
        """Test creating a cylindrical coordinate system."""
        result = await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'Cyl', 'coord_type': 'cylindrical'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['type'] == 'cylindrical'

    @pytest.mark.asyncio
    async def test_create_spherical_system(self, ai):
        """Test creating a spherical coordinate system."""
        result = await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'Sph', 'coord_type': 'spherical'},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['type'] == 'spherical'


class TestCreateVectorField:
    """Test the 'create_vector_field' tool."""

    @pytest.mark.asyncio
    async def test_create_vector_field_basic(self, ai):
        """Test creating a basic vector field."""
        # First create coordinate system
        await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'C', 'coord_type': 'cartesian'},
            ai
        )

        result = await handle_advanced_tool(
            'create_vector_field',
            {
                'coord_system_name': 'C',
                'components': {'i': '1', 'j': '0', 'k': '0'}
            },
            ai
        )
        data = json.loads(result[0].text)

        # Debug: Print actual response
        import sys
        print(f"DEBUG: Vector field response: {data}", file=sys.stderr)

        assert 'key' in data or 'error' not in data, f"Unexpected response: {data}"
        if 'status' in data:
            assert data['status'] == 'success'
        if 'key' in data:
            assert data['key'] in ai.vector_fields

    @pytest.mark.asyncio
    async def test_create_vector_field_with_variables(self, ai):
        """Test creating a vector field with variable components."""
        await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'C', 'coord_type': 'cartesian'},
            ai
        )

        result = await handle_advanced_tool(
            'create_vector_field',
            {
                'coord_system_name': 'C',
                'components': {'i': 'y', 'j': '-x', 'k': '0'},
                'key': 'rotation_field'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['key'] == 'rotation_field'


class TestCalculateCurl:
    """Test the 'calculate_curl' tool."""

    @pytest.mark.asyncio
    async def test_calculate_curl_rotation_field(self, ai):
        """Test calculating curl of a rotation field."""
        # Setup
        await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'C'},
            ai
        )
        await handle_advanced_tool(
            'create_vector_field',
            {
                'coord_system_name': 'C',
                'components': {'i': 'y', 'j': '-x', 'k': '0'},
                'key': 'field1'
            },
            ai
        )

        # Test
        result = await handle_advanced_tool(
            'calculate_curl',
            {'vector_field_key': 'field1'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'curl' in data
        assert 'result_key' in data


class TestCalculateDivergence:
    """Test the 'calculate_divergence' tool."""

    @pytest.mark.asyncio
    async def test_calculate_divergence_basic(self, ai):
        """Test calculating divergence of a vector field."""
        await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'C'},
            ai
        )
        await handle_advanced_tool(
            'create_vector_field',
            {
                'coord_system_name': 'C',
                'components': {'i': 'x', 'j': 'y', 'k': 'z'},
                'key': 'field1'
            },
            ai
        )

        result = await handle_advanced_tool(
            'calculate_divergence',
            {'vector_field_key': 'field1'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'divergence' in data
        # Divergence of (x, y, z) is 3


class TestCalculateGradient:
    """Test the 'calculate_gradient' tool."""

    @pytest.mark.asyncio
    async def test_calculate_gradient_basic(self, ai):
        """Test calculating gradient of a scalar field."""
        await handle_advanced_tool('intro_many', {'names': ['x', 'y', 'z']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 + y**2 + z**2', 'key': 'scalar1'},
            ai
        )
        await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'C'},
            ai
        )

        result = await handle_advanced_tool(
            'calculate_gradient',
            {
                'expression_key': 'scalar1',
                'coord_system_name': 'C'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert 'gradient' in data
        assert 'result_key' in data
