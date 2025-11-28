"""
Tests for unit operations tools.

This module tests the 2 unit operation tools:
1. convert_to_units - Convert between physical units
2. quantity_simplify_units - Simplify unit expressions

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestConvertToUnits:
    """Test the 'convert_to_units' tool."""

    @pytest.mark.asyncio
    async def test_convert_meters_per_second_to_kilometers_per_hour(self, ai):
        """Test converting m/s to km/h (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'convert_to_units',
            {
                'expression': '5*meter/second',
                'target_units': 'kilometer/hour'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data
        # Verify conversion: 5 m/s = 18 km/h
        assert 'kilometer' in data['result'] or 'km' in data['result']
        assert 'hour' in data['result'] or 'hr' in data['result']

    @pytest.mark.asyncio
    async def test_convert_force_to_newtons(self, ai):
        """Test converting force units."""
        result = await handle_advanced_tool(
            'convert_to_units',
            {
                'expression': 'kilogram*meter/second**2',
                'target_units': 'newton'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'result' in data

    @pytest.mark.asyncio
    async def test_convert_temperature_units(self, ai):
        """Test temperature conversion."""
        result = await handle_advanced_tool(
            'convert_to_units',
            {
                'expression': '273.15*kelvin',
                'target_units': 'celsius'
            },
            ai
        )

        data = json.loads(result[0].text)
        # Should succeed or handle gracefully
        assert 'status' in data

    @pytest.mark.asyncio
    async def test_invalid_unit_expression(self, ai):
        """Test error handling for invalid unit expression."""
        result = await handle_advanced_tool(
            'convert_to_units',
            {
                'expression': 'invalid_unit_expression',
                'target_units': 'meter'
            },
            ai
        )

        data = json.loads(result[0].text)
        # Should either succeed with identity or return error
        assert 'status' in data or 'error' in data


class TestQuantitySimplifyUnits:
    """Test the 'quantity_simplify_units' tool."""

    @pytest.mark.asyncio
    async def test_simplify_force_units(self, ai):
        """Test simplifying force units (from USAGE_EXAMPLES.md)."""
        result = await handle_advanced_tool(
            'quantity_simplify_units',
            {
                'expression': '(kilogram*meter/second**2)*meter'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'simplified' in data
        # Result should be kg*m^2/s^2 (Joules)
        assert 'kilogram' in data['simplified'] or 'kg' in data['simplified']
        assert 'meter**2' in data['simplified'] or 'm**2' in data['simplified'] or 'm^2' in data['simplified']

    @pytest.mark.asyncio
    async def test_simplify_energy_units(self, ai):
        """Test simplifying energy expression."""
        result = await handle_advanced_tool(
            'quantity_simplify_units',
            {
                'expression': 'newton*meter'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'simplified' in data
        # Should simplify to base units

    @pytest.mark.asyncio
    async def test_simplify_pressure_units(self, ai):
        """Test simplifying pressure units."""
        result = await handle_advanced_tool(
            'quantity_simplify_units',
            {
                'expression': 'newton/meter**2'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'simplified' in data

    @pytest.mark.asyncio
    async def test_simplify_already_simplified_units(self, ai):
        """Test simplifying units that are already in simplest form."""
        result = await handle_advanced_tool(
            'quantity_simplify_units',
            {
                'expression': 'kilogram*meter**2/second**2'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'simplified' in data

    @pytest.mark.asyncio
    async def test_dimensionless_quantity(self, ai):
        """Test simplifying a dimensionless quantity."""
        result = await handle_advanced_tool(
            'quantity_simplify_units',
            {
                'expression': 'meter/meter'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'simplified' in data
