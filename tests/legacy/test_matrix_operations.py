"""
Tests for Enhanced Matrix Operations Tools (5 tools).

Tools tested:
1. create_matrix - Create and store named matrices
2. matrix_determinant - Compute determinant
3. matrix_inverse - Compute inverse
4. matrix_eigenvalues - Find eigenvalues
5. matrix_eigenvectors - Find eigenvectors
"""

import pytest
import json
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestCreateMatrix:
    """Test the 'create_matrix' tool."""

    @pytest.mark.asyncio
    async def test_create_matrix_numeric_2x2(self, ai):
        """Test creating a numeric 2x2 matrix."""
        result = await handle_advanced_tool(
            'create_matrix',
            {
                'elements': [['1', '2'], ['3', '4']],
                'key': 'M1'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert data['key'] == 'M1'
        assert 'M1' in ai.matrices
        assert 'latex' in data

    @pytest.mark.asyncio
    async def test_create_matrix_symbolic(self, ai):
        """Test creating a symbolic matrix."""
        await handle_advanced_tool('intro_many', {'names': ['a', 'b', 'c', 'd']}, ai)

        result = await handle_advanced_tool(
            'create_matrix',
            {
                'elements': [['a', 'b'], ['c', 'd']],
                'key': 'Msym'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'Msym' in ai.matrices

    @pytest.mark.asyncio
    async def test_create_matrix_3x3_identity(self, ai):
        """Test creating a 3x3 identity matrix."""
        result = await handle_advanced_tool(
            'create_matrix',
            {
                'elements': [
                    ['1', '0', '0'],
                    ['0', '1', '0'],
                    ['0', '0', '1']
                ],
                'key': 'I3'
            },
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'I3' in ai.matrices

    @pytest.mark.asyncio
    async def test_create_matrix_auto_key(self, ai):
        """Test auto-generated matrix key."""
        result = await handle_advanced_tool(
            'create_matrix',
            {'elements': [['1', '0'], ['0', '1']]},
            ai
        )
        data = json.loads(result[0].text)

        assert data['status'] == 'success'
        assert 'key' in data
        assert data['key'].startswith('matrix_')


class TestMatrixDeterminant:
    """Test the 'matrix_determinant' tool."""

    @pytest.mark.asyncio
    async def test_determinant_2x2(self, ai):
        """Test computing determinant of a 2x2 matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {'elements': [['1', '2'], ['3', '4']], 'key': 'M1'},
            ai
        )

        result = await handle_advanced_tool(
            'matrix_determinant',
            {'matrix_key': 'M1'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'determinant' in data
        # det([[1,2],[3,4]]) = -2
        assert '-2' in data['determinant']

    @pytest.mark.asyncio
    async def test_determinant_identity_matrix(self, ai):
        """Test determinant of identity matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {'elements': [['1', '0'], ['0', '1']], 'key': 'I2'},
            ai
        )

        result = await handle_advanced_tool(
            'matrix_determinant',
            {'matrix_key': 'I2'},
            ai
        )
        data = json.loads(result[0].text)

        assert '1' in data['determinant']

    @pytest.mark.asyncio
    async def test_determinant_invalid_key(self, ai):
        """Test determinant with invalid matrix key."""
        result = await handle_advanced_tool(
            'matrix_determinant',
            {'matrix_key': 'nonexistent'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data


class TestMatrixInverse:
    """Test the 'matrix_inverse' tool."""

    @pytest.mark.asyncio
    async def test_inverse_2x2(self, ai):
        """Test computing inverse of a 2x2 matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {'elements': [['1', '2'], ['3', '4']], 'key': 'M1'},
            ai
        )

        result = await handle_advanced_tool(
            'matrix_inverse',
            {'matrix_key': 'M1'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'result_key' in data
        assert 'inverse' in data
        assert data['result_key'] in ai.matrices

    @pytest.mark.asyncio
    async def test_inverse_singular_matrix(self, ai):
        """Test inverse of a singular (non-invertible) matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {'elements': [['1', '2'], ['2', '4']], 'key': 'Msing'},
            ai
        )

        result = await handle_advanced_tool(
            'matrix_inverse',
            {'matrix_key': 'Msing'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data
        assert 'not invertible' in data['error'].lower()


class TestMatrixEigenvalues:
    """Test the 'matrix_eigenvalues' tool."""

    @pytest.mark.asyncio
    async def test_eigenvalues_2x2(self, ai):
        """Test finding eigenvalues of a 2x2 matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {'elements': [['3', '1'], ['1', '3']], 'key': 'M1'},
            ai
        )

        result = await handle_advanced_tool(
            'matrix_eigenvalues',
            {'matrix_key': 'M1'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'eigenvalues' in data
        # Matrix [[3,1],[1,3]] has eigenvalues 4 and 2

    @pytest.mark.asyncio
    async def test_eigenvalues_identity(self, ai):
        """Test eigenvalues of identity matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {'elements': [['1', '0'], ['0', '1']], 'key': 'I2'},
            ai
        )

        result = await handle_advanced_tool(
            'matrix_eigenvalues',
            {'matrix_key': 'I2'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'eigenvalues' in data


class TestMatrixEigenvectors:
    """Test the 'matrix_eigenvectors' tool."""

    @pytest.mark.asyncio
    async def test_eigenvectors_2x2(self, ai):
        """Test finding eigenvectors of a 2x2 matrix."""
        await handle_advanced_tool(
            'create_matrix',
            {'elements': [['3', '1'], ['1', '3']], 'key': 'M1'},
            ai
        )

        result = await handle_advanced_tool(
            'matrix_eigenvectors',
            {'matrix_key': 'M1'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'eigenvectors' in data
        assert isinstance(data['eigenvectors'], list)
        assert len(data['eigenvectors']) > 0

    @pytest.mark.asyncio
    async def test_eigenvectors_invalid_key(self, ai):
        """Test eigenvectors with invalid matrix key."""
        result = await handle_advanced_tool(
            'matrix_eigenvectors',
            {'matrix_key': 'nonexistent'},
            ai
        )
        data = json.loads(result[0].text)

        assert 'error' in data
