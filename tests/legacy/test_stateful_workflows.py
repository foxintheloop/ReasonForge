"""
Tests for Stateful Workflows - Multi-tool Chains.

Tests complete workflows that use multiple tools in sequence,
demonstrating state management across tool calls.
"""

import pytest
import json
from src.reasonforge_mcp.advanced_tools import handle_advanced_tool


class TestVariableToSolveWorkflow:
    """Test workflow: Define variable → Create expression → Solve."""

    @pytest.mark.asyncio
    async def test_complete_solving_workflow(self, ai):
        """Test complete workflow from variable introduction to solving."""
        # Step 1: Introduce variable
        step1 = await handle_advanced_tool(
            'intro',
            {'name': 'x', 'positive_assumptions': ['real']},
            ai
        )
        data1 = json.loads(step1[0].text)
        assert data1['status'] == 'success'

        # Step 2: Create expression
        step2 = await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 - 4', 'key': 'equation1'},
            ai
        )
        data2 = json.loads(step2[0].text)
        assert data2['status'] == 'success'

        # Step 3: Solve equation
        step3 = await handle_advanced_tool(
            'solve_algebraically',
            {
                'expression_key': 'equation1',
                'variable': 'x',
                'domain': 'real'
            },
            ai
        )
        data3 = json.loads(step3[0].text)
        assert 'solutions' in data3


class TestODEWorkflow:
    """Test workflow: Variable → Function → ODE → Solve."""

    @pytest.mark.asyncio
    async def test_ode_solving_workflow(self, ai):
        """Test complete ODE solving workflow."""
        # Step 1: Introduce independent variable
        step1 = await handle_advanced_tool(
            'intro',
            {'name': 't', 'positive_assumptions': ['real']},
            ai
        )
        assert json.loads(step1[0].text)['status'] == 'success'

        # Step 2: Introduce function
        step2 = await handle_advanced_tool(
            'introduce_function',
            {'name': 'y'},
            ai
        )
        assert json.loads(step2[0].text)['status'] == 'success'

        # Step 3: Define ODE
        step3 = await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'Derivative(y(t), t) - y(t)', 'key': 'ode1'},
            ai
        )
        assert json.loads(step3[0].text)['status'] == 'success'

        # Step 4: Solve ODE
        step4 = await handle_advanced_tool(
            'dsolve_ode',
            {'equation_key': 'ode1', 'function_name': 'y'},
            ai
        )
        data4 = json.loads(step4[0].text)
        assert data4['status'] == 'success'
        assert 'solution' in data4


class TestVectorCalculusWorkflow:
    """Test workflow: Coord system → Vector field → Curl/Div."""

    @pytest.mark.asyncio
    async def test_vector_field_analysis_workflow(self, ai):
        """Test complete vector field analysis workflow."""
        # Step 1: Create coordinate system
        step1 = await handle_advanced_tool(
            'create_coordinate_system',
            {'name': 'C', 'coord_type': 'cartesian'},
            ai
        )
        assert json.loads(step1[0].text)['status'] == 'success'

        # Step 2: Create vector field
        step2 = await handle_advanced_tool(
            'create_vector_field',
            {
                'coord_system_name': 'C',
                'components': {'i': 'y', 'j': '-x', 'k': '0'},
                'key': 'rotation_field'
            },
            ai
        )
        assert json.loads(step2[0].text)['status'] == 'success'

        # Step 3: Calculate curl
        step3 = await handle_advanced_tool(
            'calculate_curl',
            {'vector_field_key': 'rotation_field'},
            ai
        )
        assert 'curl' in json.loads(step3[0].text)

        # Step 4: Calculate divergence
        step4 = await handle_advanced_tool(
            'calculate_divergence',
            {'vector_field_key': 'rotation_field'},
            ai
        )
        assert 'divergence' in json.loads(step4[0].text)


class TestMatrixWorkflow:
    """Test workflow: Create matrix → Determinant → Inverse → Eigenvalues."""

    @pytest.mark.asyncio
    async def test_matrix_analysis_workflow(self, ai):
        """Test complete matrix analysis workflow."""
        # Step 1: Create matrix
        step1 = await handle_advanced_tool(
            'create_matrix',
            {
                'elements': [['4', '2'], ['1', '3']],
                'key': 'M1'
            },
            ai
        )
        assert json.loads(step1[0].text)['status'] == 'success'

        # Step 2: Compute determinant
        step2 = await handle_advanced_tool(
            'matrix_determinant',
            {'matrix_key': 'M1'},
            ai
        )
        assert 'determinant' in json.loads(step2[0].text)

        # Step 3: Compute inverse
        step3 = await handle_advanced_tool(
            'matrix_inverse',
            {'matrix_key': 'M1'},
            ai
        )
        assert 'inverse' in json.loads(step3[0].text)

        # Step 4: Find eigenvalues
        step4 = await handle_advanced_tool(
            'matrix_eigenvalues',
            {'matrix_key': 'M1'},
            ai
        )
        assert 'eigenvalues' in json.loads(step4[0].text)

        # Step 5: Find eigenvectors
        step5 = await handle_advanced_tool(
            'matrix_eigenvectors',
            {'matrix_key': 'M1'},
            ai
        )
        assert 'eigenvectors' in json.loads(step5[0].text)


class TestExpressionManipulationWorkflow:
    """Test workflow: Define → Differentiate → Integrate → Simplify."""

    @pytest.mark.asyncio
    async def test_expression_manipulation_workflow(self, ai):
        """Test complete expression manipulation workflow."""
        # Step 1: Introduce variable
        await handle_advanced_tool('intro', {'name': 'x'}, ai)

        # Step 2: Create expression
        step2 = await handle_advanced_tool(
            'introduce_expression',
            {'expression': '(x + 1)**3', 'key': 'expr1'},
            ai
        )
        assert json.loads(step2[0].text)['status'] == 'success'

        # Step 3: Differentiate
        step3 = await handle_advanced_tool(
            'differentiate_expression',
            {
                'expression_key': 'expr1',
                'variable': 'x',
                'store_key': 'derivative1'
            },
            ai
        )
        assert json.loads(step3[0].text)['result_key'] == 'derivative1'

        # Step 4: Integrate the derivative
        step4 = await handle_advanced_tool(
            'integrate_expression',
            {
                'expression_key': 'derivative1',
                'variable': 'x',
                'store_key': 'integral1'
            },
            ai
        )
        assert json.loads(step4[0].text)['result_key'] == 'integral1'

        # Step 5: Simplify
        step5 = await handle_advanced_tool(
            'simplify_expression',
            {'expression_key': 'integral1', 'method': 'default'},
            ai
        )
        assert 'result_key' in json.loads(step5[0].text)


class TestLinearSystemWorkflow:
    """Test workflow: Variables → Expressions → Solve linear system."""

    @pytest.mark.asyncio
    async def test_linear_system_workflow(self, ai):
        """Test solving a linear system workflow."""
        # Step 1: Introduce variables
        step1 = await handle_advanced_tool(
            'intro_many',
            {'names': ['x', 'y', 'z']},
            ai
        )
        assert json.loads(step1[0].text)['status'] == 'success'

        # Step 2: Create equations
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x + y + z - 6', 'key': 'eq1'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': '2*x - y + z - 1', 'key': 'eq2'},
            ai
        )
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x + 2*y - z - 3', 'key': 'eq3'},
            ai
        )

        # Step 3: Solve system
        step4 = await handle_advanced_tool(
            'solve_linear_system',
            {
                'expression_keys': ['eq1', 'eq2', 'eq3'],
                'variables': ['x', 'y', 'z']
            },
            ai
        )
        data = json.loads(step4[0].text)
        assert data['status'] == 'success'
        assert 'solutions' in data


class TestSubstitutionChain:
    """Test workflow: Expression → Substitute → Simplify."""

    @pytest.mark.asyncio
    async def test_substitution_chain_workflow(self, ai):
        """Test expression substitution chain."""
        # Setup
        await handle_advanced_tool('intro_many', {'names': ['x', 'y']}, ai)
        await handle_advanced_tool(
            'introduce_expression',
            {'expression': 'x**2 + 2*x*y + y**2', 'key': 'expr1'},
            ai
        )

        # Step 1: Substitute x=1, y=2
        step1 = await handle_advanced_tool(
            'substitute_expression',
            {
                'expression_key': 'expr1',
                'substitutions': {'x': '1', 'y': '2'},
                'store_key': 'subst1'
            },
            ai
        )
        assert json.loads(step1[0].text)['result_key'] == 'subst1'

        # Step 2: Simplify result
        step2 = await handle_advanced_tool(
            'simplify_expression',
            {'expression_key': 'subst1', 'method': 'default'},
            ai
        )
        data = json.loads(step2[0].text)
        assert 'result' in data
        # (1+2)^2 = 9
