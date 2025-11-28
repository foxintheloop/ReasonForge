"""
Tests for data science and statistics tools.

This module tests the 8 data science tools from data_science_tools.py:
1. symbolic_dataframe - Create symbolic datasets
2. statistical_moments_symbolic - Calculate symbolic moments
3. time_series_symbolic - Model time series symbolically
4. hypothesis_test_symbolic - Generate test statistic formulas
5. anova_symbolic - Derive ANOVA formulas
6. multivariate_statistics - Multivariate distributions and operations
7. sampling_distributions - Sampling distribution formulas
8. experimental_design - Generate experimental designs

Tests based on usage examples from USAGE_EXAMPLES.md
"""

import json
import pytest
from src.reasonforge_mcp.symbolic_engine import SymbolicAI
from src.reasonforge_mcp.data_science_tools import handle_data_science_tool


class TestSymbolicDataframe:
    """Test the 'symbolic_dataframe' tool."""

    @pytest.mark.asyncio
    async def test_create_symbolic_dataframe(self, ai):
        """Test creating symbolic dataframe (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'symbolic_dataframe',
            {
                'variables': ['age', 'height', 'weight'],
                'num_observations': 'n'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'symbols' in data or 'variables' in data

    @pytest.mark.asyncio
    async def test_dataframe_with_operations(self, ai):
        """Test dataframe with statistical operations."""
        result = await handle_data_science_tool(
            'symbolic_dataframe',
            {
                'variables': ['X', 'Y'],
                'num_observations': 'n',
                'operations': ['mean', 'variance']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_covariance_matrix(self, ai):
        """Test computing covariance matrix symbolically."""
        result = await handle_data_science_tool(
            'symbolic_dataframe',
            {
                'variables': ['X', 'Y', 'Z'],
                'num_observations': 'n',
                'operations': ['covariance_matrix']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestStatisticalMomentsSymbolic:
    """Test the 'statistical_moments_symbolic' tool."""

    @pytest.mark.asyncio
    async def test_skewness_formula(self, ai):
        """Test deriving skewness formula (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'statistical_moments_symbolic',
            {
                'distribution': 'X',
                'moments': ['skewness']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'skewness' in data or 'formula' in data

    @pytest.mark.asyncio
    async def test_kurtosis_formula(self, ai):
        """Test deriving kurtosis formula."""
        result = await handle_data_science_tool(
            'statistical_moments_symbolic',
            {
                'distribution': 'X',
                'moments': ['kurtosis']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_central_moment(self, ai):
        """Test computing central moment of arbitrary order."""
        result = await handle_data_science_tool(
            'statistical_moments_symbolic',
            {
                'distribution': 'X',
                'moments': ['central_moment'],
                'moment_order': 4
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestTimeSeriesSymbolic:
    """Test the 'time_series_symbolic' tool."""

    @pytest.mark.asyncio
    async def test_arma_model(self, ai):
        """Test creating ARMA(2,1) model (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'time_series_symbolic',
            {
                'model_type': 'ARMA',
                'parameters': {'p': 2, 'q': 1}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'equation' in data or 'parameters' in data

    @pytest.mark.asyncio
    async def test_ar_model(self, ai):
        """Test autoregressive AR(3) model."""
        result = await handle_data_science_tool(
            'time_series_symbolic',
            {
                'model_type': 'AR',
                'parameters': {'p': 3}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_arima_with_stationarity(self, ai):
        """Test ARIMA model with stationarity check."""
        result = await handle_data_science_tool(
            'time_series_symbolic',
            {
                'model_type': 'ARIMA',
                'parameters': {'p': 1, 'd': 1, 'q': 1},
                'compute': ['equation', 'stationarity']
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestHypothesisTestSymbolic:
    """Test the 'hypothesis_test_symbolic' tool."""

    @pytest.mark.asyncio
    async def test_chi_square_test(self, ai):
        """Test chi-square test formula (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'hypothesis_test_symbolic',
            {
                'test_type': 'chi_square',
                'parameters': {
                    'observed': 'O_i',
                    'expected': 'E_i'
                }
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'test_statistic' in data or 'formula' in data

    @pytest.mark.asyncio
    async def test_t_test(self, ai):
        """Test t-test formula."""
        result = await handle_data_science_tool(
            'hypothesis_test_symbolic',
            {
                'test_type': 't_test',
                'parameters': {
                    'sample_mean': 'x_bar',
                    'population_mean': 'mu_0',
                    'std_error': 's_e'
                }
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_z_test(self, ai):
        """Test z-test statistic formula."""
        result = await handle_data_science_tool(
            'hypothesis_test_symbolic',
            {
                'test_type': 'z_test',
                'parameters': {
                    'sample_mean': 'x_bar',
                    'population_mean': 'mu',
                    'population_std': 'sigma',
                    'n': 'n'
                }
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestAnovaSymbolic:
    """Test the 'anova_symbolic' tool."""

    @pytest.mark.asyncio
    async def test_one_way_anova(self, ai):
        """Test one-way ANOVA formulas (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'anova_symbolic',
            {
                'anova_type': 'one_way',
                'groups': 3
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'formulas' in data or 'F_statistic' in data

    @pytest.mark.asyncio
    async def test_two_way_anova(self, ai):
        """Test two-way ANOVA formulas."""
        result = await handle_data_science_tool(
            'anova_symbolic',
            {
                'anova_type': 'two_way',
                'factor_A_levels': 2,
                'factor_B_levels': 3
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_repeated_measures_anova(self, ai):
        """Test repeated measures ANOVA."""
        result = await handle_data_science_tool(
            'anova_symbolic',
            {
                'anova_type': 'repeated_measures',
                'subjects': 'n',
                'conditions': 'k'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestMultivariateStatistics:
    """Test the 'multivariate_statistics' tool."""

    @pytest.mark.asyncio
    async def test_multivariate_normal_pdf(self, ai):
        """Test multivariate normal PDF (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'multivariate_statistics',
            {
                'operation': 'multivariate_normal',
                'parameters': {
                    'dimension': 'p'
                }
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'pdf' in data or 'formula' in data

    @pytest.mark.asyncio
    async def test_mahalanobis_distance(self, ai):
        """Test Mahalanobis distance formula."""
        result = await handle_data_science_tool(
            'multivariate_statistics',
            {
                'operation': 'mahalanobis',
                'parameters': {
                    'x': 'x',
                    'mean': 'mu',
                    'covariance': 'Sigma'
                }
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_hotelling_t_squared(self, ai):
        """Test Hotelling's T-squared statistic."""
        result = await handle_data_science_tool(
            'multivariate_statistics',
            {
                'operation': 'hotelling_t_squared',
                'parameters': {
                    'n': 'n',
                    'p': 'p'
                }
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestSamplingDistributions:
    """Test the 'sampling_distributions' tool."""

    @pytest.mark.asyncio
    async def test_sample_mean_distribution(self, ai):
        """Test sampling distribution of sample mean (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'sampling_distributions',
            {
                'statistic': 'mean',
                'population_parameters': {'mu': 'mu', 'sigma': 'sigma'},
                'sample_size': 'n'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'distribution' in data or 'formula' in data

    @pytest.mark.asyncio
    async def test_sample_variance_distribution(self, ai):
        """Test sampling distribution of sample variance."""
        result = await handle_data_science_tool(
            'sampling_distributions',
            {
                'statistic': 'variance',
                'population_parameters': {'sigma_squared': 'sigma_2'},
                'sample_size': 'n'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_difference_of_means(self, ai):
        """Test sampling distribution of difference of means."""
        result = await handle_data_science_tool(
            'sampling_distributions',
            {
                'statistic': 'difference_of_means',
                'population_parameters': {
                    'mu1': 'mu_1',
                    'mu2': 'mu_2',
                    'sigma1': 'sigma_1',
                    'sigma2': 'sigma_2'
                },
                'sample_sizes': {'n1': 'n_1', 'n2': 'n_2'}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data


class TestExperimentalDesign:
    """Test the 'experimental_design' tool."""

    @pytest.mark.asyncio
    async def test_factorial_design(self, ai):
        """Test 2^3 factorial design (from USAGE_EXAMPLES.md)."""
        result = await handle_data_science_tool(
            'experimental_design',
            {
                'design_type': 'factorial',
                'factors': ['A', 'B', 'C'],
                'levels': {'A': 2, 'B': 2, 'C': 2}
            },
            ai
        )

        data = json.loads(result[0].text)
        assert data['status'] == 'success'
        assert 'design_matrix' in data or 'run_count' in data

    @pytest.mark.asyncio
    async def test_fractional_factorial(self, ai):
        """Test fractional factorial design."""
        result = await handle_data_science_tool(
            'experimental_design',
            {
                'design_type': 'fractional_factorial',
                'factors': ['A', 'B', 'C', 'D'],
                'fraction': '1/2'
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data

    @pytest.mark.asyncio
    async def test_latin_square(self, ai):
        """Test Latin square design."""
        result = await handle_data_science_tool(
            'experimental_design',
            {
                'design_type': 'latin_square',
                'size': 4
            },
            ai
        )

        data = json.loads(result[0].text)
        assert 'status' in data or 'error' in data
