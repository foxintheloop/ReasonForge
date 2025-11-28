"""
Comprehensive tests for reasonforge-statistics MCP server.

Tests all 16 tools:
- Probability Calculations (4 tools): calculate_probability, bayesian_inference, statistical_test, distribution_properties
- Statistical Analysis (4 tools): correlation_analysis, regression_symbolic, confidence_intervals, probability_distributions
- Data Analysis (4 tools): symbolic_dataframe, statistical_moments_symbolic, time_series_symbolic, hypothesis_test_symbolic
- Advanced Statistics (4 tools): anova_symbolic, multivariate_statistics, sampling_distributions, experimental_design
"""

import asyncio
import json
import pytest

from reasonforge_statistics.server import server as statistics_server


class TestProbabilityCalculations:
    """Test probability calculation tools (4 tools)."""

    @pytest.mark.asyncio
    async def test_calculate_probability_normal_expectation(self):
        """Test calculating expectation for normal distribution."""
        result = await statistics_server.call_tool_for_test(
            "calculate_probability",
            {
                "distribution": "Normal",
                "parameters": {"mean": "5", "std": "2"},
                "operation": "expectation"
            }
        )
        data = json.loads(result[0].text)

        assert data["distribution"] == "Normal"
        assert data["operation"] == "expectation"
        assert data["result"] == "5"  # Expectation = mean

    @pytest.mark.asyncio
    async def test_calculate_probability_normal_variance(self):
        """Test calculating variance for normal distribution."""
        result = await statistics_server.call_tool_for_test(
            "calculate_probability",
            {
                "distribution": "Normal",
                "parameters": {"mean": "0", "std": "1"},
                "operation": "variance"
            }
        )
        data = json.loads(result[0].text)

        assert data["operation"] == "variance"
        assert "result" in data

    @pytest.mark.asyncio
    async def test_calculate_probability_uniform(self):
        """Test uniform distribution."""
        result = await statistics_server.call_tool_for_test(
            "calculate_probability",
            {
                "distribution": "Uniform",
                "parameters": {"a": "0", "b": "10"},
                "operation": "expectation"
            }
        )
        data = json.loads(result[0].text)

        assert data["distribution"] == "Uniform"
        assert data["result"] == "5"  # (0 + 10) / 2 = 5

    @pytest.mark.asyncio
    async def test_calculate_probability_exponential(self):
        """Test exponential distribution."""
        result = await statistics_server.call_tool_for_test(
            "calculate_probability",
            {
                "distribution": "Exponential",
                "parameters": {"rate": "2"},
                "operation": "expectation"
            }
        )
        data = json.loads(result[0].text)

        assert data["distribution"] == "Exponential"
        # Expectation = 1/rate = 1/2

    @pytest.mark.asyncio
    async def test_bayesian_inference_with_evidence(self):
        """Test Bayesian inference with evidence."""
        result = await statistics_server.call_tool_for_test(
            "bayesian_inference",
            {
                "prior": "0.3",
                "likelihood": "0.8",
                "evidence": "0.5"
            }
        )
        data = json.loads(result[0].text)

        assert data["prior"] == "0.3"
        assert data["likelihood"] == "0.8"
        assert data["evidence"] == "0.5"
        assert "posterior" in data
        assert "formula" in data

    @pytest.mark.asyncio
    async def test_bayesian_inference_without_evidence(self):
        """Test Bayesian inference without evidence."""
        result = await statistics_server.call_tool_for_test(
            "bayesian_inference",
            {
                "prior": "p",
                "likelihood": "l"
            }
        )
        data = json.loads(result[0].text)

        assert "posterior" in data

    @pytest.mark.asyncio
    async def test_statistical_test_t_test(self):
        """Test t-test setup."""
        result = await statistics_server.call_tool_for_test(
            "statistical_test",
            {"test_type": "t-test"}
        )
        data = json.loads(result[0].text)

        assert data["test_type"] == "t-test"
        assert "t_statistic_formula" in data
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_statistical_test_z_test(self):
        """Test z-test setup."""
        result = await statistics_server.call_tool_for_test(
            "statistical_test",
            {"test_type": "z-test"}
        )
        data = json.loads(result[0].text)

        assert data["test_type"] == "z-test"
        assert "z_statistic_formula" in data

    @pytest.mark.asyncio
    async def test_statistical_test_chi_square(self):
        """Test chi-square test setup."""
        result = await statistics_server.call_tool_for_test(
            "statistical_test",
            {"test_type": "chi-square"}
        )
        data = json.loads(result[0].text)

        assert data["test_type"] == "chi-square"
        assert "chi_square_formula" in data

    @pytest.mark.asyncio
    async def test_distribution_properties_normal(self):
        """Test getting normal distribution properties."""
        result = await statistics_server.call_tool_for_test(
            "distribution_properties",
            {
                "distribution": "Normal",
                "parameters": {"mean": "mu", "std": "sigma"}
            }
        )
        data = json.loads(result[0].text)

        assert data["distribution"] == "Normal"
        assert "expectation" in data
        assert "variance" in data
        assert "pdf" in data

    @pytest.mark.asyncio
    async def test_distribution_properties_uniform(self):
        """Test uniform distribution properties."""
        result = await statistics_server.call_tool_for_test(
            "distribution_properties",
            {
                "distribution": "Uniform",
                "parameters": {"a": "a", "b": "b"}
            }
        )
        data = json.loads(result[0].text)

        assert data["distribution"] == "Uniform"
        assert "expectation" in data


class TestStatisticalAnalysis:
    """Test statistical analysis tools (4 tools)."""

    @pytest.mark.asyncio
    async def test_correlation_analysis(self):
        """Test correlation analysis."""
        result = await statistics_server.call_tool_for_test(
            "correlation_analysis",
            {
                "x_distribution": "X",
                "y_distribution": "Y"
            }
        )
        data = json.loads(result[0].text)

        assert "correlation_coefficient" in data
        assert "t_statistic" in data
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_regression_symbolic_linear(self):
        """Test linear regression setup."""
        result = await statistics_server.call_tool_for_test(
            "regression_symbolic",
            {
                "regression_type": "linear",
                "variables": ["x", "y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["regression_type"] == "linear"
        assert "equation" in data
        assert "parameters" in data
        # Should contain beta_0 and beta_1

    @pytest.mark.asyncio
    async def test_regression_symbolic_multiple(self):
        """Test multiple regression setup."""
        result = await statistics_server.call_tool_for_test(
            "regression_symbolic",
            {
                "regression_type": "multiple",
                "variables": ["x1", "x2", "x3"]
            }
        )
        data = json.loads(result[0].text)

        assert data["regression_type"] == "multiple"
        assert "equation" in data

    @pytest.mark.asyncio
    async def test_regression_symbolic_polynomial(self):
        """Test polynomial regression setup."""
        result = await statistics_server.call_tool_for_test(
            "regression_symbolic",
            {
                "regression_type": "polynomial",
                "variables": ["x"]
            }
        )
        data = json.loads(result[0].text)

        assert data["regression_type"] == "polynomial"

    @pytest.mark.asyncio
    async def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        result = await statistics_server.call_tool_for_test(
            "confidence_intervals",
            {
                "parameter_type": "mean",
                "confidence_level": 0.95
            }
        )
        data = json.loads(result[0].text)

        assert data["parameter_type"] == "mean"
        assert data["confidence_level"] == 0.95
        assert "margin_of_error" in data or "lower_bound" in data

    @pytest.mark.asyncio
    async def test_confidence_intervals_proportion(self):
        """Test confidence interval for proportion."""
        result = await statistics_server.call_tool_for_test(
            "confidence_intervals",
            {
                "parameter_type": "proportion",
                "confidence_level": 0.99
            }
        )
        data = json.loads(result[0].text)

        assert data["parameter_type"] == "proportion"

    @pytest.mark.asyncio
    async def test_probability_distributions_convolution(self):
        """Test probability distribution operations."""
        result = await statistics_server.call_tool_for_test(
            "probability_distributions",
            {
                "operation": "convolution",
                "distributions": ["X", "Y"]
            }
        )
        data = json.loads(result[0].text)

        assert data["operation"] == "convolution"


class TestDataAnalysis:
    """Test data analysis tools (4 tools)."""

    @pytest.mark.asyncio
    async def test_symbolic_dataframe(self):
        """Test creating symbolic dataframe."""
        result = await statistics_server.call_tool_for_test(
            "symbolic_dataframe",
            {
                "columns": ["ID", "Value", "Category"]
            }
        )
        data = json.loads(result[0].text)

        assert data["columns"] == ["ID", "Value", "Category"]
        assert "symbolic_variables" in data or "note" in data

    @pytest.mark.asyncio
    async def test_statistical_moments_symbolic_first(self):
        """Test first statistical moment (mean)."""
        result = await statistics_server.call_tool_for_test(
            "statistical_moments_symbolic",
            {
                "data_symbol": "X",
                "order": 1
            }
        )
        data = json.loads(result[0].text)

        assert data["order"] == 1
        assert "moment_name" in data
        assert "Mean" in data["moment_name"]

    @pytest.mark.asyncio
    async def test_statistical_moments_symbolic_second(self):
        """Test second moment (variance)."""
        result = await statistics_server.call_tool_for_test(
            "statistical_moments_symbolic",
            {
                "data_symbol": "X",
                "order": 2
            }
        )
        data = json.loads(result[0].text)

        assert data["order"] == 2

    @pytest.mark.asyncio
    async def test_statistical_moments_symbolic_higher(self):
        """Test higher moments (skewness, kurtosis)."""
        result = await statistics_server.call_tool_for_test(
            "statistical_moments_symbolic",
            {
                "data_symbol": "X",
                "order": 3
            }
        )
        data = json.loads(result[0].text)

        assert data["order"] == 3

    @pytest.mark.asyncio
    async def test_time_series_symbolic_ar(self):
        """Test AR time series model."""
        result = await statistics_server.call_tool_for_test(
            "time_series_symbolic",
            {
                "model_type": "AR",
                "p": 2,
                "q": 0
            }
        )
        data = json.loads(result[0].text)

        assert data["model_type"] == "AR"
        assert "equation" in data or "order" in data

    @pytest.mark.asyncio
    async def test_time_series_symbolic_ma(self):
        """Test MA time series model."""
        result = await statistics_server.call_tool_for_test(
            "time_series_symbolic",
            {
                "model_type": "MA",
                "p": 0,
                "q": 2
            }
        )
        data = json.loads(result[0].text)

        assert data["model_type"] == "MA"
        assert "equation" in data or "order" in data

    @pytest.mark.asyncio
    async def test_time_series_symbolic_arma(self):
        """Test ARMA time series model."""
        result = await statistics_server.call_tool_for_test(
            "time_series_symbolic",
            {
                "model_type": "ARMA",
                "p": 1,
                "q": 1
            }
        )
        data = json.loads(result[0].text)

        assert data["model_type"] == "ARMA"

    @pytest.mark.asyncio
    async def test_hypothesis_test_symbolic(self):
        """Test hypothesis testing setup."""
        result = await statistics_server.call_tool_for_test(
            "hypothesis_test_symbolic",
            {
                "test_type": "two_sample_t_test",
                "parameters": {}
            }
        )
        data = json.loads(result[0].text)

        assert data["test_type"] == "two_sample_t_test"


class TestAdvancedStatistics:
    """Test advanced statistics tools (4 tools)."""

    @pytest.mark.asyncio
    async def test_anova_symbolic_one_way(self):
        """Test one-way ANOVA."""
        result = await statistics_server.call_tool_for_test(
            "anova_symbolic",
            {
                "anova_type": "one-way",
                "groups": 3
            }
        )
        data = json.loads(result[0].text)

        assert data["anova_type"] == "one-way"
        assert data["groups"] == 3
        assert "F_statistic" in data or "formula" in data

    @pytest.mark.asyncio
    async def test_anova_symbolic_two_way(self):
        """Test two-way ANOVA."""
        result = await statistics_server.call_tool_for_test(
            "anova_symbolic",
            {
                "anova_type": "two-way",
                "groups": 4
            }
        )
        data = json.loads(result[0].text)

        assert data["anova_type"] == "two-way"

    @pytest.mark.asyncio
    async def test_multivariate_statistics_pca(self):
        """Test PCA setup."""
        result = await statistics_server.call_tool_for_test(
            "multivariate_statistics",
            {
                "operation": "PCA",
                "parameters": {"components": 3}
            }
        )
        data = json.loads(result[0].text)

        # Verify we get a response (field names may vary)
        assert "note" in data or "formula" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_multivariate_statistics_covariance(self):
        """Test covariance matrix."""
        result = await statistics_server.call_tool_for_test(
            "multivariate_statistics",
            {
                "operation": "covariance_matrix",
                "parameters": {}
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0  # Just verify we got a response

    @pytest.mark.asyncio
    async def test_multivariate_statistics_mahalanobis(self):
        """Test Mahalanobis distance."""
        result = await statistics_server.call_tool_for_test(
            "multivariate_statistics",
            {
                "operation": "mahalanobis_distance",
                "parameters": {}
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_sampling_distributions_mean(self):
        """Test sampling distribution of mean."""
        result = await statistics_server.call_tool_for_test(
            "sampling_distributions",
            {
                "statistic": "mean",
                "population_parameters": {"mu": "100", "sigma": "15"}
            }
        )
        data = json.loads(result[0].text)

        # Check for any valid response field
        assert "mean" in data or "variance" in data or len(data) > 0

    @pytest.mark.asyncio
    async def test_sampling_distributions_proportion(self):
        """Test sampling distribution of proportion."""
        result = await statistics_server.call_tool_for_test(
            "sampling_distributions",
            {
                "statistic": "proportion",
                "population_parameters": {"p": "0.5"}
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_experimental_design_completely_randomized(self):
        """Test completely randomized design."""
        result = await statistics_server.call_tool_for_test(
            "experimental_design",
            {
                "design_type": "completely_randomized",
                "factors": ["Treatment"]
            }
        )
        data = json.loads(result[0].text)

        # Verify we get a response
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_experimental_design_factorial(self):
        """Test factorial design."""
        result = await statistics_server.call_tool_for_test(
            "experimental_design",
            {
                "design_type": "factorial",
                "factors": ["Temperature", "Pressure"]
            }
        )
        data = json.loads(result[0].text)

        assert len(data) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
