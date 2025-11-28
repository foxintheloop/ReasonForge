"""
Comprehensive tests for Probability & Statistics Tools

Tests all 8 probability and statistics tools added to advanced_tools.py.
"""

import pytest
import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from reasonforge_mcp.symbolic_engine import SymbolicAI


# Import is async, so we'll use a helper
async def call_tool(ai, name, args):
    """Helper to call advanced tools."""
    from reasonforge_mcp.advanced_tools import handle_advanced_tool
    return await handle_advanced_tool(name, args, ai)


@pytest.fixture
def symbolic_ai():
    """Create a SymbolicAI instance for testing."""
    return SymbolicAI()


class TestCalculateProbability:
    """Test calculate_probability tool."""

    @pytest.mark.asyncio
    async def test_normal_pdf(self, symbolic_ai):
        """Test normal distribution PDF calculation."""
        args = {
            "distribution": "normal",
            "parameters": {"mu": "0", "sigma": "1"},
            "calculation": "pdf"
        }
        result = await call_tool(symbolic_ai, "calculate_probability", args)
        data = json.loads(result[0].text)

        assert "pdf" in data
        assert data["distribution"] == "normal"
        assert "latex" in data

    @pytest.mark.asyncio
    async def test_normal_expectation(self, symbolic_ai):
        """Test normal distribution expectation."""
        args = {
            "distribution": "normal",
            "parameters": {"mu": "5", "sigma": "2"},
            "calculation": "expectation"
        }
        result = await call_tool(symbolic_ai, "calculate_probability", args)
        data = json.loads(result[0].text)

        assert "expectation" in data
        assert "5" in data["expectation"]

    @pytest.mark.asyncio
    async def test_binomial_pdf(self, symbolic_ai):
        """Test binomial distribution."""
        args = {
            "distribution": "binomial",
            "parameters": {"n": 10, "p": "0.5"},
            "calculation": "pdf"
        }
        result = await call_tool(symbolic_ai, "calculate_probability", args)
        data = json.loads(result[0].text)

        assert "pdf" in data
        assert data["distribution"] == "binomial"

    @pytest.mark.asyncio
    async def test_poisson_expectation(self, symbolic_ai):
        """Test Poisson distribution expectation."""
        args = {
            "distribution": "poisson",
            "parameters": {"lambda": "3"},
            "calculation": "expectation"
        }
        result = await call_tool(symbolic_ai, "calculate_probability", args)
        data = json.loads(result[0].text)

        assert "expectation" in data
        assert "3" in data["expectation"]


class TestBayesianInference:
    """Test bayesian_inference tool."""

    @pytest.mark.asyncio
    async def test_simple_bayes(self, symbolic_ai):
        """Test simple Bayesian inference."""
        args = {
            "prior": "0.01",
            "likelihood": "0.9",
            "evidence": "0.05"
        }
        result = await call_tool(symbolic_ai, "bayesian_inference", args)
        data = json.loads(result[0].text)

        assert "posterior" in data
        assert "prior" in data
        assert "likelihood" in data
        assert data["formula"] == "P(H|E) = P(E|H) * P(H) / P(E)"

    @pytest.mark.asyncio
    async def test_symbolic_bayes(self, symbolic_ai):
        """Test Bayesian inference with symbolic expressions."""
        args = {
            "prior": "p",
            "likelihood": "q",
            "evidence": "r"
        }
        result = await call_tool(symbolic_ai, "bayesian_inference", args)
        data = json.loads(result[0].text)

        assert "posterior" in data
        assert "posterior_latex" in data

    @pytest.mark.asyncio
    async def test_bayes_no_evidence(self, symbolic_ai):
        """Test Bayesian inference without evidence (unnormalized)."""
        args = {
            "prior": "0.3",
            "likelihood": "0.8"
        }
        result = await call_tool(symbolic_ai, "bayesian_inference", args)
        data = json.loads(result[0].text)

        assert "posterior" in data


class TestStatisticalTest:
    """Test statistical_test tool."""

    @pytest.mark.asyncio
    async def test_t_test(self, symbolic_ai):
        """Test t-test statistic calculation."""
        args = {
            "test_type": "t_test",
            "sample_statistics": {
                "mean": "52",
                "mu_0": "50",
                "std": "5",
                "n": "25"
            }
        }
        result = await call_tool(symbolic_ai, "statistical_test", args)
        data = json.loads(result[0].text)

        assert "t_statistic" in data
        assert "degrees_of_freedom" in data
        assert data["formula"] == "t = (x_bar - mu_0) / (s / sqrt(n))"

    @pytest.mark.asyncio
    async def test_z_test(self, symbolic_ai):
        """Test z-test statistic calculation."""
        args = {
            "test_type": "z_test",
            "sample_statistics": {
                "mean": "105",
                "mu_0": "100",
                "sigma": "15",
                "n": "30"
            }
        }
        result = await call_tool(symbolic_ai, "statistical_test", args)
        data = json.loads(result[0].text)

        assert "z_statistic" in data
        assert "z_statistic_latex" in data

    @pytest.mark.asyncio
    async def test_chi_square(self, symbolic_ai):
        """Test chi-square statistic calculation."""
        args = {
            "test_type": "chi_square",
            "sample_statistics": {
                "observed": "25",
                "expected": "20"
            }
        }
        result = await call_tool(symbolic_ai, "statistical_test", args)
        data = json.loads(result[0].text)

        assert "chi_square_statistic" in data

    @pytest.mark.asyncio
    async def test_f_test(self, symbolic_ai):
        """Test F-test statistic calculation."""
        args = {
            "test_type": "f_test",
            "sample_statistics": {
                "var1": "25",
                "var2": "16"
            }
        }
        result = await call_tool(symbolic_ai, "statistical_test", args)
        data = json.loads(result[0].text)

        assert "f_statistic" in data


class TestDistributionProperties:
    """Test distribution_properties tool."""

    @pytest.mark.asyncio
    async def test_normal_properties(self, symbolic_ai):
        """Test calculating normal distribution properties."""
        args = {
            "distribution": "normal",
            "parameters": {"mu": "10", "sigma": "2"},
            "properties": ["mean", "variance", "std"]
        }
        result = await call_tool(symbolic_ai, "distribution_properties", args)
        data = json.loads(result[0].text)

        assert data["distribution"] == "normal"
        assert "mean" in data["properties"]
        assert "variance" in data["properties"]
        assert "std" in data["properties"]

    @pytest.mark.asyncio
    async def test_exponential_properties(self, symbolic_ai):
        """Test exponential distribution properties."""
        args = {
            "distribution": "exponential",
            "parameters": {"rate": "0.5"},
            "properties": ["mean", "variance"]
        }
        result = await call_tool(symbolic_ai, "distribution_properties", args)
        data = json.loads(result[0].text)

        assert "mean" in data["properties"]
        assert "variance" in data["properties"]

    @pytest.mark.asyncio
    async def test_all_properties(self, symbolic_ai):
        """Test all available properties."""
        args = {
            "distribution": "normal",
            "parameters": {"mu": "0", "sigma": "1"},
            "properties": ["mean", "variance", "std", "skewness", "kurtosis"]
        }
        result = await call_tool(symbolic_ai, "distribution_properties", args)
        data = json.loads(result[0].text)

        assert len(data["properties"]) == 5


class TestCorrelationAnalysis:
    """Test correlation_analysis tool."""

    @pytest.mark.asyncio
    async def test_covariance_formula(self, symbolic_ai):
        """Test covariance formula generation."""
        args = {
            "variable_x": "X",
            "variable_y": "Y",
            "calculate": ["covariance"]
        }
        result = await call_tool(symbolic_ai, "correlation_analysis", args)
        data = json.loads(result[0].text)

        assert "covariance_formula" in data["analysis"]

    @pytest.mark.asyncio
    async def test_correlation_formula(self, symbolic_ai):
        """Test correlation formula generation."""
        args = {
            "variable_x": "X",
            "variable_y": "Y",
            "calculate": ["correlation"]
        }
        result = await call_tool(symbolic_ai, "correlation_analysis", args)
        data = json.loads(result[0].text)

        assert "correlation_formula" in data["analysis"]

    @pytest.mark.asyncio
    async def test_all_analyses(self, symbolic_ai):
        """Test all correlation analyses."""
        args = {
            "variable_x": "X",
            "variable_y": "Y",
            "calculate": ["covariance", "correlation", "independence_test"]
        }
        result = await call_tool(symbolic_ai, "correlation_analysis", args)
        data = json.loads(result[0].text)

        assert len(data["analysis"]) == 3


class TestRegressionSymbolic:
    """Test regression_symbolic tool."""

    @pytest.mark.asyncio
    async def test_linear_regression(self, symbolic_ai):
        """Test linear regression model."""
        args = {
            "independent_vars": ["x"],
            "dependent_var": "y",
            "model_type": "linear"
        }
        result = await call_tool(symbolic_ai, "regression_symbolic", args)
        data = json.loads(result[0].text)

        assert "model" in data
        assert "beta_0" in data["model"]
        assert "beta_1" in data["model"]
        assert "ols_formula_beta_1" in data

    @pytest.mark.asyncio
    async def test_polynomial_regression(self, symbolic_ai):
        """Test polynomial regression model."""
        args = {
            "independent_vars": ["x"],
            "dependent_var": "y",
            "model_type": "polynomial",
            "polynomial_degree": 3
        }
        result = await call_tool(symbolic_ai, "regression_symbolic", args)
        data = json.loads(result[0].text)

        assert "model" in data
        assert data["degree"] == 3

    @pytest.mark.asyncio
    async def test_multiple_linear_regression(self, symbolic_ai):
        """Test multiple linear regression."""
        args = {
            "independent_vars": ["x1", "x2", "x3"],
            "dependent_var": "y",
            "model_type": "multiple_linear"
        }
        result = await call_tool(symbolic_ai, "regression_symbolic", args)
        data = json.loads(result[0].text)

        assert "model" in data
        assert "x1" in data["model"]
        assert "x2" in data["model"]
        assert "x3" in data["model"]


class TestConfidenceIntervals:
    """Test confidence_intervals tool."""

    @pytest.mark.asyncio
    async def test_mean_confidence_interval(self, symbolic_ai):
        """Test confidence interval for mean."""
        args = {
            "parameter": "mean",
            "sample_stats": {
                "x_bar": "50",
                "s": "5",
                "n": "25"
            }
        }
        result = await call_tool(symbolic_ai, "confidence_intervals", args)
        data = json.loads(result[0].text)

        assert "margin_of_error" in data
        assert "lower_bound" in data
        assert "upper_bound" in data
        assert "interval" in data

    @pytest.mark.asyncio
    async def test_proportion_confidence_interval(self, symbolic_ai):
        """Test confidence interval for proportion."""
        args = {
            "parameter": "proportion",
            "sample_stats": {
                "p_hat": "0.6",
                "n": "100"
            }
        }
        result = await call_tool(symbolic_ai, "confidence_intervals", args)
        data = json.loads(result[0].text)

        assert "margin_of_error" in data
        assert "interval" in data

    @pytest.mark.asyncio
    async def test_custom_confidence_level(self, symbolic_ai):
        """Test custom confidence level."""
        args = {
            "parameter": "mean",
            "sample_stats": {
                "x_bar": "100",
                "s": "10",
                "n": "30"
            },
            "confidence_level": "0.99"
        }
        result = await call_tool(symbolic_ai, "confidence_intervals", args)
        data = json.loads(result[0].text)

        assert data["confidence_level"] == "0.99"


class TestProbabilityDistributions:
    """Test probability_distributions tool."""

    @pytest.mark.asyncio
    async def test_create_normal(self, symbolic_ai):
        """Test creating normal distribution."""
        args = {
            "operation": "create",
            "distribution_type": "normal",
            "parameters": {"mu": "0", "sigma": "1"}
        }
        result = await call_tool(symbolic_ai, "probability_distributions", args)
        data = json.loads(result[0].text)

        assert data["operation"] == "create"
        assert data["distribution_type"] == "normal"
        assert "mean" in data
        assert "variance" in data

    @pytest.mark.asyncio
    async def test_create_binomial(self, symbolic_ai):
        """Test creating binomial distribution."""
        args = {
            "operation": "create",
            "distribution_type": "binomial",
            "parameters": {"n": 10, "p": "0.5"}
        }
        result = await call_tool(symbolic_ai, "probability_distributions", args)
        data = json.loads(result[0].text)

        assert data["distribution_type"] == "binomial"

    @pytest.mark.asyncio
    async def test_sum_operation(self, symbolic_ai):
        """Test sum of distributions operation."""
        args = {
            "operation": "sum"
        }
        result = await call_tool(symbolic_ai, "probability_distributions", args)
        data = json.loads(result[0].text)

        assert "note" in data

    @pytest.mark.asyncio
    async def test_transform_operation(self, symbolic_ai):
        """Test transformation operation."""
        args = {
            "operation": "transform",
            "transformation": "2*X + 3"
        }
        result = await call_tool(symbolic_ai, "probability_distributions", args)
        data = json.loads(result[0].text)

        assert data["transformation"] == "2*X + 3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
