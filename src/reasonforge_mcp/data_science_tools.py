"""
Data Science Tools for ReasonForge MCP Server

This module provides 8 symbolic data science tools for statistical analysis,
time series modeling, hypothesis testing, and experimental design - all with
symbolic formula generation rather than numerical computation.
"""

import json
from typing import Any
from mcp.types import Tool, TextContent
import sympy as sp
from sympy import symbols, Symbol, IndexedBase, Sum, sqrt, exp, log, pi
from sympy.stats import *


def get_data_science_tool_definitions() -> list[Tool]:
    """Return list of data science tool definitions."""
    return [
        Tool(
            name="symbolic_dataframe",
            description="Create symbolic data structures with variables for statistical analysis. Generate symbolic representations of datasets for formula derivation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variable names (e.g., ['X', 'Y', 'Z'])"
                    },
                    "num_observations": {
                        "type": "string",
                        "description": "Number of observations (symbolic, e.g., 'n' or numeric)"
                    },
                    "operations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Operations to perform: 'mean', 'variance', 'covariance_matrix'"
                    }
                },
                "required": ["variables"]
            }
        ),

        Tool(
            name="statistical_moments_symbolic",
            description="Calculate statistical moments symbolically: mean, variance, skewness, kurtosis, and higher-order moments with exact formulas.",
            inputSchema={
                "type": "object",
                "properties": {
                    "distribution": {
                        "type": "string",
                        "description": "Distribution or variable name"
                    },
                    "moments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": ["mean", "variance", "std", "skewness", "kurtosis", "raw_moment", "central_moment"],
                        "description": "Which moments to calculate"
                    },
                    "moment_order": {
                        "type": "integer",
                        "description": "Order for raw_moment or central_moment"
                    }
                },
                "required": ["distribution", "moments"]
            }
        ),

        Tool(
            name="time_series_symbolic",
            description="Model time series symbolically with ARMA, ARIMA, and autoregressive processes. Generate difference equations and characteristic polynomials.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["AR", "MA", "ARMA", "ARIMA", "random_walk"],
                        "description": "Type of time series model"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Model parameters (e.g., {'p': 2, 'q': 1} for ARMA(2,1))"
                    },
                    "compute": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to compute: 'equation', 'characteristic_poly', 'moments', 'stationarity'"
                    }
                },
                "required": ["model_type"]
            }
        ),

        Tool(
            name="hypothesis_test_symbolic",
            description="Generate symbolic formulas for hypothesis test statistics. Derive test statistic formulas, rejection regions, and power functions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "enum": ["one_sample_t", "two_sample_t", "paired_t", "one_sample_z", "two_sample_z", "proportion_z", "chi_square_goodness", "chi_square_independence"],
                        "description": "Type of hypothesis test"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Test parameters (e.g., {'mu_0': 'mu0', 'alpha': '0.05'})"
                    },
                    "derive": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to derive: 'test_statistic', 'rejection_region', 'p_value_formula', 'power_function'"
                    }
                },
                "required": ["test_type"]
            }
        ),

        Tool(
            name="anova_symbolic",
            description="Derive ANOVA formulas symbolically. Generate sum of squares decomposition, F-statistic formulas, and expected mean squares.",
            inputSchema={
                "type": "object",
                "properties": {
                    "anova_type": {
                        "type": "string",
                        "enum": ["one_way", "two_way", "repeated_measures", "nested"],
                        "description": "Type of ANOVA design"
                    },
                    "num_groups": {
                        "type": "string",
                        "description": "Number of groups (symbolic or numeric)"
                    },
                    "num_observations": {
                        "type": "string",
                        "description": "Observations per group (symbolic or numeric)"
                    },
                    "compute": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to compute: 'SS_between', 'SS_within', 'SS_total', 'F_statistic', 'degrees_of_freedom'"
                    }
                },
                "required": ["anova_type"]
            }
        ),

        Tool(
            name="multivariate_statistics",
            description="Work with multivariate distributions symbolically. Generate formulas for multivariate normal, Wishart, Hotelling's T², and MANOVA.",
            inputSchema={
                "type": "object",
                "properties": {
                    "distribution": {
                        "type": "string",
                        "enum": ["multivariate_normal", "wishart", "multivariate_t", "dirichlet"],
                        "description": "Multivariate distribution"
                    },
                    "dimension": {
                        "type": "string",
                        "description": "Dimension of the distribution (e.g., 'p' or numeric)"
                    },
                    "compute": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to compute: 'pdf', 'mean', 'covariance', 'marginal', 'conditional'"
                    }
                },
                "required": ["distribution"]
            }
        ),

        Tool(
            name="sampling_distributions",
            description="Derive sampling distributions symbolically. Generate distributions of sample mean, sample variance, and other statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "statistic": {
                        "type": "string",
                        "enum": ["sample_mean", "sample_variance", "sample_proportion", "difference_means", "ratio_variances"],
                        "description": "Sample statistic"
                    },
                    "population_distribution": {
                        "type": "string",
                        "description": "Underlying population distribution (e.g., 'normal', 'binomial')"
                    },
                    "sample_size": {
                        "type": "string",
                        "description": "Sample size (symbolic or numeric, e.g., 'n')"
                    },
                    "derive": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to derive: 'distribution', 'mean', 'variance', 'standard_error', 'clt_approximation'"
                    }
                },
                "required": ["statistic"]
            }
        ),

        Tool(
            name="experimental_design",
            description="Generate design matrices and analysis formulas for experimental designs. Supports factorial, blocked, Latin square, and fractional factorial designs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "design_type": {
                        "type": "string",
                        "enum": ["factorial", "randomized_block", "latin_square", "fractional_factorial", "split_plot"],
                        "description": "Type of experimental design"
                    },
                    "factors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "levels": {"type": "integer"}
                            }
                        },
                        "description": "Experimental factors and their levels"
                    },
                    "compute": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What to compute: 'model_formula', 'contrast_matrix', 'degrees_of_freedom', 'expected_ms'"
                    }
                },
                "required": ["design_type"]
            }
        )
    ]


async def handle_data_science_tool(name: str, arguments: dict[str, Any], ai) -> list[TextContent]:
    """Handle data science tool calls."""

    if name == "symbolic_dataframe":
        return await _symbolic_dataframe(arguments, ai)
    elif name == "statistical_moments_symbolic":
        return await _statistical_moments_symbolic(arguments, ai)
    elif name == "time_series_symbolic":
        return await _time_series_symbolic(arguments, ai)
    elif name == "hypothesis_test_symbolic":
        return await _hypothesis_test_symbolic(arguments, ai)
    elif name == "anova_symbolic":
        return await _anova_symbolic(arguments, ai)
    elif name == "multivariate_statistics":
        return await _multivariate_statistics(arguments, ai)
    elif name == "sampling_distributions":
        return await _sampling_distributions(arguments, ai)
    elif name == "experimental_design":
        return await _experimental_design(arguments, ai)
    else:
        raise ValueError(f"Unknown data science tool: {name}")


# Implementation functions

async def _symbolic_dataframe(args: dict, ai) -> list[TextContent]:
    """Create symbolic data structures."""
    variables = args["variables"]
    n = sp.Symbol('n', positive=True, integer=True) if "num_observations" not in args else sp.sympify(args["num_observations"])
    operations = args.get("operations", [])

    result = {
        "variables": variables,
        "num_observations": str(n)
    }

    # Create indexed variables
    indexed_vars = {}
    for var in variables:
        indexed_vars[var] = IndexedBase(var)
        result[f"{var}_notation"] = f"{var}[i] for i = 1, 2, ..., n"

    if "mean" in operations:
        means = {}
        for var in variables:
            X_i = IndexedBase(var)
            i = Symbol('i', integer=True, positive=True)
            mean_formula = Sum(X_i[i], (i, 1, n)) / n
            means[f"mean_{var}"] = str(mean_formula)
            means[f"mean_{var}_latex"] = sp.latex(mean_formula)
        result["means"] = means

    if "variance" in operations:
        variances = {}
        for var in variables:
            X_i = IndexedBase(var)
            i = Symbol('i', integer=True, positive=True)
            mu = Symbol(f'mu_{var}')
            var_formula = Sum((X_i[i] - mu)**2, (i, 1, n)) / (n - 1)
            variances[f"variance_{var}"] = str(var_formula)
            variances[f"variance_{var}_latex"] = sp.latex(var_formula)
        result["variances"] = variances

    if "covariance_matrix" in operations and len(variables) >= 2:
        result["covariance_matrix_formula"] = "Σ[i,j] = (1/(n-1)) * Σₖ(Xᵢ[k] - μᵢ)(Xⱼ[k] - μⱼ)"
        result["covariance_matrix_dimension"] = f"{len(variables)} × {len(variables)}"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _statistical_moments_symbolic(args: dict, ai) -> list[TextContent]:
    """Calculate statistical moments symbolically."""
    distribution = args["distribution"]
    moments = args["moments"]

    result = {
        "distribution": distribution,
        "requested_moments": moments
    }

    # Create symbolic variable
    X = Symbol('X')
    mu = Symbol('mu')  # mean
    sigma = Symbol('sigma', positive=True)  # std dev

    if "mean" in moments:
        result["mean"] = "E[X] = μ"
        result["mean_formula"] = "∫ x·f(x) dx (continuous) or Σ x·P(X=x) (discrete)"
        result["mean_latex"] = r"\mathbb{E}[X] = \mu"

    if "variance" in moments:
        result["variance"] = "Var(X) = E[(X - μ)²] = E[X²] - (E[X])²"
        result["variance_symbol"] = "σ²"
        result["variance_latex"] = r"\text{Var}(X) = \sigma^2 = \mathbb{E}[(X-\mu)^2]"

    if "std" in moments:
        result["standard_deviation"] = "SD(X) = √Var(X) = σ"
        result["std_latex"] = r"\sigma = \sqrt{\text{Var}(X)}"

    if "skewness" in moments:
        result["skewness"] = "E[(X - μ)³] / σ³"
        result["skewness_formula"] = "γ₁ = E[(X - μ)³] / σ³"
        result["skewness_latex"] = r"\gamma_1 = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3}"
        result["interpretation"] = {
            "γ₁ > 0": "Right-skewed (positive skew)",
            "γ₁ < 0": "Left-skewed (negative skew)",
            "γ₁ = 0": "Symmetric"
        }

    if "kurtosis" in moments:
        result["kurtosis"] = "E[(X - μ)⁴] / σ⁴"
        result["kurtosis_formula"] = "γ₂ = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)"
        result["kurtosis_latex"] = r"\gamma_2 = \frac{\mathbb{E}[(X-\mu)^4]}{\sigma^4} - 3"
        result["interpretation"] = {
            "γ₂ > 0": "Heavy tails (leptokurtic)",
            "γ₂ < 0": "Light tails (platykurtic)",
            "γ₂ = 0": "Normal kurtosis (mesokurtic)"
        }

    if "raw_moment" in moments:
        k = args.get("moment_order", 1)
        result[f"raw_moment_{k}"] = f"E[X^{k}]"
        result[f"raw_moment_{k}_formula"] = f"μ'_{k} = E[X^{k}]"

    if "central_moment" in moments:
        k = args.get("moment_order", 2)
        result[f"central_moment_{k}"] = f"E[(X - μ)^{k}]"
        result[f"central_moment_{k}_formula"] = f"μ_{k} = E[(X - μ)^{k}]"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _time_series_symbolic(args: dict, ai) -> list[TextContent]:
    """Model time series symbolically."""
    model_type = args["model_type"]
    parameters = args.get("parameters", {})
    compute = args.get("compute", ["equation"])

    result = {
        "model_type": model_type,
        "parameters": parameters
    }

    t = Symbol('t', integer=True, positive=True)
    Y_t = IndexedBase('Y')
    epsilon_t = IndexedBase('epsilon')

    if model_type == "AR":
        p = parameters.get("p", 1)
        result["model_name"] = f"AR({p}) - Autoregressive Model of Order {p}"

        if "equation" in compute:
            # Y_t = φ₁Y_{t-1} + φ₂Y_{t-2} + ... + φₚY_{t-p} + ε_t
            phi = [Symbol(f'phi_{i}') for i in range(1, p + 1)]
            equation = " + ".join([f"φ_{i}·Y[t-{i}]" for i in range(1, p + 1)]) + " + ε[t]"
            result["equation"] = f"Y[t] = {equation}"
            result["equation_latex"] = f"Y_t = " + " + ".join([f"\\phi_{{{i}}}Y_{{t-{i}}}" for i in range(1, p + 1)]) + " + \\varepsilon_t"

        if "characteristic_poly" in compute:
            result["characteristic_polynomial"] = f"1 - φ₁·B - φ₂·B² - ... - φₚ·Bᵖ = 0"
            result["characteristic_poly_latex"] = r"1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p = 0"
            result["note"] = "B is the backshift operator: B·Y_t = Y_{t-1}"

        if "stationarity" in compute:
            result["stationarity_condition"] = "All roots of characteristic polynomial must lie outside unit circle"
            result["stationarity_formula"] = "|roots| > 1"

    elif model_type == "MA":
        q = parameters.get("q", 1)
        result["model_name"] = f"MA({q}) - Moving Average Model of Order {q}"

        if "equation" in compute:
            equation = " + ".join([f"θ_{i}·ε[t-{i}]" for i in range(1, q + 1)]) + " + ε[t]"
            result["equation"] = f"Y[t] = μ + {equation}"
            result["equation_latex"] = f"Y_t = \\mu + " + " + ".join([f"\\theta_{{{i}}}\\varepsilon_{{t-{i}}}" for i in range(1, q + 1)]) + " + \\varepsilon_t"

    elif model_type == "ARMA":
        p = parameters.get("p", 1)
        q = parameters.get("q", 1)
        result["model_name"] = f"ARMA({p},{q}) - Autoregressive Moving Average"

        if "equation" in compute:
            ar_part = " + ".join([f"φ_{i}·Y[t-{i}]" for i in range(1, p + 1)])
            ma_part = " + ".join([f"θ_{i}·ε[t-{i}]" for i in range(1, q + 1)])
            result["equation"] = f"Y[t] = {ar_part} + {ma_part} + ε[t]"

    elif model_type == "ARIMA":
        p = parameters.get("p", 1)
        d = parameters.get("d", 1)
        q = parameters.get("q", 1)
        result["model_name"] = f"ARIMA({p},{d},{q})"

        if "equation" in compute:
            result["differencing"] = f"Apply {d} differences: ∇ᵈY[t]"
            result["differencing_operator"] = f"∇Y[t] = Y[t] - Y[t-1]"
            result["arma_on_differenced"] = f"Then apply ARMA({p},{q}) on differenced series"

    elif model_type == "random_walk":
        result["model_name"] = "Random Walk"
        result["equation"] = "Y[t] = Y[t-1] + ε[t]"
        result["equation_latex"] = "Y_t = Y_{t-1} + \\varepsilon_t"
        result["properties"] = {
            "non_stationary": "Variance increases with time",
            "mean": "E[Y_t] = Y_0",
            "variance": "Var(Y_t) = t·σ²"
        }

    if "moments" in compute:
        result["mean_formula"] = "E[Y_t] depends on stationarity"
        result["variance_formula"] = "Var(Y_t) = function of autocovariances"
        result["autocorrelation_function"] = "ρ(k) = Cov(Y_t, Y_{t-k}) / Var(Y_t)"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _hypothesis_test_symbolic(args: dict, ai) -> list[TextContent]:
    """Generate hypothesis test formulas."""
    test_type = args["test_type"]
    parameters = args.get("parameters", {})
    derive = args.get("derive", ["test_statistic"])

    result = {
        "test_type": test_type,
        "parameters": parameters
    }

    if test_type == "one_sample_t":
        result["test_name"] = "One-Sample t-Test"
        result["hypotheses"] = {
            "H₀": "μ = μ₀",
            "Hₐ": "μ ≠ μ₀ (two-sided) or μ > μ₀ or μ < μ₀"
        }

        if "test_statistic" in derive:
            result["test_statistic"] = "t = (x̄ - μ₀) / (s / √n)"
            result["test_statistic_latex"] = r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}"
            result["degrees_of_freedom"] = "df = n - 1"

        if "rejection_region" in derive:
            result["rejection_region_two_sided"] = "|t| > t_{α/2, n-1}"
            result["rejection_region_right"] = "t > t_{α, n-1}"
            result["rejection_region_left"] = "t < -t_{α, n-1}"

        if "p_value_formula" in derive:
            result["p_value_two_sided"] = "p = 2·P(T > |t_obs|) where T ~ t_{n-1}"
            result["p_value_right"] = "p = P(T > t_obs)"
            result["p_value_left"] = "p = P(T < t_obs)"

    elif test_type == "two_sample_t":
        result["test_name"] = "Two-Sample t-Test (Independent Samples)"
        result["hypotheses"] = {
            "H₀": "μ₁ = μ₂",
            "Hₐ": "μ₁ ≠ μ₂"
        }

        if "test_statistic" in derive:
            result["test_statistic_pooled"] = "t = (x̄₁ - x̄₂) / (sₚ·√(1/n₁ + 1/n₂))"
            result["pooled_std"] = "sₚ = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]"
            result["degrees_of_freedom"] = "df = n₁ + n₂ - 2"
            result["test_statistic_latex"] = r"t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}"

    elif test_type == "one_sample_z":
        result["test_name"] = "One-Sample z-Test"
        result["assumption"] = "Population σ is known"
        result["hypotheses"] = {
            "H₀": "μ = μ₀",
            "Hₐ": "μ ≠ μ₀"
        }

        if "test_statistic" in derive:
            result["test_statistic"] = "z = (x̄ - μ₀) / (σ / √n)"
            result["test_statistic_latex"] = r"z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}"
            result["distribution"] = "z ~ N(0, 1) under H₀"

    elif test_type == "proportion_z":
        result["test_name"] = "One-Sample z-Test for Proportion"
        result["hypotheses"] = {
            "H₀": "p = p₀",
            "Hₐ": "p ≠ p₀"
        }

        if "test_statistic" in derive:
            result["test_statistic"] = "z = (p̂ - p₀) / √(p₀(1-p₀)/n)"
            result["test_statistic_latex"] = r"z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}"
            result["sample_proportion"] = "p̂ = x/n"

    elif test_type == "chi_square" or test_type == "chi_square_goodness":
        result["test_name"] = "Chi-Square Goodness of Fit Test"
        result["hypotheses"] = {
            "H₀": "Data follows specified distribution",
            "Hₐ": "Data does not follow specified distribution"
        }

        # Always include test_statistic for chi_square
        result["test_statistic"] = "χ² = Σ[(Oᵢ - Eᵢ)² / Eᵢ]"
        result["test_statistic_latex"] = r"\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}"
        result["degrees_of_freedom"] = "df = k - 1 - (number of estimated parameters)"
        result["formula"] = "χ² = Σ[(Oᵢ - Eᵢ)² / Eᵢ]"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _anova_symbolic(args: dict, ai) -> list[TextContent]:
    """Derive ANOVA formulas."""
    anova_type = args["anova_type"]
    k = sp.Symbol('k', positive=True, integer=True)  # number of groups
    n = sp.Symbol('n', positive=True, integer=True)  # observations per group
    compute = args.get("compute", ["F_statistic"])

    result = {
        "anova_type": anova_type,
        "num_groups": str(k),
        "num_observations_per_group": str(n)
    }

    if anova_type == "one_way":
        result["model"] = "Yᵢⱼ = μ + αᵢ + εᵢⱼ"
        result["model_description"] = "i = group (1 to k), j = observation within group (1 to n)"

        if "SS_total" in compute:
            result["SS_total"] = "SST = ΣΣ(Yᵢⱼ - Ȳ··)²"
            result["SS_total_latex"] = r"SST = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{..})^2"
            result["df_total"] = "kn - 1"

        if "SS_between" in compute:
            result["SS_between"] = "SSB = n·Σ(Ȳᵢ· - Ȳ··)²"
            result["SS_between_latex"] = r"SSB = n\sum_{i=1}^{k}(\bar{Y}_{i.} - \bar{Y}_{..})^2"
            result["df_between"] = "k - 1"

        if "SS_within" in compute:
            result["SS_within"] = "SSW = ΣΣ(Yᵢⱼ - Ȳᵢ·)²"
            result["SS_within_latex"] = r"SSW = \sum_{i=1}^{k}\sum_{j=1}^{n}(Y_{ij} - \bar{Y}_{i.})^2"
            result["df_within"] = "k(n - 1) = kn - k"

        if "F_statistic" in compute:
            result["F_statistic"] = "F = MSB / MSW = [SSB/(k-1)] / [SSW/(kn-k)]"
            result["F_statistic_latex"] = r"F = \frac{MSB}{MSW} = \frac{SSB/(k-1)}{SSW/(kn-k)}"
            result["F_distribution"] = "F ~ F_{k-1, kn-k} under H₀"
            result["hypotheses"] = {
                "H₀": "All group means are equal: μ₁ = μ₂ = ... = μₖ",
                "Hₐ": "At least one group mean differs"
            }

        if "degrees_of_freedom" in compute:
            result["degrees_of_freedom"] = {
                "between": "k - 1",
                "within": "k(n - 1)",
                "total": "kn - 1"
            }

        result["partition_formula"] = "SST = SSB + SSW"

    elif anova_type == "two_way":
        result["model"] = "Yᵢⱼₖ = μ + αᵢ + βⱼ + (αβ)ᵢⱼ + εᵢⱼₖ"
        result["factors"] = "Two factors: A (α) and B (β) with interaction"

        if "F_statistic" in compute:
            result["F_statistic_A"] = "F_A = MS_A / MS_E"
            result["F_statistic_B"] = "F_B = MS_B / MS_E"
            result["F_statistic_interaction"] = "F_AB = MS_AB / MS_E"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _multivariate_statistics(args: dict, ai) -> list[TextContent]:
    """Work with multivariate distributions."""
    # Handle both 'distribution' and 'operation' fields
    distribution = args.get("distribution") or args.get("operation")
    p = sp.Symbol('p', positive=True, integer=True)  # dimension
    compute = args.get("compute", ["pdf"])

    result = {
        "dimension": str(p)
    }

    if distribution:
        result["distribution"] = distribution

    if distribution == "multivariate_normal":
        result["distribution_name"] = "Multivariate Normal Distribution"
        result["notation"] = "X ~ N_p(μ, Σ)"

        if "pdf" in compute:
            result["pdf"] = "f(x) = (2π)^(-p/2) |Σ|^(-1/2) exp[-½(x-μ)ᵀΣ⁻¹(x-μ)]"
            result["pdf_latex"] = r"f(\mathbf{x}) = (2\pi)^{-p/2}|\Sigma|^{-1/2}\exp\left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]"

        if "mean" in compute:
            result["mean"] = "E[X] = μ (p-dimensional vector)"
            result["mean_latex"] = r"\mathbb{E}[\mathbf{X}] = \boldsymbol{\mu}"

        if "covariance" in compute:
            result["covariance"] = "Cov(X) = Σ (p×p matrix)"
            result["covariance_latex"] = r"\text{Cov}(\mathbf{X}) = \Sigma"
            result["covariance_element"] = "Σᵢⱼ = Cov(Xᵢ, Xⱼ)"

        if "marginal" in compute:
            result["marginal"] = "Each marginal is univariate normal"
            result["marginal_formula"] = "Xᵢ ~ N(μᵢ, σᵢᵢ)"

        if "conditional" in compute:
            result["conditional"] = "Conditional distributions are also multivariate normal"
            result["conditional_note"] = "X₁|X₂ ~ N(μ₁ + Σ₁₂Σ₂₂⁻¹(x₂-μ₂), Σ₁₁ - Σ₁₂Σ₂₂⁻¹Σ₂₁)"

    elif distribution == "wishart":
        result["distribution_name"] = "Wishart Distribution"
        result["notation"] = "W ~ W_p(n, Σ)"
        result["use_case"] = "Distribution of sample covariance matrix"

        if "pdf" in compute:
            result["pdf_note"] = "Complex matrix-valued PDF involving matrix exponential and determinant"

        if "mean" in compute:
            result["mean"] = "E[W] = n·Σ"
            result["degrees_of_freedom"] = "n (must satisfy n ≥ p)"

    elif distribution == "dirichlet":
        result["distribution_name"] = "Dirichlet Distribution"
        result["notation"] = "X ~ Dir(α₁, α₂, ..., αₖ)"
        result["constraint"] = "Σxᵢ = 1, xᵢ ≥ 0"

        if "pdf" in compute:
            result["pdf"] = "f(x) = [Γ(Σαᵢ) / ∏Γ(αᵢ)] · ∏xᵢ^(αᵢ-1)"
            result["pdf_latex"] = r"f(\mathbf{x}) = \frac{\Gamma(\sum \alpha_i)}{\prod \Gamma(\alpha_i)} \prod x_i^{\alpha_i - 1}"

    elif distribution == "mahalanobis":
        result["statistic_name"] = "Mahalanobis Distance"
        result["formula"] = "D² = (x - μ)ᵀ Σ⁻¹ (x - μ)"
        result["formula_latex"] = r"D^2 = (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})"
        result["distance"] = "D² = (x - μ)ᵀ Σ⁻¹ (x - μ)"
        result["description"] = "Measures distance from x to mean μ accounting for covariance Σ"
        result["properties"] = {
            "scale_invariant": "Accounts for variable scales and correlations",
            "distribution": "D² ~ χ²_p if x ~ N_p(μ, Σ)"
        }

    elif distribution == "hotelling_t_squared":
        result["statistic_name"] = "Hotelling's T² Statistic"
        result["formula"] = "T² = n(x̄ - μ)ᵀ S⁻¹ (x̄ - μ)"
        result["formula_latex"] = r"T^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu})^T S^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu})"
        result["t_squared"] = "T² = n(x̄ - μ)ᵀ S⁻¹ (x̄ - μ)"
        result["description"] = "Multivariate generalization of Student's t-test"
        result["distribution"] = "[(n-p)/(p(n-1))]·T² ~ F_{p, n-p}"
        result["use_case"] = "Test H₀: μ = μ₀ for multivariate normal samples"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _sampling_distributions(args: dict, ai) -> list[TextContent]:
    """Derive sampling distributions."""
    statistic = args["statistic"]
    # Map 'mean' to 'sample_mean' for compatibility
    if statistic == "mean":
        statistic = "sample_mean"

    population_dist = args.get("population_distribution", "normal")
    n = sp.Symbol('n', positive=True, integer=True)

    result = {
        "statistic": statistic,
        "population_distribution": population_dist,
        "sample_size": str(n)
    }

    if statistic == "sample_mean":
        result["statistic_name"] = "Sample Mean: X̄"
        result["statistic_formula"] = "X̄ = (1/n)·Σxᵢ"
        result["formula"] = "X̄ = (1/n)·Σxᵢ"

        if population_dist == "normal":
            result["distribution"] = "X̄ ~ N(μ, σ²/n)"
            result["distribution_latex"] = r"\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)"
            result["exact"] = "Exact for any sample size n"

        result["mean"] = "E[X̄] = μ"
        result["variance"] = "Var(X̄) = σ²/n"
        result["standard_error"] = "SE(X̄) = σ/√n"
        result["standard_error_latex"] = r"SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}"

        result["clt"] = "For any distribution with finite variance, X̄ → N(μ, σ²/n) as n → ∞"

    elif statistic == "sample_variance":
        result["statistic_name"] = "Sample Variance: S²"
        result["statistic_formula"] = "S² = (1/(n-1))·Σ(xᵢ - X̄)²"

        if population_dist == "normal":
            result["distribution"] = "(n-1)S²/σ² ~ χ²_{n-1}"
            result["distribution_latex"] = r"\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}"

        result["mean"] = "E[S²] = σ²"
        result["variance"] = "Var(S²) = 2σ⁴/(n-1)"
        result["unbiased"] = "S² is unbiased estimator of σ²"

    elif statistic == "sample_proportion":
        result["statistic_name"] = "Sample Proportion: p̂"
        result["statistic_formula"] = "p̂ = X/n where X ~ Binomial(n, p)"

        result["mean"] = "E[p̂] = p"
        result["variance"] = "Var(p̂) = p(1-p)/n"
        result["standard_error"] = "SE(p̂) = √[p(1-p)/n]"

        result["clt_approximation"] = "p̂ ≈ N(p, p(1-p)/n) for large n"
        result["rule_of_thumb"] = "np ≥ 10 and n(1-p) ≥ 10"

    elif statistic == "difference_means":
        result["statistic_name"] = "Difference of Sample Means: X̄₁ - X̄₂"

        if population_dist == "normal":
            result["distribution"] = "(X̄₁ - X̄₂) ~ N(μ₁ - μ₂, σ₁²/n₁ + σ₂²/n₂)"
            result["distribution_latex"] = r"\bar{X}_1 - \bar{X}_2 \sim N\left(\mu_1 - \mu_2, \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}\right)"

        result["mean"] = "E[X̄₁ - X̄₂] = μ₁ - μ₂"
        result["variance"] = "Var(X̄₁ - X̄₂) = σ₁²/n₁ + σ₂²/n₂"
        result["standard_error"] = "SE = √(σ₁²/n₁ + σ₂²/n₂)"

    elif statistic == "ratio_variances":
        result["statistic_name"] = "Ratio of Sample Variances: S₁²/S₂²"

        if population_dist == "normal":
            result["distribution"] = "(S₁²/σ₁²) / (S₂²/σ₂²) ~ F_{n₁-1, n₂-1}"
            result["distribution_latex"] = r"\frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2} \sim F_{n_1-1, n_2-1}"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _experimental_design(args: dict, ai) -> list[TextContent]:
    """Generate experimental design formulas."""
    design_type = args["design_type"]
    factors = args.get("factors", [])
    compute = args.get("compute", ["model_formula"])

    result = {
        "design_type": design_type,
        "factors": factors
    }

    if design_type == "factorial":
        result["design_name"] = "Factorial Design"

        # Handle factors as either list of strings or list of dicts
        if len(factors) > 0 and isinstance(factors[0], str):
            # Factors are strings like ['A', 'B', 'C']
            # Calculate run count assuming 2 levels per factor
            factor_count = len(factors)
            run_count = 2 ** factor_count
            result["run_count"] = run_count
            result["design"] = f"2^{factor_count} Factorial"
            result["factor_names"] = factors

            if "model_formula" in compute:
                result["model"] = "Y = μ + main effects + interactions + ε"
                result["full_factorial_runs"] = run_count

        elif len(factors) == 2:
            a_levels = factors[0].get("levels", 2)
            b_levels = factors[1].get("levels", 2)
            result["design"] = f"{a_levels} × {b_levels} Factorial"
            result["run_count"] = a_levels * b_levels

            if "model_formula" in compute:
                result["model"] = "Yᵢⱼₖ = μ + αᵢ + βⱼ + (αβ)ᵢⱼ + εᵢⱼₖ"
                result["model_latex"] = r"Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}"
                result["terms"] = {
                    "μ": "Overall mean",
                    "αᵢ": "Main effect of factor A",
                    "βⱼ": "Main effect of factor B",
                    "(αβ)ᵢⱼ": "Interaction effect",
                    "εᵢⱼₖ": "Random error"
                }

            if "degrees_of_freedom" in compute:
                result["df"] = {
                    "factor_A": f"{a_levels - 1}",
                    "factor_B": f"{b_levels - 1}",
                    "interaction": f"{(a_levels - 1) * (b_levels - 1)}",
                    "total": f"{a_levels * b_levels * 'n' - 1}"
                }

    elif design_type == "randomized_block":
        result["design_name"] = "Randomized Complete Block Design (RCBD)"
        result["purpose"] = "Control for known source of variability (blocking)"

        if "model_formula" in compute:
            result["model"] = "Yᵢⱼ = μ + τᵢ + βⱼ + εᵢⱼ"
            result["model_latex"] = r"Y_{ij} = \mu + \tau_i + \beta_j + \varepsilon_{ij}"
            result["terms"] = {
                "τᵢ": "Treatment effect",
                "βⱼ": "Block effect",
                "εᵢⱼ": "Random error"
            }

    elif design_type == "latin_square":
        result["design_name"] = "Latin Square Design"
        result["purpose"] = "Control for two sources of variability"

        if "model_formula" in compute:
            result["model"] = "Yᵢⱼₖ = μ + αᵢ + βⱼ + γₖ + εᵢⱼₖ"
            result["model_description"] = "i = row, j = column, k = treatment"
            result["constraint"] = "Each treatment appears once in each row and column"

    elif design_type == "fractional_factorial":
        result["design_name"] = "Fractional Factorial Design"
        result["purpose"] = "Reduce number of runs by assuming high-order interactions negligible"

        if len(factors) >= 3:
            result["notation"] = "2^(k-p) design"
            result["explanation"] = {
                "k": "Number of factors",
                "p": "Fraction (1/2^p of full factorial)"
            }
            result["confounding"] = "Some effects are confounded (aliased) with others"

    if "contrast_matrix" in compute:
        result["contrast_matrix_note"] = "Matrix C such that contrasts are C·β"
        result["example_contrast"] = "Compare treatment 1 vs treatment 2: [1, -1, 0, ...]"

    if "expected_ms" in compute:
        result["expected_mean_squares"] = "E[MS] formulas depend on fixed vs random effects"
        result["fixed_effects"] = "E[MS_treatment] = σ² + n·Σαᵢ²/(a-1)"
        result["random_effects"] = "E[MS_treatment] = σ² + n·σ_α²"

    result["status"] = "success"
    return [TextContent(type="text", text=json.dumps(result, indent=2))]
