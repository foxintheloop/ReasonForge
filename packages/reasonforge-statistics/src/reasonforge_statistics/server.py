"""ReasonForge Statistics - Probability and Data Science

16 tools for statistics, probability, and data analysis.
"""

from typing import Dict, Any, List

import sympy as sp
from sympy import symbols, latex, simplify, Matrix
from sympy.stats import (
    Normal, Uniform, Binomial, Poisson, Exponential,
    E, variance, std, density
)

from reasonforge import (
    BaseReasonForgeServer,
    SymbolicAI,
    create_input_schema,
    safe_sympify,
    validate_variable_name,
    validate_expression_string,
    validate_list_input,
    ValidationError,
)


class StatisticsServer(BaseReasonForgeServer):
    """MCP server for statistical analysis and probability calculations."""

    def __init__(self):
        super().__init__("reasonforge-statistics")
        self.ai = SymbolicAI()

    def register_tools(self):
        """Register all statistical analysis tools."""

        self.add_tool(
            name="calculate_probability",
            description="Calculate probability for distributions.",
            handler=self.handle_calculate_probability,
            input_schema=create_input_schema(
                properties={
                    "distribution": {
                        "type": "string",
                        "description": "Distribution type (Normal, Uniform, Binomial, Poisson, Exponential)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Distribution parameters"
                    },
                    "operation": {
                        "type": "string",
                        "description": "Operation to perform (expectation, variance, std)"
                    }
                },
                required=["distribution"]
            )
        )

        self.add_tool(
            name="bayesian_inference",
            description="Apply Bayes' theorem.",
            handler=self.handle_bayesian_inference,
            input_schema=create_input_schema(
                properties={
                    "prior": {
                        "type": "string",
                        "description": "Prior probability P(H)"
                    },
                    "likelihood": {
                        "type": "string",
                        "description": "Likelihood P(E|H)"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Evidence P(E)"
                    }
                },
                required=["prior", "likelihood"]
            )
        )

        self.add_tool(
            name="statistical_test",
            description="Set up statistical tests.",
            handler=self.handle_statistical_test,
            input_schema=create_input_schema(
                properties={
                    "test_type": {
                        "type": "string",
                        "description": "Test type (t-test, z-test, chi-square)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Test parameters"
                    }
                },
                required=["test_type"]
            )
        )

        self.add_tool(
            name="distribution_properties",
            description="Get distribution properties.",
            handler=self.handle_distribution_properties,
            input_schema=create_input_schema(
                properties={
                    "distribution": {
                        "type": "string",
                        "description": "Distribution type"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Distribution parameters"
                    }
                },
                required=["distribution"]
            )
        )

        self.add_tool(
            name="correlation_analysis",
            description="Analyze correlation.",
            handler=self.handle_correlation_analysis,
            input_schema=create_input_schema(
                properties={
                    "x_distribution": {
                        "type": "string",
                        "description": "X distribution"
                    },
                    "y_distribution": {
                        "type": "string",
                        "description": "Y distribution"
                    }
                },
                required=["x_distribution", "y_distribution"]
            )
        )

        self.add_tool(
            name="regression_symbolic",
            description="Symbolic regression analysis.",
            handler=self.handle_regression_symbolic,
            input_schema=create_input_schema(
                properties={
                    "regression_type": {
                        "type": "string",
                        "description": "Regression type (linear, multiple, polynomial)"
                    },
                    "variables": {
                        "type": "array",
                        "description": "Variable names"
                    }
                },
                required=["regression_type"]
            )
        )

        self.add_tool(
            name="confidence_intervals",
            description="Calculate confidence intervals.",
            handler=self.handle_confidence_intervals,
            input_schema=create_input_schema(
                properties={
                    "parameter_type": {
                        "type": "string",
                        "description": "Parameter type (mean, proportion)"
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "Confidence level (e.g., 0.95)"
                    }
                },
                required=["parameter_type"]
            )
        )

        self.add_tool(
            name="probability_distributions",
            description="Work with probability distributions.",
            handler=self.handle_probability_distributions,
            input_schema=create_input_schema(
                properties={
                    "operation": {
                        "type": "string",
                        "description": "Operation (convolution, mixture)"
                    },
                    "distributions": {
                        "type": "array",
                        "description": "List of distributions"
                    }
                },
                required=["operation"]
            )
        )

        self.add_tool(
            name="symbolic_dataframe",
            description="Create symbolic dataframe.",
            handler=self.handle_symbolic_dataframe,
            input_schema=create_input_schema(
                properties={
                    "columns": {
                        "type": "array",
                        "description": "Column names"
                    }
                },
                required=["columns"]
            )
        )

        self.add_tool(
            name="statistical_moments_symbolic",
            description="Calculate statistical moments.",
            handler=self.handle_statistical_moments_symbolic,
            input_schema=create_input_schema(
                properties={
                    "data_symbol": {
                        "type": "string",
                        "description": "Data symbol name"
                    },
                    "order": {
                        "type": "integer",
                        "description": "Moment order (1-4)"
                    }
                },
                required=["data_symbol", "order"]
            )
        )

        self.add_tool(
            name="time_series_symbolic",
            description="Symbolic time series analysis.",
            handler=self.handle_time_series_symbolic,
            input_schema=create_input_schema(
                properties={
                    "model_type": {
                        "type": "string",
                        "description": "Model type (AR, MA, ARMA)"
                    },
                    "p": {
                        "type": "integer",
                        "description": "AR order"
                    },
                    "q": {
                        "type": "integer",
                        "description": "MA order"
                    }
                },
                required=["model_type"]
            )
        )

        self.add_tool(
            name="hypothesis_test_symbolic",
            description="Symbolic hypothesis testing.",
            handler=self.handle_hypothesis_test_symbolic,
            input_schema=create_input_schema(
                properties={
                    "test_type": {
                        "type": "string",
                        "description": "Test type"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Test parameters"
                    }
                },
                required=["test_type"]
            )
        )

        self.add_tool(
            name="anova_symbolic",
            description="Symbolic ANOVA.",
            handler=self.handle_anova_symbolic,
            input_schema=create_input_schema(
                properties={
                    "anova_type": {
                        "type": "string",
                        "description": "ANOVA type (one-way, two-way)"
                    },
                    "groups": {
                        "type": "integer",
                        "description": "Number of groups"
                    }
                },
                required=["anova_type"]
            )
        )

        self.add_tool(
            name="multivariate_statistics",
            description="Multivariate statistical analysis.",
            handler=self.handle_multivariate_statistics,
            input_schema=create_input_schema(
                properties={
                    "operation": {
                        "type": "string",
                        "description": "Operation (covariance_matrix, mahalanobis_distance, principal_components)"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Operation parameters"
                    }
                },
                required=["operation"]
            )
        )

        self.add_tool(
            name="sampling_distributions",
            description="Analyze sampling distributions.",
            handler=self.handle_sampling_distributions,
            input_schema=create_input_schema(
                properties={
                    "statistic": {
                        "type": "string",
                        "description": "Statistic (sample_mean, sample_proportion, sample_variance)"
                    },
                    "population_parameters": {
                        "type": "object",
                        "description": "Population parameters"
                    }
                },
                required=["statistic"]
            )
        )

        self.add_tool(
            name="experimental_design",
            description="Design experiments.",
            handler=self.handle_experimental_design,
            input_schema=create_input_schema(
                properties={
                    "design_type": {
                        "type": "string",
                        "description": "Design type (factorial, randomized_block, latin_square)"
                    },
                    "factors": {
                        "type": "array",
                        "description": "Factor names"
                    }
                },
                required=["design_type"]
            )
        )

    def handle_calculate_probability(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculate_probability."""
        distribution = arguments["distribution"]
        parameters = arguments.get("parameters", {})
        operation = arguments.get("operation", "expectation")

        # Create the distribution
        if distribution == "Normal":
            mean_str = parameters.get("mean", "0")
            std_str = parameters.get("std", "1")
            mean_val = validate_expression_string(mean_str)
            std_val = validate_expression_string(std_str)

            mu = safe_sympify(mean_val)
            sigma = safe_sympify(std_val)
            X = Normal('X', mu, sigma)

        elif distribution == "Uniform":
            a_str = parameters.get("a", "0")
            b_str = parameters.get("b", "1")
            a_val = validate_expression_string(a_str)
            b_val = validate_expression_string(b_str)

            a = safe_sympify(a_val)
            b = safe_sympify(b_val)
            X = Uniform('X', a, b)

        elif distribution == "Binomial":
            n = int(parameters.get("n", 10))
            p_str = parameters.get("p", "0.5")
            p_val = validate_expression_string(p_str)
            p = safe_sympify(p_val)
            X = Binomial('X', n, p)

        elif distribution == "Poisson":
            lambda_str = parameters.get("lambda", "1")
            lambda_val = validate_expression_string(lambda_str)
            lambd = safe_sympify(lambda_val)
            X = Poisson('X', lambd)

        elif distribution == "Exponential":
            rate_str = parameters.get("rate", "1")
            rate_val = validate_expression_string(rate_str)
            rate = safe_sympify(rate_val)
            X = Exponential('X', rate)

        else:
            raise ValidationError(f"Unknown distribution: {distribution}")

        # Perform operation
        if operation == "expectation":
            result_value = E(X)
        elif operation == "variance":
            result_value = variance(X)
        elif operation == "std":
            result_value = std(X)
        else:
            raise ValidationError(f"Unknown operation: {operation}")

        return {
            "distribution": distribution,
            "parameters": parameters,
            "operation": operation,
            "result": str(result_value),
            "latex": latex(result_value)
        }

    def handle_bayesian_inference(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bayesian_inference."""
        prior_str = validate_expression_string(arguments["prior"])
        likelihood_str = validate_expression_string(arguments["likelihood"])
        evidence_str = arguments.get("evidence")

        # Parse probabilities
        prior = safe_sympify(prior_str)
        likelihood = safe_sympify(likelihood_str)

        if evidence_str:
            evidence_val = validate_expression_string(evidence_str)
            evidence = safe_sympify(evidence_val)
            # Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
            posterior = (likelihood * prior) / evidence
        else:
            # Without evidence, assume evidence = likelihood * prior (simplified)
            posterior = likelihood * prior

        posterior_simplified = simplify(posterior)

        return {
            "prior": prior_str,
            "likelihood": likelihood_str,
            "evidence": evidence_str if evidence_str else "not provided",
            "posterior": str(posterior_simplified),
            "latex": latex(posterior_simplified),
            "formula": "P(H|E) = P(E|H) * P(H) / P(E)"
        }

    def handle_statistical_test(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistical_test."""
        test_type = arguments["test_type"]
        parameters = arguments.get("parameters", {})

        if test_type == "t-test":
            # Symbolic t-test formula
            x_bar = symbols('x_bar', real=True)
            mu_0 = symbols('mu_0', real=True)
            s = symbols('s', positive=True)
            n = symbols('n', positive=True)

            t_statistic = (x_bar - mu_0) / (s / sp.sqrt(n))

            return {
                "test_type": test_type,
                "t_statistic_formula": str(t_statistic),
                "latex": latex(t_statistic),
                "description": "t = (x̄ - μ₀) / (s / √n)",
                "result": str(t_statistic)
            }

        elif test_type == "z-test":
            x_bar = symbols('x_bar', real=True)
            mu_0 = symbols('mu_0', real=True)
            sigma = symbols('sigma', positive=True)
            n = symbols('n', positive=True)

            z_statistic = (x_bar - mu_0) / (sigma / sp.sqrt(n))

            return {
                "test_type": test_type,
                "z_statistic_formula": str(z_statistic),
                "latex": latex(z_statistic),
                "description": "z = (x̄ - μ₀) / (σ / √n)",
                "result": str(z_statistic)
            }

        elif test_type == "chi-square":
            O = symbols('O', real=True)  # Observed
            E_expected = symbols('E', positive=True)  # Expected

            chi_square = (O - E_expected)**2 / E_expected

            return {
                "test_type": test_type,
                "chi_square_formula": str(chi_square),
                "latex": latex(chi_square),
                "description": "χ² = Σ(O - E)² / E",
                "result": str(chi_square)
            }

        else:
            raise ValidationError(f"Unknown test type: {test_type}")

    def handle_distribution_properties(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle distribution_properties."""
        distribution = arguments["distribution"]
        parameters = arguments.get("parameters", {})

        # Create the distribution
        if distribution == "Normal":
            mean_str = parameters.get("mean", "mu")
            std_str = parameters.get("std", "sigma")
            mean_val = validate_expression_string(mean_str) if mean_str != "mu" else "mu"
            std_val = validate_expression_string(std_str) if std_str != "sigma" else "sigma"

            mu = safe_sympify(mean_val)
            sigma = safe_sympify(std_val)
            X = Normal('X', mu, sigma)

            return {
                "distribution": distribution,
                "parameters": {"mean": str(mu), "std": str(sigma)},
                "expectation": str(E(X)),
                "variance": str(variance(X)),
                "std_deviation": str(std(X)),
                "pdf": str(density(X)(symbols('x')))
            }

        elif distribution == "Uniform":
            a_str = parameters.get("a", "a")
            b_str = parameters.get("b", "b")
            a_val = validate_expression_string(a_str) if a_str != "a" else "a"
            b_val = validate_expression_string(b_str) if b_str != "b" else "b"

            a = safe_sympify(a_val)
            b = safe_sympify(b_val)
            X = Uniform('X', a, b)

            return {
                "distribution": distribution,
                "parameters": {"a": str(a), "b": str(b)},
                "expectation": str(E(X)),
                "variance": str(variance(X)),
                "pdf": str(density(X)(symbols('x')))
            }

        elif distribution == "Exponential":
            rate_str = parameters.get("rate", "lambda")
            rate_val = validate_expression_string(rate_str) if rate_str != "lambda" else "lambda"
            rate = safe_sympify(rate_val)
            X = Exponential('X', rate)

            return {
                "distribution": distribution,
                "parameters": {"rate": str(rate)},
                "expectation": str(E(X)),
                "variance": str(variance(X)),
                "pdf": str(density(X)(symbols('x')))
            }

        else:
            raise ValidationError(f"Unknown distribution: {distribution}")

    def handle_correlation_analysis(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle correlation_analysis."""
        x_dist = arguments["x_distribution"]
        y_dist = arguments["y_distribution"]

        # Symbolic correlation formula
        rho = symbols('rho', real=True)
        n = symbols('n', positive=True)

        # t-statistic for testing correlation
        t_stat = rho * sp.sqrt(n - 2) / sp.sqrt(1 - rho**2)

        return {
            "x_distribution": x_dist,
            "y_distribution": y_dist,
            "correlation_coefficient": "ρ (rho)",
            "t_statistic": str(t_stat),
            "latex": latex(t_stat),
            "description": "t = ρ√(n-2) / √(1-ρ²)",
            "result": str(t_stat)
        }

    def handle_regression_symbolic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle regression_symbolic."""
        regression_type = arguments["regression_type"]
        variables = arguments.get("variables", ["x", "y"])

        # Validate variable names
        if isinstance(variables, list):
            variables = [validate_variable_name(v) for v in variables]

        if regression_type == "linear":
            # Simple linear regression: y = β₀ + β₁x
            beta_0, beta_1 = symbols('beta_0 beta_1', real=True)
            x = symbols(variables[0] if len(variables) > 0 else 'x')

            regression_eq = beta_0 + beta_1 * x

            return {
                "regression_type": regression_type,
                "equation": str(regression_eq),
                "latex": latex(regression_eq),
                "parameters": ["beta_0 (intercept)", "beta_1 (slope)"],
                "result": str(regression_eq)
            }

        elif regression_type == "multiple":
            # Multiple linear regression
            beta_0 = symbols('beta_0', real=True)
            betas = symbols(f'beta_1:{len(variables)}', real=True)
            x_vars = symbols(' '.join(variables), real=True)

            if isinstance(betas, tuple):
                regression_eq = beta_0 + sum(b * x for b, x in zip(betas, x_vars if isinstance(x_vars, tuple) else [x_vars]))
            else:
                regression_eq = beta_0 + betas * x_vars

            return {
                "regression_type": regression_type,
                "equation": str(regression_eq),
                "latex": latex(regression_eq),
                "variables": variables,
                "result": str(regression_eq)
            }

        elif regression_type == "polynomial":
            beta_0, beta_1, beta_2 = symbols('beta_0 beta_1 beta_2', real=True)
            x = symbols(variables[0] if len(variables) > 0 else 'x')

            regression_eq = beta_0 + beta_1 * x + beta_2 * x**2

            return {
                "regression_type": regression_type,
                "equation": str(regression_eq),
                "latex": latex(regression_eq),
                "degree": 2,
                "result": str(regression_eq)
            }

        else:
            raise ValidationError(f"Unknown regression type: {regression_type}")

    def handle_confidence_intervals(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle confidence_intervals."""
        parameter_type = arguments["parameter_type"]
        confidence_level = arguments.get("confidence_level", 0.95)

        # Symbolic CI formulas
        if parameter_type == "mean":
            x_bar = symbols('x_bar', real=True)
            s = symbols('s', positive=True)
            n = symbols('n', positive=True)
            t_alpha = symbols('t_alpha', positive=True)

            margin_of_error = t_alpha * s / sp.sqrt(n)
            lower_bound = x_bar - margin_of_error
            upper_bound = x_bar + margin_of_error

            return {
                "parameter_type": parameter_type,
                "confidence_level": confidence_level,
                "lower_bound": str(lower_bound),
                "upper_bound": str(upper_bound),
                "margin_of_error": str(margin_of_error),
                "latex": latex(margin_of_error),
                "result": str(margin_of_error)
            }

        elif parameter_type == "proportion":
            p_hat = symbols('p_hat', real=True)
            n = symbols('n', positive=True)
            z_alpha = symbols('z_alpha', positive=True)

            margin_of_error = z_alpha * sp.sqrt(p_hat * (1 - p_hat) / n)
            lower_bound = p_hat - margin_of_error
            upper_bound = p_hat + margin_of_error

            return {
                "parameter_type": parameter_type,
                "confidence_level": confidence_level,
                "lower_bound": str(lower_bound),
                "upper_bound": str(upper_bound),
                "margin_of_error": str(margin_of_error),
                "latex": latex(margin_of_error),
                "result": str(margin_of_error)
            }

        else:
            raise ValidationError(f"Unknown parameter type: {parameter_type}")

    def handle_probability_distributions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle probability_distributions."""
        operation = arguments["operation"]
        distributions = arguments.get("distributions", [])

        if operation == "convolution":
            # Symbolic convolution of distributions
            return {
                "operation": operation,
                "distributions": distributions,
                "note": "Convolution of distributions represents the sum of independent random variables",
                "example": "If X ~ N(μ₁, σ₁²) and Y ~ N(μ₂, σ₂²), then X+Y ~ N(μ₁+μ₂, σ₁²+σ₂²)"
            }

        elif operation == "mixture":
            # Mixture distribution
            weights = symbols(f'w_1:{len(distributions)+1}', real=True)
            return {
                "operation": operation,
                "distributions": distributions,
                "weights": [str(w) for w in (weights if isinstance(weights, tuple) else [weights])],
                "note": "Mixture distribution: f(x) = Σ wᵢfᵢ(x)"
            }

        else:
            raise ValidationError(f"Unknown operation: {operation}")

    def handle_symbolic_dataframe(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle symbolic_dataframe."""
        columns = validate_list_input(arguments["columns"])

        # Validate column names
        columns = [validate_variable_name(col) for col in columns]

        # Create symbolic variables for each column
        dataframe_vars = {}
        for col in columns:
            dataframe_vars[col] = symbols(f'{col}_1 {col}_2 {col}_3', real=True)

        return {
            "columns": columns,
            "symbolic_variables": {col: [str(v) for v in vars] for col, vars in dataframe_vars.items()},
            "note": "Symbolic dataframe created with sample variables. Use these in statistical formulas."
        }

    def handle_statistical_moments_symbolic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistical_moments_symbolic."""
        data_symbol = validate_variable_name(arguments["data_symbol"])
        order = arguments.get("order", 1)

        # Create symbolic moment formulas
        X = symbols(data_symbol, real=True)
        mu = symbols('mu', real=True)
        n = symbols('n', positive=True)

        if order == 1:
            # First moment (mean)
            moment = symbols('sum_x') / n
            name = "Mean (1st moment)"

        elif order == 2:
            # Second central moment (variance)
            moment = symbols('sum_x_minus_mu_squared') / n
            name = "Variance (2nd central moment)"

        elif order == 3:
            # Third standardized moment (skewness)
            sigma = symbols('sigma', positive=True)
            moment = symbols('sum_x_minus_mu_cubed') / (n * sigma**3)
            name = "Skewness (3rd standardized moment)"

        elif order == 4:
            # Fourth standardized moment (kurtosis)
            sigma = symbols('sigma', positive=True)
            moment = symbols('sum_x_minus_mu_fourth') / (n * sigma**4)
            name = "Kurtosis (4th standardized moment)"

        else:
            raise ValidationError(f"Order {order} not implemented. Supported: 1-4")

        return {
            "data_symbol": data_symbol,
            "order": order,
            "moment_name": name,
            "formula": str(moment),
            "latex": latex(moment),
            "result": str(moment)
        }

    def handle_time_series_symbolic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle time_series_symbolic."""
        model_type = arguments["model_type"]
        p = arguments.get("p", 1)
        q = arguments.get("q", 0)

        if model_type == "AR":
            # Autoregressive model AR(p)
            y_t = symbols('y_t', real=True)
            phi = symbols(f'phi_1:{p+1}', real=True)
            epsilon = symbols('epsilon', real=True)

            if isinstance(phi, tuple):
                ar_terms = [phi[i] * symbols(f'y_{{t-{i+1}}}') for i in range(p)]
            else:
                ar_terms = [phi * symbols('y_{t-1}')]

            model = sum(ar_terms) + epsilon

            return {
                "model_type": model_type,
                "order": f"AR({p})",
                "equation": str(model),
                "latex": latex(model),
                "result": str(model)
            }

        elif model_type == "MA":
            # Moving average model MA(q)
            epsilon_t = symbols('epsilon_t', real=True)
            theta = symbols(f'theta_1:{q+1}', real=True)

            if isinstance(theta, tuple):
                ma_terms = [theta[i] * symbols(f'epsilon_{{t-{i+1}}}') for i in range(q)]
            else:
                ma_terms = [theta * symbols('epsilon_{t-1}')]

            model = epsilon_t + sum(ma_terms)

            return {
                "model_type": model_type,
                "order": f"MA({q})",
                "equation": str(model),
                "latex": latex(model),
                "result": str(model)
            }

        elif model_type == "ARMA":
            # ARMA(p, q) model
            return {
                "model_type": model_type,
                "order": f"ARMA({p},{q})",
                "note": "ARMA combines AR and MA components",
                "formula": "y_t = Σφᵢy_{t-i} + Σθⱼε_{t-j} + ε_t"
            }

        else:
            raise ValidationError(f"Unknown model type: {model_type}")

    def handle_hypothesis_test_symbolic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hypothesis_test_symbolic."""
        test_type = arguments["test_type"]
        parameters = arguments.get("parameters", {})

        # Define symbolic hypothesis test
        alpha = symbols('alpha', positive=True)
        H_0 = parameters.get("null_hypothesis", "H₀: μ = μ₀")
        H_1 = parameters.get("alternative_hypothesis", "H₁: μ ≠ μ₀")

        return {
            "test_type": test_type,
            "null_hypothesis": H_0,
            "alternative_hypothesis": H_1,
            "significance_level": "α",
            "decision_rule": f"Reject H₀ if p-value < α or test statistic in critical region",
            "parameters": parameters
        }

    def handle_anova_symbolic(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anova_symbolic."""
        anova_type = arguments["anova_type"]
        groups = arguments.get("groups", 3)

        if anova_type == "one-way":
            # One-way ANOVA F-statistic
            SS_between = symbols('SS_between', positive=True)
            SS_within = symbols('SS_within', positive=True)
            k = symbols('k', positive=True)  # number of groups
            n = symbols('n', positive=True)  # total sample size

            MS_between = SS_between / (k - 1)
            MS_within = SS_within / (n - k)
            F_statistic = MS_between / MS_within

            return {
                "anova_type": anova_type,
                "groups": groups,
                "F_statistic": str(F_statistic),
                "latex": latex(F_statistic),
                "formula": "F = MS_between / MS_within",
                "result": str(F_statistic)
            }

        elif anova_type == "two-way":
            return {
                "anova_type": anova_type,
                "note": "Two-way ANOVA tests effects of two factors and their interaction",
                "formula": "F = MS_factor / MS_error (calculated for each factor and interaction)"
            }

        else:
            raise ValidationError(f"Unknown ANOVA type: {anova_type}")

    def handle_multivariate_statistics(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multivariate_statistics."""
        operation = arguments["operation"]
        parameters = arguments.get("parameters", {})

        if operation == "covariance_matrix":
            # Symbolic covariance matrix
            n_vars = parameters.get("n_variables", 2)
            cov_symbols = [[symbols(f'cov_{i}_{j}', real=True) if i != j else symbols(f'var_{i}', positive=True)
                           for j in range(n_vars)] for i in range(n_vars)]

            cov_matrix = Matrix(cov_symbols)

            return {
                "operation": operation,
                "n_variables": n_vars,
                "covariance_matrix": str(cov_matrix),
                "latex": latex(cov_matrix),
                "result": str(cov_matrix)
            }

        elif operation == "mahalanobis_distance":
            # Mahalanobis distance formula
            x = symbols('x', real=True)
            mu = symbols('mu', real=True)
            Sigma_inv = symbols('Sigma_inv', real=True)

            # D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
            return {
                "operation": operation,
                "formula": "D² = (x - μ)ᵀ Σ⁻¹ (x - μ)",
                "description": "Mahalanobis distance measures distance accounting for covariance"
            }

        elif operation == "principal_components":
            return {
                "operation": operation,
                "note": "PCA finds orthogonal directions of maximum variance",
                "steps": [
                    "1. Compute covariance matrix Σ",
                    "2. Find eigenvalues and eigenvectors of Σ",
                    "3. Sort eigenvectors by eigenvalue (largest first)",
                    "4. Principal components are the top eigenvectors"
                ]
            }

        else:
            raise ValidationError(f"Unknown operation: {operation}")

    def handle_sampling_distributions(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sampling_distributions."""
        statistic = arguments["statistic"]
        population_parameters = arguments.get("population_parameters", {})

        if statistic == "sample_mean":
            # Sampling distribution of sample mean
            mu = symbols('mu', real=True)
            sigma = symbols('sigma', positive=True)
            n = symbols('n', positive=True)

            return {
                "statistic": statistic,
                "distribution": "Normal (by Central Limit Theorem)",
                "mean": str(mu),
                "standard_error": str(sigma / sp.sqrt(n)),
                "latex": latex(sigma / sp.sqrt(n)),
                "note": "x̄ ~ N(μ, σ²/n) for large n",
                "result": str(sigma / sp.sqrt(n))
            }

        elif statistic == "sample_proportion":
            p = symbols('p', real=True)
            n = symbols('n', positive=True)

            se = sp.sqrt(p * (1 - p) / n)

            return {
                "statistic": statistic,
                "distribution": "Normal (for large n)",
                "mean": str(p),
                "standard_error": str(se),
                "latex": latex(se),
                "note": "p̂ ~ N(p, p(1-p)/n) for large n",
                "result": str(se)
            }

        elif statistic == "sample_variance":
            return {
                "statistic": statistic,
                "distribution": "Chi-square",
                "note": "(n-1)s²/σ² ~ χ²(n-1)",
                "degrees_of_freedom": "n - 1"
            }

        else:
            raise ValidationError(f"Unknown statistic: {statistic}")

    def handle_experimental_design(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle experimental_design."""
        design_type = arguments["design_type"]
        factors = arguments.get("factors", [])

        # Validate factor names if provided
        if isinstance(factors, list) and len(factors) > 0:
            factors = [validate_variable_name(f) for f in factors]

        if design_type == "factorial":
            n_factors = len(factors)
            return {
                "design_type": design_type,
                "factors": factors,
                "n_factors": n_factors,
                "note": f"Full factorial design with {n_factors} factors",
                "total_combinations": f"Product of levels of all {n_factors} factors"
            }

        elif design_type == "randomized_block":
            return {
                "design_type": design_type,
                "factors": factors,
                "note": "Randomized block design controls for blocking variable",
                "structure": "Random assignment within each block"
            }

        elif design_type == "latin_square":
            return {
                "design_type": design_type,
                "note": "Latin square design controls for two blocking variables",
                "structure": "Each treatment appears once in each row and column"
            }

        else:
            raise ValidationError(f"Unknown design type: {design_type}")


# Entry point
server = StatisticsServer()

if __name__ == "__main__":
    server.run()
