# reasonforge-statistics (16 tools)

**Probability, statistics, and data science**

**Installation:** `pip install reasonforge-statistics`

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-geometry](./reasonforge-geometry-examples.md) | [Next: reasonforge-physics](./reasonforge-physics-examples.md)

---

## Probability (3 tools)

### Example 66: Calculate Probability

**User:** "What is P(X ≤ 1.5) for X ~ Normal(0, 1)?"

**Claude with ReasonForge:**
```
Using calculate_probability tool from reasonforge-statistics...

Result:
{
  "distribution": "Normal",
  "parameters": {"mu": "0", "sigma": "1"},
  "operation": "cdf",
  "value": "1.5",
  "result": "erf(3*sqrt(2)/4)/2 + 1/2",
  "numerical": "0.9332"
}
```

### Example 67: Probability Distributions

**User:** "Create a combination of two normal distributions"

**Claude with ReasonForge:**
```
Using probability_distributions tool from reasonforge-statistics...

Result:
{
  "operation": "combine",
  "distributions": ["Normal(0, 1)", "Normal(2, 1)"],
  "result": "0.5*Normal(0, 1) + 0.5*Normal(2, 1)",
  "type": "mixture",
  "latex": "\\frac{1}{2}\\mathcal{N}(0,1) + \\frac{1}{2}\\mathcal{N}(2,1)"
}
```

### Example 68: Distribution Properties

**User:** "Calculate the mean and variance of a Poisson distribution with λ=5"

**Claude with ReasonForge:**
```
Using distribution_properties tool from reasonforge-statistics...

Result:
{
  "distribution": "Poisson",
  "parameters": {"lambda": "5"},
  "properties": {
    "mean": "5",
    "variance": "5",
    "skewness": "1/sqrt(5)",
    "kurtosis": "1/5"
  }
}
```

## Inference (4 tools)

### Example 69: Bayesian Inference

**User:** "Use Bayes' theorem: P(A|B) given P(B|A), P(A), and P(B)"

**Claude with ReasonForge:**
```
Using bayesian_inference tool from reasonforge-statistics...

Result:
{
  "prior": "P_A",
  "likelihood": "P_B_given_A",
  "evidence": "P_B",
  "posterior": "P_A*P_B_given_A/P_B",
  "latex": "P(A|B) = \\frac{P(A) \\cdot P(B|A)}{P(B)}"
}
```

### Example 70: Statistical Test

**User:** "Set up a t-test for comparing two means"

**Claude with ReasonForge:**
```
Using statistical_test tool from reasonforge-statistics...

Result:
{
  "test_type": "t-test",
  "parameters": {
    "mean1": "mu1",
    "mean2": "mu2",
    "var1": "s1**2",
    "var2": "s2**2",
    "n1": "n1",
    "n2": "n2"
  },
  "test_statistic": "(mu1 - mu2)/sqrt(s1**2/n1 + s2**2/n2)",
  "degrees_of_freedom": "n1 + n2 - 2",
  "latex": "t = \\frac{\\mu_1 - \\mu_2}{\\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}}"
}
```

### Example 71: Hypothesis Test Symbolic

**User:** "Generate the formula for a chi-square test statistic"

**Claude with ReasonForge:**
```
Using hypothesis_test_symbolic tool from reasonforge-statistics...

Result:
{
  "test_type": "chi-square",
  "parameters": {
    "observed": "O_i",
    "expected": "E_i"
  },
  "test_statistic": "Sum((O_i - E_i)^2 / E_i)",
  "latex": "\\chi^2 = \\sum_{i=1}^{n} \\frac{(O_i - E_i)^2}{E_i}",
  "degrees_of_freedom": "n - 1"
}
```

### Example 72: Confidence Intervals

**User:** "Calculate a 95% confidence interval for a population mean"

**Claude with ReasonForge:**
```
Using confidence_intervals tool from reasonforge-statistics...

Result:
{
  "parameter_type": "mean",
  "confidence_level": 0.95,
  "sample_stats": {"mean": "xbar", "std": "s", "n": "n"},
  "interval": "(xbar - 1.96*s/sqrt(n), xbar + 1.96*s/sqrt(n))",
  "latex": "\\bar{x} \\pm 1.96 \\frac{s}{\\sqrt{n}}"
}
```

## Regression (2 tools)

### Example 73: Regression Symbolic

**User:** "Derive the equation for linear regression"

**Claude with ReasonForge:**
```
Using regression_symbolic tool from reasonforge-statistics...

Result:
{
  "regression_type": "linear",
  "variables": ["x"],
  "equation": "y = beta_0 + beta_1*x",
  "coefficients": {
    "beta_1": "Cov(x,y)/Var(x)",
    "beta_0": "mean(y) - beta_1*mean(x)"
  },
  "latex": "y = \\beta_0 + \\beta_1 x"
}
```

### Example 74: Correlation Analysis

**User:** "Find the correlation between two normal variables"

**Claude with ReasonForge:**
```
Using correlation_analysis tool from reasonforge-statistics...

Result:
{
  "x_distribution": "Normal",
  "y_distribution": "Normal",
  "parameters": {"rho": "rho"},
  "covariance": "rho*sigma_x*sigma_y",
  "correlation": "rho",
  "latex": "\\text{Corr}(X,Y) = \\rho"
}
```

## Data Science (7 tools)

### Example 75: Symbolic Dataframe

**User:** "Create a symbolic dataframe with columns ['age', 'height', 'weight']"

**Claude with ReasonForge:**
```
Using symbolic_dataframe tool from reasonforge-statistics...

Result:
{
  "columns": ["age", "height", "weight"],
  "symbolic_data": true,
  "symbols": {
    "age": "age_i",
    "height": "h_i",
    "weight": "w_i"
  },
  "dataframe_key": "df_1"
}
```

### Example 76: Statistical Moments Symbolic

**User:** "Derive the formula for the third central moment (skewness)"

**Claude with ReasonForge:**
```
Using statistical_moments_symbolic tool from reasonforge-statistics...

Result:
{
  "data_symbol": "X",
  "order": 3,
  "central": true,
  "standardized": true,
  "formula": "E[(X - mu)^3] / sigma^3",
  "latex": "\\gamma_1 = \\frac{E[(X - \\mu)^3]}{\\sigma^3}",
  "interpretation": "Skewness coefficient"
}
```

### Example 77: Time Series Symbolic

**User:** "Create an ARMA(2,1) model"

**Claude with ReasonForge:**
```
Using time_series_symbolic tool from reasonforge-statistics...

Result:
{
  "model_type": "ARMA",
  "p": 2,
  "q": 1,
  "equation": "X_t = phi_1*X_{t-1} + phi_2*X_{t-2} + epsilon_t + theta_1*epsilon_{t-1}",
  "parameters": ["phi_1", "phi_2", "theta_1"],
  "latex": "X_t = \\phi_1 X_{t-1} + \\phi_2 X_{t-2} + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1}"
}
```

### Example 78: ANOVA Symbolic

**User:** "Derive one-way ANOVA formulas for 3 groups"

**Claude with ReasonForge:**
```
Using anova_symbolic tool from reasonforge-statistics...

Result:
{
  "anova_type": "one_way",
  "groups": 3,
  "formulas": {
    "SST": "Sum((X_ij - X_bar)^2)",
    "SSB": "Sum(n_i*(X_bar_i - X_bar)^2)",
    "SSW": "Sum((X_ij - X_bar_i)^2)",
    "F_statistic": "MSB/MSW"
  },
  "latex": "F = \\frac{MSB}{MSW} = \\frac{SSB/(k-1)}{SSW/(N-k)}"
}
```

### Example 79: Multivariate Statistics

**User:** "Generate the multivariate normal PDF formula"

**Claude with ReasonForge:**
```
Using multivariate_statistics tool from reasonforge-statistics...

Result:
{
  "operation": "multivariate_normal",
  "parameters": {
    "mean": "mu",
    "covariance": "Sigma",
    "dimension": "p"
  },
  "pdf": "(2*pi)^(-p/2) * det(Sigma)^(-1/2) * exp(-1/2 * (x-mu)^T * Sigma^(-1) * (x-mu))",
  "latex": "f(\\mathbf{x}) = \\frac{1}{(2\\pi)^{p/2}|\\Sigma|^{1/2}} e^{-\\frac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^T\\Sigma^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})}"
}
```

### Example 80: Sampling Distributions

**User:** "Derive the sampling distribution of the sample mean"

**Claude with ReasonForge:**
```
Using sampling_distributions tool from reasonforge-statistics...

Result:
{
  "statistic": "mean",
  "population_parameters": {"mu": "mu", "sigma": "sigma"},
  "sample_size": "n",
  "distribution": "Normal(mu, sigma/sqrt(n))",
  "latex": "\\bar{X} \\sim \\mathcal{N}\\left(\\mu, \\frac{\\sigma}{\\sqrt{n}}\\right)"
}
```

### Example 81: Experimental Design

**User:** "Generate a 2^3 factorial design"

**Claude with ReasonForge:**
```
Using experimental_design tool from reasonforge-statistics...

Result:
{
  "design_type": "factorial",
  "factors": ["A", "B", "C"],
  "levels": {"A": 2, "B": 2, "C": 2},
  "design_matrix": [
    {"A": -1, "B": -1, "C": -1},
    {"A": 1, "B": -1, "C": -1},
    {"A": -1, "B": 1, "C": -1},
    {"A": 1, "B": 1, "C": -1},
    {"A": -1, "B": -1, "C": 1},
    {"A": 1, "B": -1, "C": 1},
    {"A": -1, "B": 1, "C": 1},
    {"A": 1, "B": 1, "C": 1}
  ],
  "run_count": 8
}
```

---

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-geometry](./reasonforge-geometry-examples.md) | [Next: reasonforge-physics](./reasonforge-physics-examples.md)
