# ReasonForge Statistics

**Probability, statistics, and data science - 16 tools**

An MCP (Model Context Protocol) server that provides Claude with statistical and probabilistic reasoning capabilities using SymPy's symbolic engine.

> **Beta Status**: This package is functional but still undergoing testing. Some edge cases may not be fully covered. Please report issues on [GitHub](https://github.com/foxintheloop/ReasonForge/issues).

## Capabilities

- **Probability Theory** - Symbolic probability calculations and distributions
- **Statistical Inference** - Bayesian inference, hypothesis testing, confidence intervals
- **Regression Analysis** - Symbolic regression and correlation
- **Data Science** - Statistical moments, time series, ANOVA, multivariate statistics

## Installation

```bash
pip install reasonforge-statistics
```

Or install from source:

```bash
git clone https://github.com/foxintheloop/ReasonForge.git
cd ReasonForge
pip install -e packages/reasonforge -e packages/reasonforge-statistics
```

## Claude Desktop Configuration

Add to your Claude Desktop config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "reasonforge-statistics": {
      "command": "python",
      "args": ["-m", "reasonforge_statistics"]
    }
  }
}
```

## Tools

### Probability (3 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `calculate_probability` | Compute symbolic probabilities | P(A and B) given P(A), P(B) |
| `probability_distributions` | Work with probability distributions | Normal, Poisson, Binomial |
| `distribution_properties` | Get distribution properties | Mean, variance, MGF |

### Statistical Inference (4 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `bayesian_inference` | Perform Bayesian updates | Posterior from prior and likelihood |
| `statistical_test` | Conduct statistical tests | t-test, chi-square |
| `hypothesis_test_symbolic` | Symbolic hypothesis testing | Power analysis |
| `confidence_intervals` | Compute confidence intervals | 95% CI for mean |

### Regression (2 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `regression_symbolic` | Symbolic regression analysis | Fit polynomial models |
| `correlation_analysis` | Compute correlation coefficients | Pearson, Spearman |

### Data Science (7 tools)

| Tool | Description | Example Use |
|------|-------------|-------------|
| `symbolic_dataframe` | Symbolic data manipulation | Transform data symbolically |
| `statistical_moments_symbolic` | Compute moments of distributions | Skewness, kurtosis |
| `time_series_symbolic` | Time series analysis | Trend, seasonality |
| `anova_symbolic` | Analysis of variance | One-way, two-way ANOVA |
| `multivariate_statistics` | Multivariate analysis | PCA, factor analysis |
| `sampling_distributions` | Sampling distribution theory | CLT applications |
| `experimental_design` | Design of experiments | Factorial designs |

## Example Usage

Once configured, you can ask Claude:

**Probability:**
- "What is the probability of getting at least 3 heads in 5 coin flips?"
- "Calculate the expected value of a Poisson distribution with lambda = 5"

**Inference:**
- "Perform a Bayesian update: prior is Beta(2,2), observed 7 successes in 10 trials"
- "Calculate the 95% confidence interval for a sample mean"

**Regression:**
- "Fit a quadratic model to these data points: (1,2), (2,5), (3,10), (4,17)"
- "What is the correlation coefficient for these paired observations?"

**Data Analysis:**
- "Compute the skewness and kurtosis of a normal distribution"
- "Set up a one-way ANOVA for comparing three treatment groups"

## Dependencies

- Python >= 3.10
- mcp >= 1.0.0
- sympy >= 1.12
- reasonforge (core library)

## Running Tests

```bash
pytest packages/reasonforge-statistics/tests/ -v
```

## License

MIT License - See [LICENSE](../../LICENSE) for details.

## Related Packages

- [reasonforge](https://pypi.org/project/reasonforge/) - Core symbolic computation library
- [reasonforge-algebra](https://pypi.org/project/reasonforge-algebra/) - Matrix operations for statistics
- [reasonforge-analysis](https://pypi.org/project/reasonforge-analysis/) - Calculus for probability theory
- [reasonforge-logic](https://pypi.org/project/reasonforge-logic/) - Logical reasoning
