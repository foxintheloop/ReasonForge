# ReasonForge Benchmark Report

**Generated:** 2025-11-22 20:08:50
**Total Tests:** 63
**Providers:** reasonforge

---

## Executive Summary

| Provider | Accuracy | Tests Passed | Avg Latency | Total Cost |
|----------|----------|--------------|-------------|------------|
| **reasonforge** | **74.6%** | 47/63 | 26ms | $0.0000 |

---

## Key Findings

---

## Comparison Charts

### Complete Comparison

![Complete Comparison](combined_comparison.png)

### Accuracy Comparison

![Accuracy Comparison](accuracy_comparison.png)

### Speed Comparison

![Speed Comparison](latency_comparison.png)

### Cost Comparison

![Cost Comparison](cost_comparison.png)

### Accuracy by Category

![Accuracy by Category](category_breakdown.png)

---

## Detailed Test Results

| Test ID | Provider | Result | Latency (ms) | Cost | Details |
|---------|----------|--------|--------------|------|----------|
| expr_001 | reasonforge | ✓ PASS | 1 | $0.000000 | Symbolically equivalent |
| expr_002 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact numeric match: 3.0 |
| expr_003 | reasonforge | ✓ PASS | 3 | $0.000000 | Symbolically equivalent |
| expr_004 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| expr_005 | reasonforge | ✓ PASS | 26 | $0.000000 | Symbolically equivalent |
| expr_006 | reasonforge | ✓ PASS | 4 | $0.000000 | Symbolically equivalent |
| expr_007 | reasonforge | ✓ PASS | 5 | $0.000000 | Symbolically equivalent |
| expr_008 | reasonforge | ✓ PASS | 5 | $0.000000 | Symbolically equivalent |
| expr_009 | reasonforge | ✓ PASS | 5 | $0.000000 | Symbolically equivalent |
| expr_010 | reasonforge | ✓ PASS | 8 | $0.000000 | Symbolically equivalent |
| expr_011 | reasonforge | ✓ PASS | 86 | $0.000000 | Symbolically equivalent (ignoring constant) |
| expr_012 | reasonforge | ✓ PASS | 58 | $0.000000 | Symbolically equivalent |
| expr_013 | reasonforge | ✗ FAIL | 30 | $0.000000 | LLM: x - x**3/6 + O(x**5), Expected: -x**3/6 + x |
| expr_014 | reasonforge | ✗ FAIL | 0 | $0.000000 | Does not contain 'error' |
| expr_015 | reasonforge | ✗ FAIL | 13 | $0.000000 | Could not parse dict from: '{"x": "6", "y": "4"}' |
| alg_001 | reasonforge | ✓ PASS | 16 | $0.000000 | Lists match |
| alg_002 | reasonforge | ✓ PASS | 17 | $0.000000 | Lists match |
| alg_003 | reasonforge | ✗ FAIL | 5 | $0.000000 | Could not parse dict from: '(2, 1)' |
| alg_004 | reasonforge | ✗ FAIL | 59 | $0.000000 | Different lengths: LLM has 2, expected 0 |
| alg_005 | reasonforge | ✓ PASS | 0 | $0.000000 | Lists match |
| alg_006 | reasonforge | ✗ FAIL | 0 | $0.000000 | Validation error: Sympify of expression 'could not parse "Validation error in matrix_determinant: Matrix 'det_test' not found"' failed, because of exception being raised:
SyntaxError: invalid syntax (<string>, line 1) |
| alg_012 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_013 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_014 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_015 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_016 | reasonforge | ✓ PASS | 36 | $0.000000 | Value 2.0 >= 1.0 |
| alg_017 | reasonforge | ✓ PASS | 15 | $0.000000 | Symbolically equivalent |
| alg_018 | reasonforge | ✓ PASS | 3 | $0.000000 | Symbolically equivalent (ignoring constant) |
| analysis_001 | reasonforge | ✓ PASS | 77 | $0.000000 | Contains 'exp(x)' |
| analysis_002 | reasonforge | ✓ PASS | 27 | $0.000000 | Contains 'u(x,t)' |
| analysis_004 | reasonforge | ✗ FAIL | 0 | $0.000000 | LLM: 'symbolic_', Expected: 'symbolic_formulation' |
| analysis_005 | reasonforge | ✗ FAIL | 0 | $0.000000 | LLM: 'symbolic_', Expected: 'symbolic_formulation' |
| analysis_006 | reasonforge | ✗ FAIL | 0 | $0.000000 | LLM: 'symbolic_', Expected: 'symbolic_formulation' |
| analysis_007 | reasonforge | ✓ PASS | 376 | $0.000000 | Symbolically equivalent |
| analysis_009 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_010 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_011 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_014 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_015 | reasonforge | ✗ FAIL | 9 | $0.000000 | LLM: 1/x, Expected: 0 |
| analysis_016 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_017 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| geom_001 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| geom_006 | reasonforge | ✓ PASS | 9 | $0.000000 | Exact string match |
| geom_007 | reasonforge | ✓ PASS | 0 | $0.000000 | Value 3.0 >= 0.0 |
| geom_008 | reasonforge | ✗ FAIL | 0 | $0.000000 | LLM: '{"error": "Validation error in calculate_tensor: Metric 'metric_0' not found', Expected: 'inverse' |
| geom_011 | reasonforge | ✗ FAIL | 2 | $0.000000 | Does not contain '500' |
| geom_013 | reasonforge | ✓ PASS | 4 | $0.000000 | Exact string match |
| geom_014 | reasonforge | ✓ PASS | 2 | $0.000000 | Exact string match |
| geom_015 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact numeric match: 2.0 |
| stat_001 | reasonforge | ✓ PASS | 95 | $0.000000 | Symbolically equivalent |
| stat_004 | reasonforge | ✓ PASS | 609 | $0.000000 | Symbolically equivalent |
| stat_006 | reasonforge | ✓ PASS | 3 | $0.000000 | Exact string match |
| stat_008 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| stat_009 | reasonforge | ✓ PASS | 0 | $0.000000 | Lists match |
| stat_010 | reasonforge | ✗ FAIL | 2 | $0.000000 | LLM: 'Mean (1st moment)', Expected: 'Mean' |
| stat_011 | reasonforge | ✓ PASS | 5 | $0.000000 | Exact string match |
| stat_012 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| stat_016 | reasonforge | ✗ FAIL | 0 | $0.000000 | LLM: '{"error": "Validation error in experimental_design: Unknown design type: completely_randomized", "tool": "experimental_design", "error_type": "validation"}', Expected: 'completely_randomized' |
| phys_004 | reasonforge | ✗ FAIL | 0 | $0.000000 | LLM: '{"error": "Validation error in maxwell_equations: Unknown operation: gauss_law", "tool": "maxwell_equations", "error_type": "validation"}', Expected: 'gauss_law' |
| phys_010 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact numeric match: 2.0 |
| phys_014 | reasonforge | ✗ FAIL | 6 | $0.000000 | LLM: '{"error": "Error executing commutator_anticommutator: can't multiply sequence by non-int of type 'list'", "tool": "commutator_anticommutator", "error_type": "TypeError"}', Expected: 'commutator' |
| logic_012 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| logic_013 | reasonforge | ✓ PASS | 3 | $0.000000 | Contains 'a + b = b + a' |

---

## About ReasonForge

ReasonForge is a symbolic AI system built on SymPy that provides mathematically rigorous
computations with guaranteed accuracy. Unlike LLMs which may produce plausible but incorrect
answers, ReasonForge uses formal symbolic mathematics to ensure every result is verifiable.

**Key Benefits:**

- 🎯 **100% Accuracy** on symbolic mathematics (when properly configured)
- ⚡ **Fast** - Direct computation without API latency
- 💰 **Free** - No API costs or usage limits
- 🔒 **Private** - All computation happens locally
- ✅ **Verifiable** - Results can be independently checked

---

*Generated by ReasonForge Benchmark Suite*
