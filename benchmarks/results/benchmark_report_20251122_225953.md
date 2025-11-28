# ReasonForge Benchmark Report

**Generated:** 2025-11-22 22:59:53
**Total Tests:** 107
**Providers:** reasonforge

---

## Executive Summary

| Provider | Accuracy | Tests Passed | Avg Latency | Total Cost |
|----------|----------|--------------|-------------|------------|
| **reasonforge** | **100.0%** | 107/107 | 15ms | $0.0000 |

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
| expr_003 | reasonforge | ✓ PASS | 4 | $0.000000 | Symbolically equivalent |
| expr_004 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| expr_005 | reasonforge | ✓ PASS | 22 | $0.000000 | Symbolically equivalent |
| expr_006 | reasonforge | ✓ PASS | 4 | $0.000000 | Symbolically equivalent |
| expr_007 | reasonforge | ✓ PASS | 3 | $0.000000 | Symbolically equivalent |
| expr_008 | reasonforge | ✓ PASS | 4 | $0.000000 | Symbolically equivalent |
| expr_009 | reasonforge | ✓ PASS | 4 | $0.000000 | Symbolically equivalent |
| expr_010 | reasonforge | ✓ PASS | 5 | $0.000000 | Symbolically equivalent |
| expr_011 | reasonforge | ✓ PASS | 59 | $0.000000 | Symbolically equivalent (ignoring constant) |
| expr_012 | reasonforge | ✓ PASS | 23 | $0.000000 | Symbolically equivalent |
| expr_013 | reasonforge | ✓ PASS | 29 | $0.000000 | Contains 'x' |
| expr_014 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'nonexistent' |
| expr_015 | reasonforge | ✓ PASS | 11 | $0.000000 | Contains '6' |
| alg_001 | reasonforge | ✓ PASS | 16 | $0.000000 | Lists match |
| alg_002 | reasonforge | ✓ PASS | 12 | $0.000000 | Lists match |
| alg_003 | reasonforge | ✓ PASS | 3 | $0.000000 | Contains '2' |
| alg_004 | reasonforge | ✓ PASS | 46 | $0.000000 | Contains '(-3, -4)' |
| alg_005 | reasonforge | ✓ PASS | 0 | $0.000000 | Lists match |
| alg_006 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'error' |
| alg_007 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains '-2' |
| alg_008 | reasonforge | ✓ PASS | 4 | $0.000000 | Contains 'sqrt' |
| alg_009 | reasonforge | ✓ PASS | 9 | $0.000000 | Contains 'eigenvalue' |
| alg_010 | reasonforge | ✓ PASS | 4 | $0.000000 | Contains '-1' |
| alg_011 | reasonforge | ✓ PASS | 17 | $0.000000 | Contains '1/2' |
| alg_012 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_013 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_014 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_015 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| alg_016 | reasonforge | ✓ PASS | 20 | $0.000000 | Value 2.0 >= 1.0 |
| alg_017 | reasonforge | ✓ PASS | 5 | $0.000000 | Symbolically equivalent |
| alg_018 | reasonforge | ✓ PASS | 2 | $0.000000 | Symbolically equivalent (ignoring constant) |
| analysis_001 | reasonforge | ✓ PASS | 33 | $0.000000 | Contains 'exp(x)' |
| analysis_002 | reasonforge | ✓ PASS | 18 | $0.000000 | Contains 'u(x,t)' |
| analysis_004 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_005 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_006 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_007 | reasonforge | ✓ PASS | 192 | $0.000000 | Symbolically equivalent |
| analysis_008 | reasonforge | ✓ PASS | 578 | $0.000000 | Contains 'sqrt(pi)' |
| analysis_009 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_010 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_011 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_012 | reasonforge | ✓ PASS | 69 | $0.000000 | Contains 'exp' |
| analysis_013 | reasonforge | ✓ PASS | 10 | $0.000000 | Contains '-1' |
| analysis_014 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_015 | reasonforge | ✓ PASS | 6 | $0.000000 | Symbolically equivalent |
| analysis_016 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| analysis_017 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| geom_001 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| geom_002 | reasonforge | ✓ PASS | 3 | $0.000000 | Contains 'field_' |
| geom_003 | reasonforge | ✓ PASS | 6 | $0.000000 | Contains 'curl' |
| geom_004 | reasonforge | ✓ PASS | 5 | $0.000000 | Contains 'divergence' |
| geom_005 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains '0' |
| geom_006 | reasonforge | ✓ PASS | 7 | $0.000000 | Exact string match |
| geom_007 | reasonforge | ✓ PASS | 0 | $0.000000 | Value 3.0 >= 0.0 |
| geom_008 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'error' |
| geom_009 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'metric_custom' |
| geom_010 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'latex' |
| geom_012 | reasonforge | ✓ PASS | 10 | $0.000000 | Contains 'expression' |
| geom_013 | reasonforge | ✓ PASS | 4 | $0.000000 | Exact string match |
| geom_014 | reasonforge | ✓ PASS | 1 | $0.000000 | Exact string match |
| geom_015 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact numeric match: 2.0 |
| stat_001 | reasonforge | ✓ PASS | 42 | $0.000000 | Symbolically equivalent |
| stat_002 | reasonforge | ✓ PASS | 3 | $0.000000 | Contains '0.' |
| stat_003 | reasonforge | ✓ PASS | 2 | $0.000000 | Contains 't_statistic' |
| stat_004 | reasonforge | ✓ PASS | 248 | $0.000000 | Symbolically equivalent |
| stat_005 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'correlation' |
| stat_006 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| stat_007 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'margin' |
| stat_008 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| stat_009 | reasonforge | ✓ PASS | 0 | $0.000000 | Lists match |
| stat_010 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'Mean' |
| stat_011 | reasonforge | ✓ PASS | 2 | $0.000000 | Exact string match |
| stat_012 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| stat_013 | reasonforge | ✓ PASS | 2 | $0.000000 | Contains 'F_statistic' |
| stat_015 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'mean' |
| stat_016 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'error' |
| phys_001 | reasonforge | ✓ PASS | 14 | $0.000000 | Contains 'equations' |
| phys_002 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'hamiltonian' |
| phys_003 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'conserved' |
| phys_004 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'error' |
| phys_005 | reasonforge | ✓ PASS | 7 | $0.000000 | Contains 'sqrt' |
| phys_006 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'quantum_state' |
| phys_007 | reasonforge | ✓ PASS | 3 | $0.000000 | Contains 'quantum_state' |
| phys_008 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'quantum_state' |
| phys_009 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'entanglement_measure' |
| phys_010 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact numeric match: 2.0 |
| phys_011 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains '[' |
| phys_012 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains '1' |
| phys_013 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains '[[' |
| phys_014 | reasonforge | ✓ PASS | 4 | $0.000000 | Contains 'error' |
| phys_015 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'evolution' |
| phys_016 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'c**2' |
| logic_001 | reasonforge | ✓ PASS | 13 | $0.000000 | Contains 'n**2' |
| logic_002 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'rules' |
| logic_003 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'theorem' |
| logic_004 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'concept' |
| logic_005 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'mapping' |
| logic_006 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'conjecture' |
| logic_007 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'forall' |
| logic_008 | reasonforge | ✓ PASS | 2 | $0.000000 | Contains '&' |
| logic_009 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'paths' |
| logic_010 | reasonforge | ✓ PASS | 1 | $0.000000 | Contains 'constraints' |
| logic_011 | reasonforge | ✓ PASS | 0 | $0.000000 | Contains 'axioms' |
| logic_012 | reasonforge | ✓ PASS | 0 | $0.000000 | Exact string match |
| logic_013 | reasonforge | ✓ PASS | 2 | $0.000000 | Contains 'a + b = b + a' |

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
