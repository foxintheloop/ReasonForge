# reasonforge-analysis (17 tools)

**Differential equations, transforms, and signal processing**

**Installation:** `pip install reasonforge-analysis`

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-algebra](./reasonforge-algebra-examples.md) | [Next: reasonforge-geometry](./reasonforge-geometry-examples.md)

---

## Differential Equations (3 tools)

### Example 34: Solve ODE (dsolve_ode)

**User:** "Solve the differential equation dy/dx = 2*x with initial condition y(0) = 1"

**Claude with ReasonForge:**
```
Using dsolve_ode tool from reasonforge-analysis...

Result:
{
  "equation": "Derivative(y(x), x) - 2*x",
  "function": "y(x)",
  "solution": "y(x) = x**2 + 1",
  "initial_conditions": {"y(0)": "1"},
  "verified": true
}
```

### Example 35: Solve PDE (pdsolve_pde)

**User:** "Solve the wave equation ∂²u/∂t² = c²∂²u/∂x²"

**Claude with ReasonForge:**
```
Using pdsolve_pde tool from reasonforge-analysis...

Result:
{
  "equation": "Derivative(u(x, t), (t, 2)) - c**2*Derivative(u(x, t), (x, 2))",
  "function": "u(x, t)",
  "solution": "u(x, t) = F(x - c*t) + G(x + c*t)",
  "method": "separation_of_variables"
}
```

### Example 36: Symbolic ODE Initial Conditions

**User:** "Solve y' = a*y with y(0) = y_0 (symbolic initial condition)"

**Claude with ReasonForge:**
```
Using symbolic_ode_initial_conditions tool from reasonforge-analysis...

Result:
{
  "equation": "Derivative(y(x), x) - a*y(x)",
  "function": "y(x)",
  "initial_conditions": {"y(0)": "y_0"},
  "solution": "y(x) = y_0*exp(a*x)",
  "parameters": ["a", "y_0"],
  "latex": "y(x) = y_0 e^{ax}"
}
```

## Physics PDEs (3 tools)

### Example 37: Schrödinger Equation Solver

**User:** "Solve the 1D infinite square well problem"

**Claude with ReasonForge:**
```
Using schrodinger_equation_solver tool from reasonforge-analysis...

Result:
{
  "equation_type": "time_independent",
  "potential": "0 (inside), ∞ (outside)",
  "boundary_conditions": {"psi(0)": "0", "psi(L)": "0"},
  "dimension": 1,
  "wavefunctions": "sqrt(2/L) * sin(n*pi*x/L)",
  "energy_eigenvalues": "n^2*pi^2*hbar^2/(2*m*L^2)",
  "quantum_numbers": "n = 1, 2, 3, ..."
}
```

### Example 38: Wave Equation Solver

**User:** "Solve the 1D wave equation on a string with fixed ends"

**Claude with ReasonForge:**
```
Using wave_equation_solver tool from reasonforge-analysis...

Result:
{
  "wave_type": "string",
  "dimension": 1,
  "boundary_conditions": {"u(0,t)": "0", "u(L,t)": "0"},
  "initial_conditions": {"u(x,0)": "f(x)", "u_t(x,0)": "g(x)"},
  "solution": "Sum(A_n*cos(n*pi*c*t/L) + B_n*sin(n*pi*c*t/L))*sin(n*pi*x/L)",
  "wave_speed": "c = sqrt(T/mu)"
}
```

### Example 39: Heat Equation Solver

**User:** "Solve 1D heat equation with Dirichlet boundary conditions"

**Claude with ReasonForge:**
```
Using heat_equation_solver tool from reasonforge-analysis...

Result:
{
  "geometry": "1D",
  "boundary_conditions": {"u(0,t)": "0", "u(L,t)": "0"},
  "initial_conditions": {"u(x,0)": "f(x)"},
  "solution": "Sum(B_n * exp(-n^2*pi^2*alpha*t/L^2) * sin(n*pi*x/L))",
  "thermal_diffusivity": "alpha = k/(rho*c)"
}
```

## Transforms (5 tools)

### Example 40: Laplace Transform

**User:** "Find the Laplace transform of e^(-at)"

**Claude with ReasonForge:**
```
Using laplace_transform tool from reasonforge-analysis...

Result:
{
  "expression": "exp(-a*t)",
  "variable": "t",
  "transform_variable": "s",
  "inverse": false,
  "result": "1/(s + a)",
  "latex": "\\mathcal{L}\\{e^{-at}\\} = \\frac{1}{s + a}",
  "region_of_convergence": "Re(s) > -Re(a)"
}
```

### Example 41: Fourier Transform

**User:** "Calculate the Fourier transform of a Gaussian e^(-x^2)"

**Claude with ReasonForge:**
```
Using fourier_transform tool from reasonforge-analysis...

Result:
{
  "expression": "exp(-x**2)",
  "variable": "x",
  "transform_variable": "k",
  "inverse": false,
  "result": "sqrt(pi)*exp(-k**2/4)",
  "latex": "\\mathcal{F}\\{e^{-x^2}\\} = \\sqrt{\\pi} e^{-k^2/4}"
}
```

### Example 42: Z-Transform

**User:** "Find the Z-transform of a^n*u[n]"

**Claude with ReasonForge:**
```
Using z_transform tool from reasonforge-analysis...

Result:
{
  "sequence": "a**n",
  "n_variable": "n",
  "z_variable": "z",
  "inverse": false,
  "result": "z/(z - a)",
  "latex": "\\mathcal{Z}\\{a^n u[n]\\} = \\frac{z}{z - a}",
  "region_of_convergence": "|z| > |a|"
}
```

### Example 43: Mellin Transform

**User:** "Find the Mellin transform of e^(-x)"

**Claude with ReasonForge:**
```
Using mellin_transform tool from reasonforge-analysis...

Result:
{
  "expression": "exp(-x)",
  "variable": "x",
  "transform_variable": "s",
  "inverse": false,
  "result": "gamma(s)",
  "latex": "\\mathcal{M}\\{e^{-x}\\} = \\Gamma(s)"
}
```

### Example 44: Integral Transforms Custom

**User:** "Apply Hankel transform of order 0 to f(r) = e^(-r^2)"

**Claude with ReasonForge:**
```
Using integral_transforms_custom tool from reasonforge-analysis...

Result:
{
  "transform_type": "hankel",
  "order": 0,
  "expression": "exp(-r**2)",
  "variable": "r",
  "kernel": "r*J_0(k*r)",
  "result": "exp(-k^2/4)/(2)",
  "latex": "\\mathcal{H}_0\\{e^{-r^2}\\} = \\frac{1}{2}e^{-k^2/4}",
  "inverse_available": true
}
```

## Signal Processing (2 tools)

### Example 45: Convolution

**User:** "Compute the convolution of e^(-t) and e^(-2t)"

**Claude with ReasonForge:**
```
Using convolution tool from reasonforge-analysis...

Result:
{
  "f": "exp(-t)",
  "g": "exp(-2*t)",
  "variable": "t",
  "convolution_type": "continuous",
  "result": "exp(-t) - exp(-2*t)",
  "latex": "(f * g)(t) = e^{-t} - e^{-2t}"
}
```

### Example 46: Transfer Function Analysis

**User:** "Analyze the transfer function H(s) = 1/(s^2 + 2s + 2)"

**Claude with ReasonForge:**
```
Using transfer_function_analysis tool from reasonforge-analysis...

Result:
{
  "transfer_function": "1/(s**2 + 2*s + 2)",
  "variable": "s",
  "poles": ["-1 - I", "-1 + I"],
  "zeros": [],
  "stability": "stable (all poles in left half-plane)",
  "damping_ratio": "0.707",
  "natural_frequency": "sqrt(2)"
}
```

## Asymptotic Methods (2 tools)

### Example 47: Perturbation Theory

**User:** "Apply perturbation theory to x'' + x + ε*x^3 = 0"

**Claude with ReasonForge:**
```
Using perturbation_theory tool from reasonforge-analysis...

Result:
{
  "equation": "x'' + x + epsilon*x**3",
  "perturbation_type": "regular",
  "small_parameter": "epsilon",
  "order": 2,
  "expansion": "x = x_0 + epsilon*x_1 + epsilon^2*x_2 + O(epsilon^3)",
  "zeroth_order": "x_0'' + x_0 = 0, x_0 = A*cos(t + phi)",
  "first_order": "x_1'' + x_1 = -x_0^3",
  "secular_terms": "Identified and removed"
}
```

### Example 48: Asymptotic Analysis

**User:** "Find asymptotic expansion of e^x for x → ∞"

**Claude with ReasonForge:**
```
Using asymptotic_analysis tool from reasonforge-analysis...

Result:
{
  "expression": "exp(x)",
  "variable": "x",
  "limit_point": "inf",
  "order": 3,
  "asymptotic_series": "exp(x) ~ exp(x) * (1 + O(1/x))",
  "leading_behavior": "exp(x)",
  "interpretation": "Exponential growth dominates all polynomial terms"
}
```

## Special Functions (2 tools)

### Example 49: Special Functions Properties

**User:** "Get the recurrence relation for Legendre polynomials"

**Claude with ReasonForge:**
```
Using special_functions_properties tool from reasonforge-analysis...

Result:
{
  "function_type": "legendre",
  "operation": "recurrence",
  "parameters": {"n": "n"},
  "recurrence_relation": "(n+1)*P_{n+1}(x) = (2*n+1)*x*P_n(x) - n*P_{n-1}(x)",
  "latex": "(n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)",
  "initial_conditions": {"P_0": "1", "P_1": "x"}
}
```

### Example 50: Symbolic Optimization Setup

**User:** "Set up optimization problem: minimize x^2 + y^2 subject to x + y = 1"

**Claude with ReasonForge:**
```
Using symbolic_optimization_setup tool from reasonforge-analysis...

Result:
{
  "objective": "x**2 + y**2",
  "equality_constraints": ["x + y - 1"],
  "inequality_constraints": [],
  "variables": ["x", "y"],
  "lagrangian": "x**2 + y**2 - lambda*(x + y - 1)",
  "kkt_conditions": [
    "2*x - lambda = 0",
    "2*y - lambda = 0",
    "x + y - 1 = 0"
  ],
  "stationary_points": [{"x": "1/2", "y": "1/2", "lambda": "1"}]
}
```

---

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-algebra](./reasonforge-algebra-examples.md) | [Next: reasonforge-geometry](./reasonforge-geometry-examples.md)
