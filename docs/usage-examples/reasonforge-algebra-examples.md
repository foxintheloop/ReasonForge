# reasonforge-algebra (18 tools)

**Equation solving, matrices, and optimization**

**Installation:** `pip install reasonforge-algebra`

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-expressions](./reasonforge-expressions-examples.md) | [Next: reasonforge-analysis](./reasonforge-analysis-examples.md)

---

## Equation Solving (4 tools)

### Example 16: Solve Equations

**User:** "Solve the equation x^2 - 5x + 6 = 0"

**Claude with ReasonForge:**
```
Using solve_equations tool from reasonforge-algebra...

Result:
{
  "solutions": [
    {"x": "2"},
    {"x": "3"}
  ],
  "explanation": "Given system of equations:\n  1. x**2 - 5*x + 6 = 0\n\nSolving for variables: ['x']\n\nSolutions found:\n  Solution 1:\n    x = 2\n  Solution 2:\n    x = 3",
  "verification": [true, true]
}
```

### Example 17: Solve Algebraically

**User:** "Solve x^2 = 2 over the real numbers"

**Claude with ReasonForge:**
```
Using solve_algebraically tool from reasonforge-algebra...

Result:
{
  "expression_key": "expr_1",
  "variable": "x",
  "domain": "real",
  "solutions": ["-sqrt(2)", "sqrt(2)"],
  "count": 2
}
```

### Example 18: Solve Linear System

**User:** "Solve the system: 2x + 3y = 7, x - y = 1"

**Claude with ReasonForge:**
```
Using solve_linear_system tool from reasonforge-algebra...

Result:
{
  "equations": ["2*x + 3*y - 7", "x - y - 1"],
  "variables": ["x", "y"],
  "solution": {
    "x": "2",
    "y": "1"
  },
  "method": "linsolve"
}
```

### Example 19: Solve Nonlinear System

**User:** "Solve x^2 + y^2 = 25, x - y = 1"

**Claude with ReasonForge:**
```
Using solve_nonlinear_system tool from reasonforge-algebra...

Result:
{
  "equations": ["x**2 + y**2 - 25", "x - y - 1"],
  "variables": ["x", "y"],
  "solutions": [
    {"x": "4", "y": "3"},
    {"x": "-3", "y": "-4"}
  ],
  "method": "nonlinsolve"
}
```

## Matrix Operations (5 tools)

### Example 20: Create Matrix

**User:** "Create a 2x2 matrix [[1, 2], [3, 4]]"

**Claude with ReasonForge:**
```
Using create_matrix tool from reasonforge-algebra...

Result:
{
  "elements": [[1, 2], [3, 4]],
  "key": "matrix_1",
  "shape": [2, 2],
  "latex": "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}"
}
```

### Example 21: Matrix Determinant

**User:** "Find the determinant of [[1, 2], [3, 4]]"

**Claude with ReasonForge:**
```
Using matrix_determinant tool from reasonforge-algebra...

Result:
{
  "matrix_key": "matrix_1",
  "determinant": "-2",
  "latex": "\\det(A) = -2"
}
```

### Example 22: Matrix Inverse

**User:** "Calculate the inverse of [[1, 2], [3, 4]]"

**Claude with ReasonForge:**
```
Using matrix_inverse tool from reasonforge-algebra...

Result:
{
  "matrix_key": "matrix_1",
  "inverse": [[-2, 1], [3/2, -1/2]],
  "stored_key": "matrix_2",
  "latex": "A^{-1} = \\begin{bmatrix} -2 & 1 \\\\ \\frac{3}{2} & -\\frac{1}{2} \\end{bmatrix}"
}
```

### Example 23: Matrix Eigenvalues

**User:** "Find the eigenvalues of [[1, 2], [3, 4]]"

**Claude with ReasonForge:**
```
Using matrix_eigenvalues tool from reasonforge-algebra...

Result:
{
  "matrix_key": "matrix_1",
  "eigenvalues": {
    "5/2 - sqrt(33)/2": 1,
    "5/2 + sqrt(33)/2": 1
  },
  "latex": "\\left\\{ \\frac{5}{2} - \\frac{\\sqrt{33}}{2} : 1, \\  \\frac{5}{2} + \\frac{\\sqrt{33}}{2} : 1\\right\\}"
}
```

### Example 24: Matrix Eigenvectors

**User:** "Find the eigenvectors of [[1, 2], [3, 4]]"

**Claude with ReasonForge:**
```
Using matrix_eigenvectors tool from reasonforge-algebra...

Result:
{
  "matrix_key": "matrix_1",
  "eigenvectors": [
    {
      "eigenvalue": "5/2 - sqrt(33)/2",
      "eigenvector": "[(-3/4 + sqrt(33)/4, 1)]",
      "multiplicity": {"algebraic": 1, "geometric": 1}
    },
    {
      "eigenvalue": "5/2 + sqrt(33)/2",
      "eigenvector": "[(-sqrt(33)/4 - 3/4, 1)]",
      "multiplicity": {"algebraic": 1, "geometric": 1}
    }
  ]
}
```

## Optimization (6 tools)

### Example 25: Optimize Function

**User:** "Find the critical points of f(x,y) = x^2 + y^2 - 2x - 4y + 5"

**Claude with ReasonForge:**
```
Using optimize_function tool from reasonforge-algebra...

Result:
{
  "objective": "x**2 + y**2 - 2*x - 4*y + 5",
  "variables": ["x", "y"],
  "critical_points": [
    {"x": "1", "y": "2"}
  ],
  "function_values": [
    {"point": {"x": "1", "y": "2"}, "value": "0"}
  ]
}
```

### Example 26: Lagrange Multipliers

**User:** "Maximize f(x,y) = xy subject to x^2 + y^2 = 1"

**Claude with ReasonForge:**
```
Using lagrange_multipliers tool from reasonforge-algebra...

Result:
{
  "objective": "x*y",
  "constraints": ["x**2 + y**2 - 1"],
  "variables": ["x", "y"],
  "lagrangian": "x*y - lambda*(x**2 + y**2 - 1)",
  "critical_points": [
    {"x": "sqrt(2)/2", "y": "sqrt(2)/2", "lambda": "1/2"},
    {"x": "-sqrt(2)/2", "y": "-sqrt(2)/2", "lambda": "1/2"}
  ],
  "optimal_value": "1/2"
}
```

### Example 27: Linear Programming

**User:** "Minimize 3x + 4y subject to x + y ≥ 5, 2x + y ≥ 8, x,y ≥ 0"

**Claude with ReasonForge:**
```
Using linear_programming tool from reasonforge-algebra...

Result:
{
  "objective": "3*x + 4*y",
  "constraints": ["x + y - 5", "2*x + y - 8", "x", "y"],
  "variables": ["x", "y"],
  "minimize": true,
  "optimal_solution": {"x": "3", "y": "2"},
  "optimal_value": "17"
}
```

### Example 28: Convex Optimization

**User:** "Verify that f(x) = x^2 is convex and find the minimum"

**Claude with ReasonForge:**
```
Using convex_optimization tool from reasonforge-algebra...

Result:
{
  "objective": "x**2",
  "constraints": [],
  "variables": ["x"],
  "is_convex": true,
  "hessian": "2 (positive definite)",
  "optimal_solution": {"x": "0"},
  "optimal_value": "0"
}
```

### Example 29: Calculus of Variations

**User:** "Find the shortest path between two points (geodesic)"

**Claude with ReasonForge:**
```
Using calculus_of_variations tool from reasonforge-algebra...

Result:
{
  "functional": "sqrt(1 + y'**2)",
  "function_name": "y",
  "independent_var": "x",
  "euler_lagrange": "y'' = 0",
  "extremal_function": "y = a*x + b",
  "interpretation": "Straight line (geodesic in Euclidean space)"
}
```

### Example 30: Dynamic Programming

**User:** "Set up the Bellman equation for a simple inventory problem"

**Claude with ReasonForge:**
```
Using dynamic_programming tool from reasonforge-algebra...

Result:
{
  "value_function": "V(s)",
  "state_variables": ["s"],
  "decision_variables": ["a"],
  "transition": "s' = s - d + a",
  "bellman_equation": "V(s) = min_a [c*a + h*s + E[V(s - d + a)]]",
  "optimal_policy": "a* = argmin_a [c*a + h*s + E[V(s - d + a)]]"
}
```

## Other (3 tools)

### Example 31: Recognize Pattern

**User:** "What's the pattern in this sequence: 1, 4, 9, 16, 25, 36?"

**Claude with ReasonForge:**
```
Using recognize_pattern tool from reasonforge-algebra...

Result:
{
  "sequence": [1, 4, 9, 16, 25, 36],
  "patterns_found": 1,
  "most_likely": {
    "type": "polynomial",
    "formula": "n**2",
    "next_terms": ["49", "64", "81"]
  }
}
```

### Example 32: Differentiate Expression

**User:** "Find the second derivative of x^4 with respect to x"

**Claude with ReasonForge:**
```
Using differentiate_expression tool from reasonforge-algebra...

Result:
{
  "expression_key": "expr_10",
  "variable": "x",
  "order": 2,
  "original": "x**4",
  "result": "12*x**2",
  "latex": "12 x^{2}",
  "stored_key": "expr_11"
}
```

### Example 33: Integrate Expression

**User:** "Integrate the stored expression x^3 from 0 to 2"

**Claude with ReasonForge:**
```
Using integrate_expression tool from reasonforge-algebra...

Result:
{
  "expression_key": "expr_8",
  "variable": "x",
  "bounds": [0, 2],
  "indefinite": "x**4/4",
  "definite": "4",
  "stored_key": "expr_9"
}
```

---

[View all packages](../../USAGE_EXAMPLES.md) | [Prev: reasonforge-expressions](./reasonforge-expressions-examples.md) | [Next: reasonforge-analysis](./reasonforge-analysis-examples.md)
