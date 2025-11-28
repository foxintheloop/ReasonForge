# reasonforge-expressions (15 tools)

**Essential package for symbolic manipulation, basic calculus, and expression operations**

**Installation:** `pip install reasonforge-expressions`

[View all packages](../../USAGE_EXAMPLES.md) | [Next: reasonforge-algebra](./reasonforge-algebra-examples.md)

---

## Variable Management (4 tools)

### Example 1: Introduce Variable (intro)

**User:** "Create a positive real variable called r"

**Claude with ReasonForge:**
```
Using intro tool from reasonforge-expressions...

Result:
{
  "name": "r",
  "variable": "r",
  "assumptions": {
    "positive": true,
    "real": true
  },
  "latex": "r"
}
```

### Example 2: Introduce Many Variables (intro_many)

**User:** "Create positive variables x, y, and z"

**Claude with ReasonForge:**
```
Using intro_many tool from reasonforge-expressions...

Result:
{
  "names": ["x", "y", "z"],
  "variables": ["x", "y", "z"],
  "assumptions": {
    "positive": true
  },
  "count": 3
}
```

### Example 3: Introduce Expression

**User:** "Store the expression x^2 + y^2 as 'circle_eq'"

**Claude with ReasonForge:**
```
Using introduce_expression tool from reasonforge-expressions...

Result:
{
  "expression": "x**2 + y**2",
  "key": "circle_eq",
  "stored": true,
  "latex": "x^{2} + y^{2}"
}
```

### Example 4: Introduce Function

**User:** "Define a function f(x) for use in differential equations"

**Claude with ReasonForge:**
```
Using introduce_function tool from reasonforge-expressions...

Result:
{
  "name": "f",
  "function": "f",
  "type": "undefined_function",
  "usage": "f(x)"
}
```

## Expression Operations (5 tools)

### Example 5: Simplify Expression

**User:** "Simplify sin(x)^2 + cos(x)^2"

**Claude with ReasonForge:**
```
Using simplify_expression tool from reasonforge-expressions...

Result:
{
  "expression_key": "expr_5",
  "method": "trigsimp",
  "original": "sin(x)**2 + cos(x)**2",
  "simplified": "1",
  "latex": "1"
}
```

### Example 6: Substitute Expression

**User:** "Substitute x=π into the stored expression sin(2*x)"

**Claude with ReasonForge:**
```
Using substitute_expression tool from reasonforge-expressions...

Result:
{
  "expression_key": "expr_6",
  "substitutions": {"x": "pi"},
  "original": "sin(2*x)",
  "result": "0",
  "stored_key": "expr_7"
}
```

### Example 7: Expand Expression

**User:** "Expand (x + y)^3"

**Claude with ReasonForge:**
```
Using expand_expression tool from reasonforge-expressions...

Result:
{
  "expression": "(x + y)**3",
  "expanded": "x**3 + 3*x**2*y + 3*x*y**2 + y**3",
  "latex": "x^{3} + 3 x^{2} y + 3 x y^{2} + y^{3}"
}
```

### Example 8: Factor Expression

**User:** "Factor x^3 - 6x^2 + 11x - 6"

**Claude with ReasonForge:**
```
Using factor_expression tool from reasonforge-expressions...

Result:
{
  "expression": "x**3 - 6*x**2 + 11*x - 6",
  "factored": "(x - 3)*(x - 2)*(x - 1)",
  "latex": "\\left(x - 3\\right) \\left(x - 2\\right) \\left(x - 1\\right)"
}
```

### Example 9: Substitute Values

**User:** "Substitute x=2 and y=3 into x^2 + 2xy + y^2"

**Claude with ReasonForge:**
```
Using substitute_values tool from reasonforge-expressions...

Result:
{
  "expression": "x**2 + 2*x*y + y**2",
  "substitutions": {"x": "2", "y": "3"},
  "result": "49",
  "latex": "49"
}
```

## Basic Calculus (4 tools)

### Example 10: Differentiate

**User:** "What is the derivative of sin(x) * cos(x) * e^x?"

**Claude with ReasonForge:**
```
Using differentiate tool from reasonforge-expressions...

Result:
{
  "expression": "sin(x)*cos(x)*exp(x)",
  "variable": "x",
  "operation": "diff",
  "result": "(-sin(x)**2 + cos(x)**2)*exp(x) + sin(x)*cos(x)*exp(x)",
  "latex": "\\left(- \\sin^{2}{\\left(x \\right)} + \\cos^{2}{\\left(x \\right)}\\right) e^{x} + \\sin{\\left(x \\right)} \\cos{\\left(x \\right)} e^{x}"
}
```

### Example 11: Integrate

**User:** "Find the integral of x^2 * ln(x) with respect to x"

**Claude with ReasonForge:**
```
Using integrate tool from reasonforge-expressions...

Result:
{
  "expression": "x**2 * log(x)",
  "variable": "x",
  "operation": "integrate",
  "result": "x**3*log(x)/3 - x**3/9",
  "latex": "\\frac{x^{3} \\log{\\left(x \\right)}}{3} - \\frac{x^{3}}{9}",
  "pretty": "  3           3\n x ⋅log(x)   x \n ───────── - ──\n     3       9"
}
```

### Example 12: Compute Limit

**User:** "What is the limit of (sin(x)/x) as x approaches 0?"

**Claude with ReasonForge:**
```
Using compute_limit tool from reasonforge-expressions...

Result:
{
  "expression": "sin(x)/x",
  "variable": "x",
  "point": "zero",
  "limit": "1",
  "latex": "1"
}
```

### Example 13: Expand Series

**User:** "Give me the Taylor series expansion of e^x around x=0 up to order 5"

**Claude with ReasonForge:**
```
Using expand_series tool from reasonforge-expressions...

Result:
{
  "expression": "exp(x)",
  "variable": "x",
  "point": 0,
  "order": 5,
  "series": "1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)",
  "latex": "1 + x + \\frac{x^{2}}{2} + \\frac{x^{3}}{6} + \\frac{x^{4}}{24} + \\frac{x^{5}}{120} + O\\left(x^{6}\\right)"
}
```

## Utilities (2 tools)

### Example 14: Print LaTeX Expression

**User:** "Show me the LaTeX for the stored expression 'circle_eq'"

**Claude with ReasonForge:**
```
Using print_latex_expression tool from reasonforge-expressions...

Result:
{
  "key": "circle_eq",
  "expression": "x**2 + y**2",
  "latex": "x^{2} + y^{2}",
  "assumptions": {
    "x": "positive, real",
    "y": "positive, real"
  }
}
```

### Example 15: Solve Word Problem

**User:** "A rectangle has perimeter 20 and area 24. Find its dimensions."

**Claude with ReasonForge:**
```
Using solve_word_problem tool from reasonforge-expressions...

Result:
{
  "problem": "Rectangle with perimeter 20 and area 24",
  "equations": ["2*l + 2*w - 20", "l*w - 24"],
  "unknowns": ["l", "w"],
  "solutions": [
    {"l": "4", "w": "6"},
    {"l": "6", "w": "4"}
  ],
  "interpretation": "The rectangle has dimensions 4 by 6 (or equivalently 6 by 4)"
}
```

---

[View all packages](../../USAGE_EXAMPLES.md) | [Next: reasonforge-algebra](./reasonforge-algebra-examples.md)
