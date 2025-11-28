# Input Validation Guide

This document describes how to use the input validation utilities in ReasonForge to prevent security vulnerabilities.

## Overview

All user inputs should be validated before processing to prevent:
- **Code execution attacks** via `sympify()`
- **Denial of service** via extremely large inputs
- **Injection attacks** via malicious expressions

## Quick Start

```python
from reasonforge import safe_sympify, validate_variable_name, ValidationError

# Validate and parse expressions
try:
    expr = safe_sympify("x**2 + 2*x + 1")  # Safe
    name = validate_variable_name("x")      # Safe
except ValidationError as e:
    # Handle validation error
    return {"error": str(e)}
```

## Available Functions

### `safe_sympify(expr, locals_dict=None, strict=True)`

Safely parse expressions with validation.

```python
from reasonforge import safe_sympify, ValidationError

# Safe usage
try:
    expr = safe_sympify("x**2 + sin(y)")
    # Process expr...
except ValidationError as e:
    return {"error": f"Invalid expression: {e}"}

# With local variables
x, y = symbols('x y')
expr = safe_sympify("2*x + y", locals_dict={'x': x, 'y': y})
```

**Blocks:**
- Dunder methods (`__import__`, `__class__`, etc.)
- Dangerous functions (`eval`, `exec`, `compile`, `open`)
- Import statements
- Lambda expressions
- Attribute access (`getattr`, `setattr`)

### `validate_variable_name(name)`

Validate variable names before creating symbols.

```python
from reasonforge import validate_variable_name, ValidationError

# Safe usage
try:
    name = validate_variable_name("x")  # OK
    name = validate_variable_name("my_var")  # OK
    name = validate_variable_name("__import__")  # Raises ValidationError
except ValidationError as e:
    return {"error": f"Invalid variable name: {e}"}
```

**Checks:**
- Valid Python identifier
- Length limits (≤100 chars)
- No dangerous patterns

### `validate_expression_string(expr)`

Validate expression strings before parsing.

```python
from reasonforge import validate_expression_string, ValidationError

try:
    expr_str = validate_expression_string("x**2 + 1")  # OK
    # Then parse with sympify or safe_sympify
except ValidationError as e:
    return {"error": str(e)}
```

### `validate_list_input(data, max_length=1000)`

Validate list inputs.

```python
from reasonforge import validate_list_input, ValidationError

try:
    coeffs = validate_list_input([1, 2, 3, 4])  # OK
    huge_list = validate_list_input(range(10000))  # Raises ValidationError
except ValidationError as e:
    return {"error": str(e)}
```

### `validate_dict_input(data, required_keys=None)`

Validate dictionary inputs.

```python
from reasonforge import validate_dict_input, ValidationError

try:
    params = validate_dict_input({"x": 1, "y": 2})  # OK
    params = validate_dict_input(
        {"x": 1},
        required_keys=["x", "y"]  # Raises ValidationError (missing 'y')
    )
except ValidationError as e:
    return {"error": str(e)}
```

### `validate_numeric_input(value, allow_symbolic=False)`

Validate numeric inputs.

```python
from reasonforge import validate_numeric_input, ValidationError

try:
    x = validate_numeric_input(3.14)  # OK: float
    x = validate_numeric_input("3.14")  # OK: parses to float
    x = validate_numeric_input("pi", allow_symbolic=True)  # OK: symbolic
    x = validate_numeric_input("abc")  # Raises ValidationError
except ValidationError as e:
    return {"error": str(e)}
```

### `validate_matrix_dimensions(rows, cols, max_size=100)`

Validate matrix dimensions.

```python
from reasonforge import validate_matrix_dimensions, ValidationError

try:
    rows, cols = validate_matrix_dimensions(3, 3)  # OK
    rows, cols = validate_matrix_dimensions(1000, 1000)  # Raises ValidationError
except ValidationError as e:
    return {"error": str(e)}
```

### `validate_key_format(key, prefix="")`

Validate storage keys.

```python
from reasonforge import validate_key_format, ValidationError

try:
    key = validate_key_format("expr_0")  # OK
    key = validate_key_format("matrix_5", prefix="matrix_")  # OK
    key = validate_key_format("invalid-key")  # Raises ValidationError
except ValidationError as e:
    return {"error": str(e)}
```

### `sanitize_latex_input(latex_str)`

Sanitize LaTeX strings.

```python
from reasonforge import sanitize_latex_input, ValidationError

try:
    latex = sanitize_latex_input(r"\frac{x}{y}")  # OK
    latex = sanitize_latex_input(r"\input{/etc/passwd}")  # Raises ValidationError
except ValidationError as e:
    return {"error": str(e)}
```

## Best Practices for MCP Server Tools

### 1. Always validate user input first

```python
@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        if name == "intro":
            # ❌ UNSAFE
            var = sp.symbols(arguments.get("name"))

            # ✅ SAFE
            var_name = validate_variable_name(arguments.get("name"))
            var = sp.symbols(var_name)
```

### 2. Use safe_sympify instead of sp.sympify

```python
# ❌ UNSAFE - Direct sympify
expression = sp.sympify(arguments.get("expression"))

# ✅ SAFE - Validated sympify
from reasonforge import safe_sympify, ValidationError

try:
    expression = safe_sympify(arguments.get("expression"))
except ValidationError as e:
    return [TextContent(
        type="text",
        text=json.dumps({"error": f"Invalid expression: {str(e)}"})
    )]
```

### 3. Validate all list/dict inputs

```python
# ❌ UNSAFE
values = arguments.get("values")
for v in values:
    # Process without validation

# ✅ SAFE
from reasonforge import validate_list_input, ValidationError

try:
    values = validate_list_input(arguments.get("values"))
    for v in values:
        # Safe to process
except ValidationError as e:
    return [TextContent(
        type="text",
        text=json.dumps({"error": str(e)})
    )]
```

### 4. Wrap tool implementation in try-except

```python
@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        if name == "solve_equation":
            # Validate inputs
            equation_str = validate_expression_string(arguments.get("equation"))
            var_name = validate_variable_name(arguments.get("variable"))

            # Parse safely
            equation = safe_sympify(equation_str)
            var = sp.symbols(var_name)

            # Solve
            solution = sp.solve(equation, var)

            return [TextContent(
                type="text",
                text=json.dumps({"solution": str(solution)})
            )]

    except ValidationError as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Validation failed: {str(e)}"})
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Error: {str(e)}"})
        )]
```

## Migration Checklist

When refactoring existing servers, ensure:

- [ ] Replace all `sp.sympify()` with `safe_sympify()`
- [ ] Replace all `sp.symbols()` with validated variable names
- [ ] Validate all list inputs with `validate_list_input()`
- [ ] Validate all dict inputs with `validate_dict_input()`
- [ ] Validate numeric inputs with `validate_numeric_input()`
- [ ] Wrap all tool implementations in try-except blocks
- [ ] Return proper error messages for ValidationError

## Security Notes

### Why This Matters

SymPy's `sympify()` function can execute arbitrary Python code:

```python
# ⚠️ DANGEROUS - Can execute code
sympify("__import__('os').system('rm -rf /')")
sympify("open('/etc/passwd').read()")
sympify("eval('malicious code')")
```

The `safe_sympify()` function prevents these attacks by:
1. Checking for dangerous patterns before parsing
2. Using `sympify(evaluate=False)` to prevent auto-execution
3. Enforcing length limits to prevent DoS

### Attack Vectors Prevented

1. **Code Execution**: `__import__`, `eval`, `exec`, `compile`
2. **File Access**: `open`, `read`, `write`
3. **Attribute Access**: `getattr`, `setattr`, `__dict__`
4. **Denial of Service**: Length limits on all inputs
5. **LaTeX Injection**: Sanitization of LaTeX commands

## Testing

Test validation in your server tools:

```python
import pytest
from reasonforge import safe_sympify, ValidationError

def test_safe_sympify_blocks_dangerous_input():
    # Should raise ValidationError
    with pytest.raises(ValidationError):
        safe_sympify("__import__('os')")

    with pytest.raises(ValidationError):
        safe_sympify("eval('1+1')")

def test_safe_sympify_allows_valid_input():
    # Should work fine
    expr = safe_sympify("x**2 + 2*x + 1")
    assert str(expr) == "x**2 + 2*x + 1"
```

## Further Reading

- [SymPy Security Documentation](https://docs.sympy.org/latest/modules/core.html#module-sympy.core.sympify)
- [Python Code Injection Attacks](https://owasp.org/www-community/attacks/Code_Injection)
- [Input Validation Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)
