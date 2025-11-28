# MCP Base Server Guide

This guide explains how to use `BaseReasonForgeServer` to eliminate code duplication and create cleaner MCP servers.

## Benefits

Using `BaseReasonForgeServer` provides:

✅ **40% less code** - Eliminates boilerplate across all servers
✅ **Centralized error handling** - Consistent error responses
✅ **Built-in validation** - Automatic `ValidationError` handling
✅ **Type safety** - Better IDE support and fewer bugs
✅ **Easier maintenance** - Update once, apply everywhere

## Quick Start

### Old Way (654 lines of boilerplate)

```python
# packages/reasonforge-expressions/src/reasonforge_expressions/server.py
import json
import sys
from typing import Any
import sympy as sp
from reasonforge import SymbolicAI
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("reasonforge-expressions")
ai = SymbolicAI()

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="intro",
            description="Introduce a variable",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        ),
        # ... 14 more tools
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        if name == "intro":
            var_name = arguments.get("name")
            var = ai.define_variables(var_name)[0]
            return [TextContent(
                type="text",
                text=json.dumps({
                    "name": var_name,
                    "variable": str(var)
                })
            )]
        elif name == "tool2":
            # ... implementation
        # ... 13 more elif branches
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Error: {str(e)}"})
        )]

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    asyncio.run(main())
```

### New Way (Clean and Simple)

```python
# packages/reasonforge-expressions/src/reasonforge_expressions/server.py
from reasonforge import BaseReasonForgeServer, SymbolicAI, create_input_schema
from reasonforge import validate_variable_name, ValidationError


class ExpressionsServer(BaseReasonForgeServer):
    """MCP server for expression manipulation."""

    def __init__(self):
        super().__init__("reasonforge-expressions")
        self.ai = SymbolicAI()

    def register_tools(self):
        """Register all expression tools."""
        self.add_tool(
            name="intro",
            description="Introduce a variable",
            handler=self.handle_intro,
            input_schema=create_input_schema(
                properties={
                    "name": {"type": "string", "description": "Variable name"}
                },
                required=["name"]
            )
        )
        # ... register other tools

    def handle_intro(self, arguments: dict) -> dict:
        """Handle the intro tool."""
        var_name = validate_variable_name(arguments.get("name"))
        var = self.ai.define_variables(var_name)[0]

        return {
            "name": var_name,
            "variable": str(var),
            "latex": f"${var_name}$"
        }


# Entry point
if __name__ == "__main__":
    server = ExpressionsServer()
    server.run()
```

**Result**: 40% less code, automatic error handling, built-in validation!

## Complete Example

Here's a complete example showing all features:

```python
from reasonforge import (
    BaseReasonForgeServer,
    SymbolicAI,
    create_input_schema,
    safe_sympify,
    validate_variable_name,
    ValidationError,
)


class MyServer(BaseReasonForgeServer):
    """Example MCP server using base class."""

    def __init__(self):
        # Initialize base with server name
        super().__init__("reasonforge-myserver")

        # Initialize your state
        self.ai = SymbolicAI()
        self.custom_state = {}

    def register_tools(self):
        """Register all tools - called automatically during init."""

        # Simple tool
        self.add_tool(
            name="simple_tool",
            description="A simple tool",
            handler=self.handle_simple,
            input_schema=create_input_schema(
                properties={
                    "value": {
                        "type": "string",
                        "description": "Some value"
                    }
                },
                required=["value"]
            )
        )

        # Complex tool with multiple parameters
        self.add_tool(
            name="complex_tool",
            description="A more complex tool",
            handler=self.handle_complex,
            input_schema=create_input_schema(
                properties={
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression"
                    },
                    "variable": {
                        "type": "string",
                        "description": "Variable name"
                    },
                    "options": {
                        "type": "object",
                        "description": "Optional parameters",
                        "properties": {
                            "simplify": {
                                "type": "boolean",
                                "default": True
                            }
                        }
                    }
                },
                required=["expression", "variable"]
            )
        )

    def handle_simple(self, arguments: dict) -> dict:
        """
        Handle simple_tool.

        Args:
            arguments: Dictionary of tool arguments

        Returns:
            Dictionary response (automatically converted to JSON)

        Raises:
            ValidationError: Automatically caught and returned as error response
        """
        value = arguments.get("value")

        return {
            "result": f"Processed: {value}",
            "timestamp": "2024-01-01"
        }

    def handle_complex(self, arguments: dict) -> dict:
        """Handle complex_tool with validation."""

        # Validate inputs (raises ValidationError on failure)
        expr_str = arguments.get("expression")
        var_name = validate_variable_name(arguments.get("variable"))
        options = arguments.get("options", {})

        # Parse expression safely
        expr = safe_sympify(expr_str)
        var = self.ai.define_variables(var_name)[0]

        # Process
        if options.get("simplify", True):
            expr = sp.simplify(expr)

        return {
            "expression": str(expr),
            "variable": var_name,
            "latex": sp.latex(expr),
            "simplified": options.get("simplify", True)
        }


# Entry point
if __name__ == "__main__":
    server = MyServer()
    server.run()
```

## Migration Guide

### Step 1: Import Base Class

```python
# Before
from mcp.server import Server
from mcp.types import Tool, TextContent

# After
from reasonforge import BaseReasonForgeServer, create_input_schema
```

### Step 2: Create Server Class

```python
# Before
server = Server("reasonforge-myserver")
ai = SymbolicAI()

# After
class MyServer(BaseReasonForgeServer):
    def __init__(self):
        super().__init__("reasonforge-myserver")
        self.ai = SymbolicAI()
```

### Step 3: Move Tool Definitions to register_tools()

```python
# Before
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="Does something",
            inputSchema={"type": "object", ...}
        )
    ]

# After
def register_tools(self):
    self.add_tool(
        name="my_tool",
        description="Does something",
        handler=self.handle_my_tool,
        input_schema=create_input_schema(...)
    )
```

### Step 4: Extract Tool Handlers

```python
# Before
@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    try:
        if name == "my_tool":
            result = # ... implementation
            return [TextContent(
                type="text",
                text=json.dumps(result)
            )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

# After
def handle_my_tool(self, arguments: dict) -> dict:
    # Just return dict - JSON conversion is automatic
    return {
        "result": "success"
    }
```

### Step 5: Update Entry Point

```python
# Before
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(...)

    asyncio.run(main())

# After
if __name__ == "__main__":
    server = MyServer()
    server.run()
```

## Error Handling

The base class provides automatic error handling:

### Validation Errors

```python
def handle_tool(self, arguments: dict) -> dict:
    # Validation errors are automatically caught
    var_name = validate_variable_name(arguments.get("name"))  # May raise ValidationError

    return {"result": "success"}

# User gets:
# {
#   "error": "Validation error in tool_name: Invalid variable name",
#   "tool": "tool_name",
#   "error_type": "validation"
# }
```

### Other Exceptions

```python
def handle_tool(self, arguments: dict) -> dict:
    # Other exceptions are caught too
    result = 1 / 0  # ZeroDivisionError

    return {"result": result}

# User gets:
# {
#   "error": "Error executing tool_name: division by zero",
#   "tool": "tool_name",
#   "error_type": "ZeroDivisionError"
# }
```

### Manual Error Responses

```python
def handle_tool(self, arguments: dict) -> dict:
    # You can return errors manually too
    if not some_condition:
        return {"error": "Custom error message"}

    return {"result": "success"}
```

## Best Practices

### 1. One Handler Per Tool

```python
# ✅ Good - Each tool has its own handler
def register_tools(self):
    self.add_tool("tool1", "...", self.handle_tool1, {...})
    self.add_tool("tool2", "...", self.handle_tool2, {...})

def handle_tool1(self, arguments: dict) -> dict:
    ...

def handle_tool2(self, arguments: dict) -> dict:
    ...
```

### 2. Validate All Inputs

```python
# ✅ Good - Validate before processing
def handle_tool(self, arguments: dict) -> dict:
    expr_str = arguments.get("expression")
    var_name = validate_variable_name(arguments.get("variable"))

    expr = safe_sympify(expr_str)
    var = self.ai.define_variables(var_name)[0]

    return {"result": str(expr.subs(var, 0))}
```

### 3. Use create_input_schema Helper

```python
# ✅ Good - Use helper for cleaner schemas
input_schema = create_input_schema(
    properties={
        "expression": {"type": "string"},
        "variable": {"type": "string"}
    },
    required=["expression", "variable"]
)

# ❌ Avoid - Manual schema construction (error-prone)
input_schema = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"},
        "variable": {"type": "string"}
    },
    "required": ["expression", "variable"]
}
```

### 4. Keep Handlers Simple

```python
# ✅ Good - Handler focuses on business logic
def handle_solve(self, arguments: dict) -> dict:
    equation = safe_sympify(arguments.get("equation"))
    var_name = validate_variable_name(arguments.get("variable"))

    var = self.ai.define_variables(var_name)[0]
    solution = sp.solve(equation, var)

    return {
        "equation": str(equation),
        "variable": var_name,
        "solution": str(solution)
    }

# ❌ Avoid - Too much error handling (automatic now)
def handle_solve(self, arguments: dict) -> dict:
    try:
        equation = safe_sympify(arguments.get("equation"))
        try:
            var_name = validate_variable_name(arguments.get("variable"))
            # ... more nested try-except
        except ValidationError as e:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
```

## Testing

Test your server using the base class:

```python
import pytest
from my_server import MyServer


@pytest.fixture
def server():
    return MyServer()


def test_tool_registration(server):
    """Test that tools are registered."""
    assert "my_tool" in server.tools
    assert server.tools["my_tool"].name == "my_tool"


@pytest.mark.asyncio
async def test_tool_execution(server):
    """Test tool execution."""
    # Access the server's MCP server
    result = await server.server.call_tool(
        "my_tool",
        {"argument": "value"}
    )

    # Check response
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "result" in data
```

## Troubleshooting

### ImportError: cannot import name 'BaseReasonForgeServer'

**Solution**: Make sure reasonforge package is installed:
```bash
pip install -e packages/reasonforge
```

### AttributeError: 'MyServer' object has no attribute 'tools'

**Solution**: Make sure you call `super().__init__()`:
```python
def __init__(self):
    super().__init__("my-server-name")  # Don't forget this!
    self.ai = SymbolicAI()
```

### Tool handlers not being called

**Solution**: Make sure you're registering tools in `register_tools()`:
```python
def register_tools(self):  # Must be named exactly this
    self.add_tool(...)
```

## Full Migration Checklist

When migrating a server, ensure:

- [ ] Import `BaseReasonForgeServer` instead of `Server`
- [ ] Create class inheriting from `BaseReasonForgeServer`
- [ ] Call `super().__init__(server_name)` in `__init__()`
- [ ] Implement `register_tools()` method
- [ ] Move each tool from `list_tools()` to `add_tool()` call
- [ ] Extract each `if name == "..."` branch to its own handler method
- [ ] Remove `@server.list_tools()` decorator
- [ ] Remove `@server.call_tool()` decorator
- [ ] Remove manual `TextContent` creation (return dict instead)
- [ ] Remove manual `json.dumps()` (automatic now)
- [ ] Update entry point to `server = MyServer(); server.run()`
- [ ] Test all tools still work
- [ ] Update tests to use new class-based structure

## Further Reading

- [validation.py](./validation.py) - Input validation utilities
- [VALIDATION.md](./VALIDATION.md) - Validation guide
- [../SECURITY.md](../../SECURITY.md) - Security best practices
