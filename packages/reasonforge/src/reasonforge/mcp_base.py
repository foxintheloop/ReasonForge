"""
Base MCP Server implementation for ReasonForge servers.

This module provides a base class that eliminates code duplication across
all ReasonForge MCP server packages.
"""

import json
from typing import Any, Callable, Dict, List, Optional
from abc import ABC, abstractmethod

from mcp.server import Server
from mcp.types import Tool, TextContent

from .validation import ValidationError


class ToolHandler:
    """Represents a single tool handler with metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        input_schema: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.handler = handler
        self.input_schema = input_schema

    def to_tool(self) -> Tool:
        """Convert to MCP Tool definition."""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
        )


class BaseReasonForgeServer(ABC):
    """
    Base class for all ReasonForge MCP servers.

    This class provides:
    - Standard server initialization
    - Tool registration mechanism
    - Centralized error handling
    - Validation integration
    - JSON response formatting

    Usage:
        class MyServer(BaseReasonForgeServer):
            def __init__(self):
                super().__init__("reasonforge-myserver")
                self.ai = SymbolicAI()

            def register_tools(self):
                self.add_tool(
                    name="my_tool",
                    description="Does something",
                    handler=self.handle_my_tool,
                    input_schema={...}
                )

            def handle_my_tool(self, arguments: dict) -> dict:
                # Tool implementation
                return {"result": "success"}
    """

    def __init__(self, server_name: str):
        """
        Initialize the base server.

        Args:
            server_name: Name of the MCP server (e.g., "reasonforge-expressions")
        """
        self.server = Server(server_name)
        self.server_name = server_name
        self.tools: Dict[str, ToolHandler] = {}

        # Set up server handlers
        self._setup_handlers()

        # Let subclass register its tools
        self.register_tools()

    @abstractmethod
    def register_tools(self):
        """
        Register all tools for this server.

        Subclasses must implement this method to register their tools using add_tool().
        """
        pass

    def add_tool(
        self,
        name: str,
        description: str,
        handler: Callable[[dict], dict],
        input_schema: Dict[str, Any],
    ):
        """
        Register a tool handler.

        Args:
            name: Tool name
            description: Tool description
            handler: Function that handles the tool (takes dict, returns dict)
            input_schema: JSON schema for tool inputs
        """
        tool_handler = ToolHandler(name, description, handler, input_schema)
        self.tools[name] = tool_handler

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools."""
            return [tool.to_tool() for tool in self.tools.values()]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """
            Call a tool with validation and error handling.

            This method provides:
            - Tool lookup
            - Input validation
            - Error handling
            - JSON response formatting
            """
            try:
                # Validate tool exists
                if name not in self.tools:
                    return self._error_response(
                        f"Unknown tool: {name}",
                        tool_name=name
                    )

                # Get tool handler
                tool_handler = self.tools[name]

                # Call handler with arguments
                result = tool_handler.handler(arguments)

                # Ensure result is a dict
                if not isinstance(result, dict):
                    return self._error_response(
                        f"Tool {name} returned non-dict result",
                        tool_name=name
                    )

                # Return success response
                return self._success_response(result)

            except ValidationError as e:
                # Validation errors - safe to return details
                return self._error_response(
                    f"Validation error in {name}: {str(e)}",
                    tool_name=name,
                    error_type="validation"
                )

            except Exception as e:
                # Other errors - be cautious about leaking information
                return self._error_response(
                    f"Error executing {name}: {str(e)}",
                    tool_name=name,
                    error_type=type(e).__name__
                )

    def _success_response(self, data: dict) -> List[TextContent]:
        """
        Create a success response.

        Args:
            data: Response data dictionary

        Returns:
            List containing a single TextContent with JSON data
        """
        return [
            TextContent(
                type="text",
                text=json.dumps(data, indent=2)
            )
        ]

    def _error_response(
        self,
        message: str,
        tool_name: Optional[str] = None,
        error_type: Optional[str] = None
    ) -> List[TextContent]:
        """
        Create an error response.

        Args:
            message: Error message
            tool_name: Name of the tool that errored (optional)
            error_type: Type of error (optional)

        Returns:
            List containing a single TextContent with error JSON
        """
        error_data = {
            "error": message,
        }

        if tool_name:
            error_data["tool"] = tool_name

        if error_type:
            error_data["error_type"] = error_type

        return [
            TextContent(
                type="text",
                text=json.dumps(error_data, indent=2)
            )
        ]

    async def call_tool_for_test(self, name: str, arguments: Any) -> List[TextContent]:
        """
        Call a tool for testing purposes.

        This method bypasses the MCP framework and directly calls the tool handler,
        making it easy to test tools without needing to set up the full MCP infrastructure.

        Args:
            name: Tool name
            arguments: Tool arguments dictionary

        Returns:
            List of TextContent with the response

        Example:
            >>> server = MyServer()
            >>> result = await server.call_tool_for_test("my_tool", {"arg": "value"})
            >>> data = json.loads(result[0].text)
        """
        try:
            # Validate tool exists
            if name not in self.tools:
                return self._error_response(
                    f"Unknown tool: {name}",
                    tool_name=name
                )

            # Get tool handler
            tool_handler = self.tools[name]

            # Call handler with arguments
            result = tool_handler.handler(arguments)

            # Ensure result is a dict
            if not isinstance(result, dict):
                return self._error_response(
                    f"Tool {name} returned non-dict result",
                    tool_name=name
                )

            # Return success response
            return self._success_response(result)

        except ValidationError as e:
            # Validation errors - safe to return details
            return self._error_response(
                f"Validation error in {name}: {str(e)}",
                tool_name=name,
                error_type="validation"
            )

        except Exception as e:
            # Other errors - be cautious about leaking information
            return self._error_response(
                f"Error executing {name}: {str(e)}",
                tool_name=name,
                error_type=type(e).__name__
            )

    def run(self):
        """
        Run the MCP server.

        This is the entry point for the server.
        """
        import asyncio
        from mcp.server.stdio import stdio_server

        async def main():
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )

        asyncio.run(main())


def create_input_schema(
    properties: Dict[str, Any],
    required: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Helper function to create input schemas.

    Args:
        properties: Dictionary of property definitions
        required: List of required property names

    Returns:
        JSON schema dictionary

    Example:
        schema = create_input_schema(
            properties={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression"
                },
                "variable": {
                    "type": "string",
                    "description": "Variable name"
                }
            },
            required=["expression", "variable"]
        )
    """
    schema = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema
