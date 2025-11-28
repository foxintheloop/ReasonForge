"""
Stateful Test Runner - Handles multi-step tests with setup/teardown.

This module extends the benchmark framework to support stateful tools that
require pre-configuration before they can be tested.
"""

import json
import time
from typing import Dict, List, Any, Optional


class StatefulTestRunner:
    """Runner for multi-step stateful tests with setup and teardown."""

    def __init__(self, server):
        """
        Initialize stateful test runner.

        Args:
            server: MCP server instance to test against
        """
        self.server = server
        self.state_created = []  # Track created state for cleanup
        self.setup_results = []  # Store setup results for debugging

    async def run_stateful_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a test with setup and teardown phases.

        Args:
            test_case: Test case dictionary with setup_steps

        Returns:
            Result dictionary with success status, answer, and metadata
        """
        start_time = time.time()

        # Phase 1: Setup
        setup_steps = test_case.get("setup_steps", [])
        if not await self._execute_setup(setup_steps):
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "answer": None,
                "raw_response": None,
                "tokens_used": 0,
                "cost": 0.0,
                "latency_ms": latency_ms,
                "success": False,
                "error": "Setup phase failed",
                "setup_results": self.setup_results
            }

        # Phase 2: Main Test
        try:
            tool_name = test_case["reasonforge_tool"]
            params = test_case["reasonforge_params"]
            response_field = test_case.get("response_field", "result")

            # Call the tool
            result = await self.server.call_tool_for_test(tool_name, params)
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content from result
            if isinstance(result, list) and len(result) > 0:
                content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            else:
                content = str(result)

            # Parse JSON and extract answer using response_field
            answer = self._extract_answer(content, response_field)

            return {
                "answer": answer,
                "raw_response": content,
                "tokens_used": 0,
                "cost": 0.0,
                "latency_ms": latency_ms,
                "success": True,
                "error": None,
                "setup_results": self.setup_results
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "answer": None,
                "raw_response": None,
                "tokens_used": 0,
                "cost": 0.0,
                "latency_ms": latency_ms,
                "success": False,
                "error": f"Test execution failed: {str(e)}",
                "setup_results": self.setup_results
            }

        finally:
            # Phase 3: Teardown (cleanup)
            teardown_steps = test_case.get("teardown_steps", [])
            await self._execute_teardown(teardown_steps)

    async def _execute_setup(self, setup_steps: List[Dict[str, Any]]) -> bool:
        """
        Execute setup steps to prepare test state.

        Args:
            setup_steps: List of setup step dictionaries

        Returns:
            True if all setup steps succeeded, False otherwise
        """
        self.setup_results = []

        for i, step in enumerate(setup_steps):
            try:
                tool_name = step["tool"]
                params = step["params"]

                # Execute setup step
                result = await self.server.call_tool_for_test(tool_name, params)

                # Extract content
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].text if hasattr(result[0], 'text') else str(result[0])
                else:
                    content = str(result)

                # Try to parse as JSON
                try:
                    parsed_result = json.loads(content)
                    self.setup_results.append({
                        "step": i,
                        "tool": tool_name,
                        "success": True,
                        "result": parsed_result
                    })

                    # Track created keys for potential cleanup
                    if "key" in parsed_result:
                        self.state_created.append({
                            "type": tool_name,
                            "key": parsed_result["key"]
                        })

                except json.JSONDecodeError:
                    # Not JSON, just store raw content
                    self.setup_results.append({
                        "step": i,
                        "tool": tool_name,
                        "success": True,
                        "result": content
                    })

            except Exception as e:
                # Setup failed
                self.setup_results.append({
                    "step": i,
                    "tool": step.get("tool", "unknown"),
                    "success": False,
                    "error": str(e)
                })
                return False

        return True

    async def _execute_teardown(self, teardown_steps: List[Dict[str, Any]]):
        """
        Execute teardown steps to clean up test state.

        Args:
            teardown_steps: List of teardown step dictionaries
        """
        for step in teardown_steps:
            try:
                tool_name = step["tool"]
                params = step["params"]
                await self.server.call_tool_for_test(tool_name, params)
            except Exception:
                # Ignore teardown errors - state will be cleared anyway
                pass

        # Clear tracking
        self.state_created = []

    def _extract_answer(self, content: str, response_field: str) -> str:
        """
        Extract answer from tool response using response_field.

        Args:
            content: Raw response content
            response_field: Field name to extract from JSON

        Returns:
            Extracted answer as string
        """
        try:
            parsed = json.loads(content)

            # Extract the value from the specified field
            if response_field in parsed:
                result_value = parsed[response_field]
            else:
                # Field not found, try to use the entire parsed content
                result_value = parsed

            # Convert result_value to string for validation
            if isinstance(result_value, dict):
                # For dict, check if it has an "expression" field (common pattern)
                if "expression" in result_value:
                    return str(result_value["expression"])
                else:
                    # Convert entire dict to JSON string
                    return json.dumps(result_value)
            elif isinstance(result_value, list):
                # Convert list to JSON string
                return json.dumps(result_value)
            elif isinstance(result_value, bool):
                # Convert boolean to string (True/False)
                return str(result_value)
            elif isinstance(result_value, (int, float)):
                # Convert number to string
                return str(result_value)
            elif result_value is None:
                # None value means field was null
                return "None"
            else:
                # String or other type
                return str(result_value)

        except (json.JSONDecodeError, ValueError):
            # Not JSON, use raw content as answer
            return content

    async def cleanup(self):
        """
        Clean up all created state.

        This resets the server's AI instance to clear all stateful objects
        created during testing.
        """
        # Note: We can't directly reset the AI instance from here
        # The server will handle cleanup between test runs
        self.state_created = []
        self.setup_results = []
