"""
LLM Provider integrations for benchmarking.

Supports multiple LLM providers with user-provided API keys.
Each provider returns responses in a standardized format for comparison.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str, model: str):
        """
        Initialize LLM provider.

        Args:
            api_key: API key for the provider
            model: Model identifier (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4')
        """
        self.api_key = api_key
        self.model = model
        self.total_cost = 0.0
        self.total_requests = 0

    @abstractmethod
    def query(self, problem: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Send a problem to the LLM and get a response.

        Args:
            problem: The mathematical problem to solve
            max_retries: Maximum number of retry attempts on failure

        Returns:
            Dict containing:
                - answer: The LLM's answer
                - raw_response: Full response text
                - tokens_used: Token count
                - cost: Estimated cost in USD
                - latency_ms: Response time in milliseconds
                - success: Whether the query succeeded
                - error: Error message if failed
        """
        pass

    @abstractmethod
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for this provider based on token usage."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_cost": round(self.total_cost, 4)
        }


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    # Pricing per million tokens (as of Jan 2025)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Anthropic provider."""
        super().__init__(api_key, model)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def query(self, problem: str, max_retries: int = 3) -> Dict[str, Any]:
        """Query Claude with a mathematical problem."""
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=[{
                        "role": "user",
                        "content": f"Solve this mathematical problem. Provide only the final answer in the simplest form possible, without explanations:\n\n{problem}"
                    }]
                )

                latency_ms = int((time.time() - start_time) * 1000)

                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                self.total_cost += cost
                self.total_requests += 1

                answer = message.content[0].text.strip()

                return {
                    "answer": answer,
                    "raw_response": message.content[0].text,
                    "tokens_used": input_tokens + output_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "latency_ms": latency_ms,
                    "success": True,
                    "error": None
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "answer": None,
                        "raw_response": None,
                        "tokens_used": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0,
                        "latency_ms": 0,
                        "success": False,
                        "error": str(e)
                    }
                time.sleep(2 ** attempt)  # Exponential backoff

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Anthropic."""
        pricing = self.PRICING.get(self.model, self.PRICING["claude-3-5-sonnet-20241022"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    # Pricing per million tokens (as of Jan 2025)
    PRICING = {
        "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize OpenAI provider."""
        super().__init__(api_key, model)
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def query(self, problem: str, max_retries: int = 3) -> Dict[str, Any]:
        """Query GPT with a mathematical problem."""
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": f"Solve this mathematical problem. Provide only the final answer in the simplest form possible, without explanations:\n\n{problem}"
                    }],
                    max_tokens=2048
                )

                latency_ms = int((time.time() - start_time) * 1000)

                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cost = self._calculate_cost(input_tokens, output_tokens)

                self.total_cost += cost
                self.total_requests += 1

                answer = response.choices[0].message.content.strip()

                return {
                    "answer": answer,
                    "raw_response": response.choices[0].message.content,
                    "tokens_used": input_tokens + output_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost": cost,
                    "latency_ms": latency_ms,
                    "success": True,
                    "error": None
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "answer": None,
                        "raw_response": None,
                        "tokens_used": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost": 0.0,
                        "latency_ms": 0,
                        "success": False,
                        "error": str(e)
                    }
                time.sleep(2 ** attempt)  # Exponential backoff

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenAI."""
        pricing = self.PRICING.get(self.model, self.PRICING["gpt-4o"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


class ReasonForgeProvider:
    """
    ReasonForge MCP Server provider.

    This is not an LLM but our symbolic AI system.
    Included here for consistent benchmarking interface.
    """

    def __init__(self):
        """Initialize ReasonForge provider."""
        self.total_requests = 0
        self.total_cost = 0.0  # Always 0 for ReasonForge
        self.model = "ReasonForge (SymPy)"

    async def query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a ReasonForge tool and get the result.

        Args:
            test_case: Test case dict from test_cases.py

        Returns:
            Dict containing:
                - answer: The computed answer
                - raw_response: Full response
                - tokens_used: 0 (not applicable)
                - cost: 0.0 (always free)
                - latency_ms: Execution time
                - success: Whether computation succeeded
                - error: Error message if failed
        """
        try:
            start_time = time.time()

            # Import the appropriate server based on tool category
            tool_name = test_case["reasonforge_tool"]
            params = test_case["reasonforge_params"]

            # Dynamically import and call the appropriate tool
            result = await self._call_reasonforge_tool(tool_name, params)

            latency_ms = int((time.time() - start_time) * 1000)
            self.total_requests += 1

            return {
                "answer": str(result),
                "raw_response": result,
                "tokens_used": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "latency_ms": latency_ms,
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "answer": None,
                "raw_response": None,
                "tokens_used": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "latency_ms": 0,
                "success": False,
                "error": str(e)
            }

    async def _call_reasonforge_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Call a ReasonForge tool.

        This is a placeholder that will be implemented by the benchmark runner
        which has access to the MCP servers.
        """
        raise NotImplementedError(
            "ReasonForge tool calling must be implemented by benchmark runner"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "provider": "ReasonForge",
            "model": self.model,
            "total_requests": self.total_requests,
            "total_cost": 0.0
        }


def create_provider(provider_name: str, api_key: Optional[str] = None,
                   model: Optional[str] = None) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_name: 'anthropic', 'openai', or 'reasonforge'
        api_key: API key (not needed for ReasonForge)
        model: Model identifier (uses default if not specified)

    Returns:
        Initialized provider instance
    """
    provider_name = provider_name.lower()

    if provider_name == "anthropic":
        if not api_key:
            raise ValueError("API key required for Anthropic")
        return AnthropicProvider(api_key, model or "claude-3-5-sonnet-20241022")

    elif provider_name == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI")
        return OpenAIProvider(api_key, model or "gpt-4o")

    elif provider_name == "reasonforge":
        return ReasonForgeProvider()

    else:
        raise ValueError(f"Unknown provider: {provider_name}")
