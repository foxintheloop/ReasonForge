"""
Benchmark Runner - Orchestrates performance testing of ReasonForge vs LLMs.

This module runs mathematical test cases against both ReasonForge and
configured LLM providers, validates results, and generates comparison reports.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-expressions" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-algebra" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-analysis" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-geometry" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-statistics" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-physics" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "reasonforge-logic" / "src"))

from test_cases import TEST_CASES, get_test_case_count
from llm_providers import create_provider, ReasonForgeProvider
from metrics import validate_answer, BenchmarkMetrics
from stateful_test_runner import StatefulTestRunner


class BenchmarkRunner:
    """Orchestrates benchmark execution across providers."""

    def __init__(self, config_path: str = "benchmarks/config.yaml"):
        """
        Initialize benchmark runner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.metrics = BenchmarkMetrics()
        self.providers = {}
        self.reasonforge_servers = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)

        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}")
            print("Using default configuration (ReasonForge only)")
            return {
                "providers": {
                    "reasonforge": {"enabled": True}
                },
                "test_selection": {
                    "categories": None,  # None means all
                    "difficulties": None,  # None means all
                    "test_ids": None  # None means all
                }
            }

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def _initialize_providers(self):
        """Initialize all enabled LLM providers."""
        providers_config = self.config.get("providers", {})

        for provider_name, provider_config in providers_config.items():
            if not provider_config.get("enabled", False):
                continue

            if provider_name == "reasonforge":
                self.providers["reasonforge"] = await self._initialize_reasonforge()
                print("[OK] ReasonForge initialized")

            elif provider_name == "anthropic":
                api_key = provider_config.get("api_key")
                model = provider_config.get("model", "claude-3-5-sonnet-20241022")

                if not api_key:
                    print(f"[!] Skipping Anthropic: No API key provided")
                    continue

                try:
                    self.providers["anthropic"] = create_provider("anthropic", api_key, model)
                    print(f"[OK] Anthropic initialized ({model})")
                except Exception as e:
                    print(f"[!] Failed to initialize Anthropic: {e}")

            elif provider_name == "openai":
                api_key = provider_config.get("api_key")
                model = provider_config.get("model", "gpt-4o")

                if not api_key:
                    print(f"[!] Skipping OpenAI: No API key provided")
                    continue

                try:
                    self.providers["openai"] = create_provider("openai", api_key, model)
                    print(f"[OK] OpenAI initialized ({model})")
                except Exception as e:
                    print(f"[!] Failed to initialize OpenAI: {e}")

    async def _initialize_reasonforge(self) -> Dict[str, Any]:
        """Initialize all ReasonForge MCP servers."""
        from reasonforge_expressions.server import server as expressions_server
        from reasonforge_algebra.server import server as algebra_server
        from reasonforge_analysis.server import server as analysis_server
        from reasonforge_geometry.server import server as geometry_server
        from reasonforge_statistics.server import server as statistics_server
        from reasonforge_physics.server import server as physics_server
        from reasonforge_logic.server import server as logic_server

        self.reasonforge_servers = {
            "expressions": expressions_server,
            "algebra": algebra_server,
            "analysis": analysis_server,
            "geometry": geometry_server,
            "statistics": statistics_server,
            "physics": physics_server,
            "logic": logic_server
        }

        # Create a wrapper provider
        return {
            "type": "reasonforge",
            "servers": self.reasonforge_servers
        }

    def _get_test_cases(self) -> List[Dict[str, Any]]:
        """Get filtered test cases based on configuration."""
        test_selection = self.config.get("test_selection", {})
        selected_cases = TEST_CASES

        # Filter by categories
        categories = test_selection.get("categories")
        if categories:
            selected_cases = [tc for tc in selected_cases if tc["category"] in categories]

        # Filter by difficulties
        difficulties = test_selection.get("difficulties")
        if difficulties:
            selected_cases = [tc for tc in selected_cases if tc["difficulty"] in difficulties]

        # Filter by specific test IDs
        test_ids = test_selection.get("test_ids")
        if test_ids:
            selected_cases = [tc for tc in selected_cases if tc["id"] in test_ids]

        return selected_cases

    async def _run_reasonforge_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a test case against ReasonForge.

        Args:
            test_case: Test case dictionary

        Returns:
            Result dictionary
        """
        import time

        # Get server by category
        category = test_case["category"]
        server = self.reasonforge_servers.get(category)
        if not server:
            raise ValueError(f"No server found for category {category}")

        # Check if this is a stateful test (has setup_steps)
        if "setup_steps" in test_case and test_case["setup_steps"]:
            # Use StatefulTestRunner for tests with setup
            runner = StatefulTestRunner(server)
            result = await runner.run_stateful_test(test_case)
            await runner.cleanup()
            return result

        # Original stateless test logic
        try:
            tool_name = test_case["reasonforge_tool"]
            params = test_case["reasonforge_params"]
            response_field = test_case.get("response_field", None)  # None instead of "result"

            # Call the tool
            start_time = time.time()
            result = await server.call_tool_for_test(tool_name, params)
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content from result
            if isinstance(result, list) and len(result) > 0:
                content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            else:
                content = str(result)

            # Parse JSON and extract answer using response_field (if specified)
            try:
                parsed = json.loads(content)

                # Only extract specific field if response_field is explicitly set
                if response_field is not None:
                    # Extract the value from the specified field
                    if response_field in parsed:
                        result_value = parsed[response_field]
                    else:
                        # Field not found, try to use the entire parsed content
                        result_value = parsed
                else:
                    # No response_field specified, use entire JSON content as string
                    result_value = content

                # Convert result_value to string for validation
                if isinstance(result_value, dict):
                    # For dict, check if it has an "expression" field (common pattern)
                    if "expression" in result_value:
                        answer = str(result_value["expression"])
                    else:
                        # Convert entire dict to JSON string
                        answer = json.dumps(result_value)
                elif isinstance(result_value, list):
                    # Convert list to JSON string
                    answer = json.dumps(result_value)
                elif isinstance(result_value, bool):
                    # Convert boolean to string (True/False)
                    answer = str(result_value)
                elif isinstance(result_value, (int, float)):
                    # Convert number to string
                    answer = str(result_value)
                elif result_value is None:
                    # None value means field was null
                    answer = "None"
                else:
                    # String or other type
                    answer = str(result_value)

            except (json.JSONDecodeError, ValueError):
                # Not JSON, use raw content as answer
                answer = content

            return {
                "answer": answer,
                "raw_response": content,
                "tokens_used": 0,
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
                "cost": 0.0,
                "latency_ms": 0,
                "success": False,
                "error": str(e)
            }

    async def _run_llm_test(self, provider_name: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a test case against an LLM provider.

        Args:
            provider_name: Name of the provider
            test_case: Test case dictionary

        Returns:
            Result dictionary
        """
        provider = self.providers[provider_name]
        problem = test_case["problem"]

        return provider.query(problem)

    async def run_benchmark(self, verbose: bool = True):
        """
        Run the complete benchmark suite.

        Args:
            verbose: Print progress information
        """
        print("=" * 80)
        print("ReasonForge Benchmark Suite")
        print("=" * 80)
        print()

        # Initialize providers
        print("Initializing providers...")
        await self._initialize_providers()
        print()

        if not self.providers:
            print("Error: No providers enabled. Please configure at least one provider.")
            return

        # Get test cases
        test_cases = self._get_test_cases()
        test_count = len(test_cases)

        if test_count == 0:
            print("Error: No test cases selected.")
            return

        print(f"Running {test_count} test cases across {len(self.providers)} provider(s)...")
        print()

        # Run tests for each provider
        for provider_name in self.providers.keys():
            print(f"Testing {provider_name}...")
            print("-" * 80)

            for i, test_case in enumerate(test_cases, 1):
                # Skip tests marked as SKIP (stateful tools that need setup)
                if test_case.get("expected_answer") == "SKIP":
                    if verbose:
                        print(f"  [{i}/{test_count}] {test_case['id']}: [SKIP] Stateful tool")
                    continue

                if verbose:
                    print(f"  [{i}/{test_count}] {test_case['id']}: {test_case['problem'][:60]}...")

                # Run test
                if provider_name == "reasonforge":
                    result = await self._run_reasonforge_test(test_case)
                else:
                    result = await self._run_llm_test(provider_name, test_case)

                # Validate answer
                if result["success"] and result["answer"]:
                    is_correct, explanation = validate_answer(
                        result["answer"],
                        test_case["expected_answer"],
                        test_case["validation_type"]
                    )
                else:
                    is_correct = False
                    explanation = result.get("error", "No answer provided")

                # Record result
                self.metrics.add_result(
                    test_id=test_case["id"],
                    provider=provider_name,
                    correct=is_correct,
                    latency_ms=result["latency_ms"],
                    cost=result["cost"],
                    explanation=explanation
                )

                if verbose:
                    status = "[PASS]" if is_correct else "[FAIL]"
                    print(f"    {status} {explanation[:80]}")

            print()

        # Print summary
        print("=" * 80)
        print("Benchmark Complete!")
        print("=" * 80)
        print()
        self._print_summary()

    def _print_summary(self):
        """Print benchmark summary statistics."""
        summary = self.metrics.get_summary()

        print(f"Total Tests: {summary['total_tests']}")
        print()

        for provider, stats in summary["providers"].items():
            print(f"{provider}:")
            print(f"  Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.0f}ms")
            print(f"  Total Cost: ${stats['total_cost']:.4f}")
            print()

    def save_results(self, output_path: str = "benchmarks/results"):
        """
        Save benchmark results to JSON file.

        Args:
            output_path: Directory to save results
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"benchmark_results_{timestamp}.json"

        results = {
            "timestamp": timestamp,
            "config": self.config,
            "summary": self.metrics.get_summary(),
            "detailed_results": self.metrics.results
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")
        return results_file

    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results as dictionary."""
        return {
            "summary": self.metrics.get_summary(),
            "detailed_results": self.metrics.results
        }


async def main():
    """Main entry point for benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="ReasonForge Benchmark Suite")
    parser.add_argument(
        "--config",
        default="benchmarks/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Run benchmark
    runner = BenchmarkRunner(config_path=args.config)
    await runner.run_benchmark(verbose=not args.quiet)

    # Save results
    results_file = runner.save_results(output_path=args.output)

    # Generate visualizations and reports (if available)
    try:
        from visualizations import create_comparison_charts
        from report_generator import generate_html_report, generate_markdown_report

        results = runner.get_results()

        # Create charts
        chart_files = create_comparison_charts(results, output_dir=args.output)
        print(f"Charts saved to: {args.output}")

        # Generate reports
        html_report = generate_html_report(results, chart_files, output_dir=args.output)
        md_report = generate_markdown_report(results, chart_files, output_dir=args.output)

        print(f"HTML report: {html_report}")
        print(f"Markdown report: {md_report}")

    except ImportError:
        print("Visualization and report modules not yet available")


if __name__ == "__main__":
    asyncio.run(main())
