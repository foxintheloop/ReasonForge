"""
Visualization generation for benchmark results.

Creates comparison charts between ReasonForge and LLMs showing:
- Accuracy comparison
- Latency comparison
- Cost comparison
- Category-wise breakdown
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from datetime import datetime


def create_accuracy_chart(
    summary: Dict[str, Any],
    output_path: str = "benchmarks/results/accuracy_comparison.png"
) -> str:
    """
    Create bar chart comparing accuracy across providers.

    Args:
        summary: Benchmark summary statistics
        output_path: Where to save the chart

    Returns:
        Path to saved chart
    """
    providers = list(summary["providers"].keys())
    accuracies = [summary["providers"][p]["accuracy"] for p in providers]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars with colors
    colors = ['#2ecc71' if p == 'reasonforge' else '#3498db' for p in providers]
    bars = ax.bar(providers, accuracies, color=colors, alpha=0.8, edgecolor='black')

    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mathematical Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_latency_chart(
    summary: Dict[str, Any],
    output_path: str = "benchmarks/results/latency_comparison.png"
) -> str:
    """
    Create bar chart comparing average latency across providers.

    Args:
        summary: Benchmark summary statistics
        output_path: Where to save the chart

    Returns:
        Path to saved chart
    """
    providers = list(summary["providers"].keys())
    latencies = [summary["providers"][p]["avg_latency_ms"] for p in providers]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    colors = ['#2ecc71' if p == 'reasonforge' else '#e74c3c' for p in providers]
    bars = ax.bar(providers, latencies, color=colors, alpha=0.8, edgecolor='black')

    # Customize
    ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + max(latencies) * 0.02,
            f'{height:.0f}ms',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_cost_chart(
    summary: Dict[str, Any],
    output_path: str = "benchmarks/results/cost_comparison.png"
) -> str:
    """
    Create bar chart comparing total cost across providers.

    Args:
        summary: Benchmark summary statistics
        output_path: Where to save the chart

    Returns:
        Path to saved chart
    """
    providers = list(summary["providers"].keys())
    costs = [summary["providers"][p]["total_cost"] for p in providers]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    colors = ['#2ecc71' if p == 'reasonforge' else '#f39c12' for p in providers]
    bars = ax.bar(providers, costs, color=colors, alpha=0.8, edgecolor='black')

    # Customize
    ax.set_ylabel('Total Cost (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Cost Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + max(costs) * 0.02 if max(costs) > 0 else 0.01,
            f'${height:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_category_breakdown(
    detailed_results: List[Dict[str, Any]],
    output_path: str = "benchmarks/results/category_breakdown.png"
) -> str:
    """
    Create grouped bar chart showing accuracy by category for each provider.

    Args:
        detailed_results: Detailed test results
        output_path: Where to save the chart

    Returns:
        Path to saved chart
    """
    # Extract categories and providers
    from test_cases import get_test_case_by_id

    providers = list(set(r["provider"] for r in detailed_results))
    categories = list(set(get_test_case_by_id(r["test_id"])["category"] for r in detailed_results))

    # Calculate accuracy by category and provider
    accuracy_data = {provider: {cat: [] for cat in categories} for provider in providers}

    for result in detailed_results:
        provider = result["provider"]
        test_case = get_test_case_by_id(result["test_id"])
        category = test_case["category"]
        accuracy_data[provider][category].append(1 if result["correct"] else 0)

    # Calculate percentages
    accuracy_percentages = {}
    for provider in providers:
        accuracy_percentages[provider] = {}
        for category in categories:
            results = accuracy_data[provider][category]
            if results:
                accuracy_percentages[provider][category] = (sum(results) / len(results)) * 100
            else:
                accuracy_percentages[provider][category] = 0

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.8 / len(providers)

    # Create bars for each provider
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    for i, provider in enumerate(providers):
        values = [accuracy_percentages[provider][cat] for cat in categories]
        offset = (i - len(providers) / 2) * width + width / 2
        ax.bar(
            x + offset,
            values,
            width,
            label=provider,
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor='black'
        )

    # Customize
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Mathematical Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_combined_comparison(
    summary: Dict[str, Any],
    output_path: str = "benchmarks/results/combined_comparison.png"
) -> str:
    """
    Create combined chart showing accuracy, speed, and cost.

    Args:
        summary: Benchmark summary statistics
        output_path: Where to save the chart

    Returns:
        Path to saved chart
    """
    providers = list(summary["providers"].keys())

    # Extract metrics
    accuracies = [summary["providers"][p]["accuracy"] for p in providers]
    latencies = [summary["providers"][p]["avg_latency_ms"] for p in providers]
    costs = [summary["providers"][p]["total_cost"] for p in providers]

    # Normalize latencies and costs to 0-100 scale for comparison
    # Lower is better, so invert the scale
    max_latency = max(latencies) if max(latencies) > 0 else 1
    normalized_speed = [(1 - (lat / max_latency)) * 100 for lat in latencies]

    max_cost = max(costs) if max(costs) > 0 else 1
    normalized_cost_efficiency = [(1 - (cost / max_cost)) * 100 for cost in costs]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#2ecc71' if p == 'reasonforge' else '#3498db' for p in providers]

    # Accuracy subplot
    axes[0].bar(providers, accuracies, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # Speed subplot
    axes[1].bar(providers, latencies, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Latency (ms)', fontweight='bold')
    axes[1].set_title('Speed (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(latencies):
        axes[1].text(i, v + max(latencies) * 0.02, f'{v:.0f}ms', ha='center', fontweight='bold')

    # Cost subplot
    axes[2].bar(providers, costs, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Cost (USD)', fontweight='bold')
    axes[2].set_title('Cost (Lower is Better)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(costs):
        axes[2].text(
            i,
            v + max(costs) * 0.02 if max(costs) > 0 else 0.01,
            f'${v:.4f}',
            ha='center',
            fontweight='bold'
        )

    plt.suptitle('ReasonForge vs LLMs: Complete Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def create_comparison_charts(
    results: Dict[str, Any],
    output_dir: str = "benchmarks/results"
) -> Dict[str, str]:
    """
    Create all comparison charts.

    Args:
        results: Complete benchmark results
        output_dir: Directory to save charts

    Returns:
        Dictionary mapping chart names to file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = results["summary"]
    detailed_results = results["detailed_results"]

    chart_files = {}

    # Create individual charts
    print("Generating charts...")

    chart_files["accuracy"] = create_accuracy_chart(
        summary,
        str(output_path / "accuracy_comparison.png")
    )
    print("  [OK] Accuracy chart created")

    chart_files["latency"] = create_latency_chart(
        summary,
        str(output_path / "latency_comparison.png")
    )
    print("  [OK] Latency chart created")

    chart_files["cost"] = create_cost_chart(
        summary,
        str(output_path / "cost_comparison.png")
    )
    print("  [OK] Cost chart created")

    chart_files["category"] = create_category_breakdown(
        detailed_results,
        str(output_path / "category_breakdown.png")
    )
    print("  [OK] Category breakdown created")

    chart_files["combined"] = create_combined_comparison(
        summary,
        str(output_path / "combined_comparison.png")
    )
    print("  [OK] Combined comparison created")

    return chart_files
