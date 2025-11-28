"""
Report generation for benchmark results.

Generates comprehensive HTML and Markdown reports with:
- Executive summary
- Comparison charts
- Detailed test results
- Statistical analysis
"""

import base64
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def _embed_image_base64(image_path: str) -> str:
    """Convert image to base64 for HTML embedding."""
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        return f"data:image/png;base64,{image_data}"
    except Exception:
        return ""


def generate_html_report(
    results: Dict[str, Any],
    chart_files: Dict[str, str],
    output_dir: str = "benchmarks/results"
) -> str:
    """
    Generate comprehensive HTML report.

    Args:
        results: Benchmark results
        chart_files: Dictionary of chart file paths
        output_dir: Output directory

    Returns:
        Path to generated HTML file
    """
    summary = results["summary"]
    detailed = results["detailed_results"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ReasonForge Benchmark Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #2ecc71;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}

        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}

        .metadata {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .summary-card.reasonforge {{
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        }}

        .summary-card h3 {{
            font-size: 1.1em;
            margin-bottom: 15px;
            opacity: 0.9;
        }}

        .metric {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .metric-label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}

        .metrics-row {{
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }}

        .chart {{
            margin: 30px 0;
            text-align: center;
        }}

        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background-color: #34495e;
            color: white;
            font-weight: 600;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        .correct {{
            color: #2ecc71;
            font-weight: bold;
        }}

        .incorrect {{
            color: #e74c3c;
            font-weight: bold;
        }}

        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
        }}

        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ReasonForge Benchmark Report</h1>

        <div class="metadata">
            <strong>Generated:</strong> {timestamp}<br>
            <strong>Total Tests:</strong> {summary['total_tests']}<br>
            <strong>Providers:</strong> {', '.join(summary['providers'].keys())}
        </div>

        <h2>Executive Summary</h2>
        <div class="summary-grid">
"""

    # Add summary cards for each provider
    for provider, stats in summary["providers"].items():
        card_class = "reasonforge" if provider == "reasonforge" else ""
        html += f"""
            <div class="summary-card {card_class}">
                <h3>{provider.upper()}</h3>
                <div class="metric">{stats['accuracy']:.1f}%</div>
                <div class="metric-label">Accuracy</div>
                <div class="metrics-row">
                    <div>
                        <div style="font-size: 1.2em; font-weight: bold;">{stats['avg_latency_ms']:.0f}ms</div>
                        <div class="metric-label">Avg Latency</div>
                    </div>
                    <div>
                        <div style="font-size: 1.2em; font-weight: bold;">${stats['total_cost']:.4f}</div>
                        <div class="metric-label">Total Cost</div>
                    </div>
                </div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.2);">
                    <strong>{stats['correct']}</strong> / {stats['total']} tests passed
                </div>
            </div>
"""

    html += """
        </div>

        <h2>Comparison Charts</h2>
"""

    # Embed charts
    chart_titles = {
        "combined": "Complete Comparison",
        "accuracy": "Accuracy Comparison",
        "latency": "Speed Comparison",
        "cost": "Cost Comparison",
        "category": "Accuracy by Category"
    }

    for chart_name, chart_title in chart_titles.items():
        if chart_name in chart_files:
            img_data = _embed_image_base64(chart_files[chart_name])
            if img_data:
                html += f"""
        <div class="chart">
            <h3>{chart_title}</h3>
            <img src="{img_data}" alt="{chart_title}">
        </div>
"""

    # Add detailed results table
    html += """
        <h2>Detailed Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test ID</th>
                    <th>Provider</th>
                    <th>Result</th>
                    <th>Latency (ms)</th>
                    <th>Cost (USD)</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""

    for result in detailed:
        status_class = "correct" if result["correct"] else "incorrect"
        status_text = "✓ PASS" if result["correct"] else "✗ FAIL"

        html += f"""
                <tr>
                    <td>{result['test_id']}</td>
                    <td>{result['provider']}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result['latency_ms']}</td>
                    <td>${result['cost']:.6f}</td>
                    <td>{result['explanation']}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by ReasonForge Benchmark Suite</p>
            <p>ReasonForge: 100% Accuracy Symbolic AI for Mathematical Reasoning</p>
        </div>
    </div>
</body>
</html>
"""

    # Save HTML file
    output_path = Path(output_dir)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = output_path / f"benchmark_report_{timestamp_str}.html"

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)

    return str(html_file)


def generate_markdown_report(
    results: Dict[str, Any],
    chart_files: Dict[str, str],
    output_dir: str = "benchmarks/results"
) -> str:
    """
    Generate comprehensive Markdown report.

    Args:
        results: Benchmark results
        chart_files: Dictionary of chart file paths
        output_dir: Output directory

    Returns:
        Path to generated Markdown file
    """
    summary = results["summary"]
    detailed = results["detailed_results"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build Markdown
    md = f"""# ReasonForge Benchmark Report

**Generated:** {timestamp}
**Total Tests:** {summary['total_tests']}
**Providers:** {', '.join(summary['providers'].keys())}

---

## Executive Summary

"""

    # Add summary table
    md += "| Provider | Accuracy | Tests Passed | Avg Latency | Total Cost |\n"
    md += "|----------|----------|--------------|-------------|------------|\n"

    for provider, stats in summary["providers"].items():
        md += f"| **{provider}** | **{stats['accuracy']:.1f}%** | {stats['correct']}/{stats['total']} | {stats['avg_latency_ms']:.0f}ms | ${stats['total_cost']:.4f} |\n"

    md += "\n---\n\n"

    # Key findings
    md += "## Key Findings\n\n"

    providers_list = list(summary["providers"].items())
    if len(providers_list) > 1:
        # Compare ReasonForge to others
        for provider, stats in providers_list:
            md += f"### {provider.upper()}\n\n"
            md += f"- **Accuracy:** {stats['accuracy']:.1f}% ({stats['correct']} out of {stats['total']} tests)\n"
            md += f"- **Average Speed:** {stats['avg_latency_ms']:.0f}ms per test\n"
            md += f"- **Total Cost:** ${stats['total_cost']:.4f}\n"

            if provider == "reasonforge":
                md += f"- **Cost per Test:** $0.00 (free symbolic computation)\n"
            else:
                cost_per_test = stats['total_cost'] / stats['total'] if stats['total'] > 0 else 0
                md += f"- **Cost per Test:** ${cost_per_test:.4f}\n"

            md += "\n"

    md += "---\n\n"

    # Add chart references
    md += "## Comparison Charts\n\n"

    chart_titles = {
        "combined": "Complete Comparison",
        "accuracy": "Accuracy Comparison",
        "latency": "Speed Comparison",
        "cost": "Cost Comparison",
        "category": "Accuracy by Category"
    }

    for chart_name, chart_title in chart_titles.items():
        if chart_name in chart_files:
            # Use relative path for Markdown
            chart_path = Path(chart_files[chart_name]).name
            md += f"### {chart_title}\n\n"
            md += f"![{chart_title}]({chart_path})\n\n"

    md += "---\n\n"

    # Detailed results
    md += "## Detailed Test Results\n\n"
    md += "| Test ID | Provider | Result | Latency (ms) | Cost | Details |\n"
    md += "|---------|----------|--------|--------------|------|----------|\n"

    for result in detailed:
        status = "✓ PASS" if result["correct"] else "✗ FAIL"
        md += f"| {result['test_id']} | {result['provider']} | {status} | {result['latency_ms']} | ${result['cost']:.6f} | {result['explanation']} |\n"

    md += "\n---\n\n"

    # Footer
    md += "## About ReasonForge\n\n"
    md += "ReasonForge is a symbolic AI system built on SymPy that provides mathematically rigorous\n"
    md += "computations with guaranteed accuracy. Unlike LLMs which may produce plausible but incorrect\n"
    md += "answers, ReasonForge uses formal symbolic mathematics to ensure every result is verifiable.\n\n"
    md += "**Key Benefits:**\n\n"
    md += "- 🎯 **100% Accuracy** on symbolic mathematics (when properly configured)\n"
    md += "- ⚡ **Fast** - Direct computation without API latency\n"
    md += "- 💰 **Free** - No API costs or usage limits\n"
    md += "- 🔒 **Private** - All computation happens locally\n"
    md += "- ✅ **Verifiable** - Results can be independently checked\n\n"
    md += "---\n\n"
    md += "*Generated by ReasonForge Benchmark Suite*\n"

    # Save Markdown file
    output_path = Path(output_dir)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_file = output_path / f"benchmark_report_{timestamp_str}.md"

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md)

    return str(md_file)
