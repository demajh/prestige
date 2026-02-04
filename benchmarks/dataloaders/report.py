"""Report generation for dataloader benchmarks.

This module generates human-readable reports from benchmark results,
focusing on actionable insights for data scientists.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import BenchmarkResult, GeneralizationMetrics


def generate_summary_report(results: List[BenchmarkResult]) -> str:
    """Generate executive summary report.

    Args:
        results: List of benchmark results

    Returns:
        Human-readable summary string
    """
    lines = [
        "=" * 80,
        "                    PRESTIGE DATALOADERS BENCHMARK REPORT",
        "=" * 80,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total benchmarks: {len(results)}",
        "",
    ]

    # Find key metrics
    gen_results = [r for r in results if r.generalization and r.generalization.test_accuracies]
    cont_results = [r for r in results if r.contamination]
    quality_results = [r for r in results if r.dedup_quality]

    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)

    # Best generalization result (only from actual test accuracy benchmarks)
    # Exclude comparison benchmarks that use GeneralizationMetrics for other purposes
    test_accuracy_benchmarks = [
        "test_accuracy_with_dedup",
        "overfitting_reduction",
        "convergence_speed",
        "sample_efficiency",
        "cross_validation_variance",
        "mode_accuracy_comparison",
        "paraphrase_impact_on_model",
        "threshold_tuning",
    ]

    real_gen_results = [r for r in gen_results if r.benchmark_name in test_accuracy_benchmarks]

    if real_gen_results:
        best_gen = None
        best_improvement = -999

        for r in real_gen_results:
            if r.generalization.baseline_test_accuracies:
                improvement = r.generalization.test_accuracy_improvement()
                if improvement.mean > best_improvement:
                    best_improvement = improvement.mean
                    best_gen = r

        if best_gen and best_gen.generalization:
            improvement = best_gen.generalization.test_accuracy_improvement()
            effect = best_gen.generalization.effect_size()
            test = best_gen.generalization.hypothesis_test()

            if improvement.mean > 0:
                lines.append(f"  Test accuracy improved by +{improvement.mean*100:.1f}% [{improvement.lower*100:.1f}%, {improvement.upper*100:.1f}%]")
            else:
                lines.append(f"  Test accuracy changed by {improvement.mean*100:.1f}% [{improvement.lower*100:.1f}%, {improvement.upper*100:.1f}%]")

            lines.append(f"  Effect size (Cohen's d): {effect.cohens_d:.2f} ({effect.interpretation})")

            sig_marker = "**" if test.p_value < 0.01 else "*" if test.is_significant else ""
            lines.append(f"  Statistical significance: p={test.p_value:.4f}{sig_marker}")
        lines.append("")

    # Contamination summary
    if cont_results:
        total_contaminated = sum(r.contamination.contaminated_count for r in cont_results if r.contamination)
        total_samples = sum(r.contamination.total_test_samples for r in cont_results if r.contamination)

        if total_samples > 0:
            overall_rate = total_contaminated / total_samples
            severity = "none" if overall_rate < 0.001 else "low" if overall_rate < 0.01 else "medium" if overall_rate < 0.05 else "high"

            if overall_rate > 0.001:
                lines.append(f"  Contamination detected: {overall_rate*100:.2f}% ({severity} severity)")
            else:
                lines.append(f"  No significant contamination detected")
        lines.append("")

    # Quality summary
    if quality_results:
        avg_precision = sum(r.dedup_quality.precision for r in quality_results) / len(quality_results)
        avg_recall = sum(r.dedup_quality.recall for r in quality_results) / len(quality_results)
        avg_f1 = sum(r.dedup_quality.f1_score for r in quality_results) / len(quality_results)

        lines.append(f"  Dedup precision: {avg_precision:.2%}")
        lines.append(f"  Dedup recall: {avg_recall:.2%}")
        lines.append(f"  Dedup F1: {avg_f1:.2%}")
        lines.append("")

    # Detailed Results
    lines.append("")
    lines.append("DETAILED RESULTS")
    lines.append("-" * 40)

    # Group by benchmark name
    benchmarks: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.benchmark_name not in benchmarks:
            benchmarks[r.benchmark_name] = []
        benchmarks[r.benchmark_name].append(r)

    for name, bench_results in benchmarks.items():
        lines.append(f"\n{name}:")

        for r in bench_results[:5]:  # Limit to 5 results per benchmark
            detail_parts = [f"  - {r.dataset_name}, {r.dedup_mode}"]

            if r.threshold is not None:
                detail_parts.append(f"threshold={r.threshold}")

            if r.generalization and r.generalization.test_accuracies:
                acc = r.generalization.test_accuracies[0] if r.generalization.test_accuracies else 0
                detail_parts.append(f"test_acc={acc:.4f}")

            if r.contamination:
                detail_parts.append(f"contamination={r.contamination.contamination_rate:.2%}")

            if r.dedup_quality:
                detail_parts.append(f"precision={r.dedup_quality.precision:.2%}")

            if r.throughput_samples_per_sec:
                detail_parts.append(f"throughput={r.throughput_samples_per_sec:.0f}/s")

            lines.append(", ".join(detail_parts))

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_json_report(
    results: List[BenchmarkResult],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate JSON-serializable report.

    Args:
        results: List of benchmark results
        config: Optional configuration dictionary

    Returns:
        Dictionary suitable for JSON serialization
    """
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_benchmarks": len(results),
            "version": "1.0.0",
        },
        "config": config or {},
        "summary": {},
        "results": [],
    }

    # Compute summary statistics
    gen_results = [r for r in results if r.generalization and r.generalization.test_accuracies]
    if gen_results:
        improvements = []
        for r in gen_results:
            if r.generalization.baseline_test_accuracies:
                imp = r.generalization.test_accuracy_improvement()
                improvements.append(imp.mean)

        if improvements:
            report["summary"]["avg_test_accuracy_improvement"] = sum(improvements) / len(improvements)

    cont_results = [r for r in results if r.contamination]
    if cont_results:
        rates = [r.contamination.contamination_rate for r in cont_results]
        report["summary"]["avg_contamination_rate"] = sum(rates) / len(rates)

    quality_results = [r for r in results if r.dedup_quality]
    if quality_results:
        report["summary"]["avg_precision"] = sum(r.dedup_quality.precision for r in quality_results) / len(quality_results)
        report["summary"]["avg_recall"] = sum(r.dedup_quality.recall for r in quality_results) / len(quality_results)
        report["summary"]["avg_f1"] = sum(r.dedup_quality.f1_score for r in quality_results) / len(quality_results)

    # Add individual results
    for r in results:
        result_dict = {
            "benchmark_name": r.benchmark_name,
            "dataset_name": r.dataset_name,
            "dedup_mode": r.dedup_mode,
            "threshold": r.threshold,
        }

        if r.generalization:
            result_dict["generalization"] = {
                "test_accuracies": r.generalization.test_accuracies,
                "baseline_test_accuracies": r.generalization.baseline_test_accuracies,
            }

        if r.contamination:
            result_dict["contamination"] = {
                "rate": r.contamination.contamination_rate,
                "count": r.contamination.contaminated_count,
                "severity": r.contamination.leakage_severity,
            }

        if r.dedup_quality:
            result_dict["quality"] = {
                "precision": r.dedup_quality.precision,
                "recall": r.dedup_quality.recall,
                "f1": r.dedup_quality.f1_score,
            }

        if r.throughput_samples_per_sec:
            result_dict["throughput"] = r.throughput_samples_per_sec

        if r.peak_memory_gb:
            result_dict["peak_memory_gb"] = r.peak_memory_gb

        report["results"].append(result_dict)

    return report


def save_report(
    results: List[BenchmarkResult],
    output_path: Path,
    format: str = "txt",
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save report to file.

    Args:
        results: List of benchmark results
        output_path: Output file path
        format: "txt", "json", or "html"
        config: Optional configuration

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        report = generate_json_report(results, config)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    elif format == "html":
        # Generate simple HTML report
        txt_report = generate_summary_report(results)
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Prestige Dataloaders Benchmark Report</title>
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        pre {{ background: #f5f5f5; padding: 20px; overflow-x: auto; }}
    </style>
</head>
<body>
<pre>{txt_report}</pre>
</body>
</html>"""
        with open(output_path, "w") as f:
            f.write(html)

    else:  # txt
        report = generate_summary_report(results)
        with open(output_path, "w") as f:
            f.write(report)

    return output_path


def compare_reports(
    baseline_path: Path,
    current_path: Path,
) -> str:
    """Compare two benchmark reports.

    Args:
        baseline_path: Path to baseline results JSON
        current_path: Path to current results JSON

    Returns:
        Comparison summary string
    """
    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(current_path) as f:
        current = json.load(f)

    lines = [
        "BENCHMARK COMPARISON",
        "=" * 60,
        "",
        f"Baseline: {baseline_path.name}",
        f"Current: {current_path.name}",
        "",
    ]

    # Compare summary metrics
    if "summary" in baseline and "summary" in current:
        lines.append("Summary Metrics:")

        for key in baseline["summary"]:
            if key in current["summary"]:
                base_val = baseline["summary"][key]
                curr_val = current["summary"][key]
                change = curr_val - base_val
                pct_change = (change / base_val * 100) if base_val != 0 else 0

                indicator = ""
                if abs(pct_change) > 5:
                    indicator = " ***" if pct_change > 0 else " !!!"

                lines.append(f"  {key}: {base_val:.4f} -> {curr_val:.4f} ({pct_change:+.1f}%){indicator}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
