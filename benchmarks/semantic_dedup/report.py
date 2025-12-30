"""Report generation for semantic deduplication benchmarks."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .metrics import MetricsAggregator


@dataclass
class BenchmarkReport:
    """Complete benchmark report data."""
    timestamp: str
    dataset_name: str
    thresholds: List[float]
    results: List[Dict]
    best_f1: Dict
    summary: Dict


class ReportGenerator:
    """Generates JSON and HTML reports from benchmark results."""

    def __init__(self, dataset_name: str, results: MetricsAggregator):
        """Initialize report generator.

        Args:
            dataset_name: Name of the dataset
            results: Metrics aggregator with benchmark results
        """
        self.dataset_name = dataset_name
        self.results = results
        self.timestamp = datetime.utcnow().isoformat() + "Z"

    def generate_report(self) -> BenchmarkReport:
        """Generate complete report data.

        Returns:
            BenchmarkReport with all metrics
        """
        all_results = self.results.get_all_results()
        best_f1 = self.results.get_best_f1()

        # Extract thresholds
        thresholds = [r["threshold"] for r in all_results]

        # Create summary statistics
        summary = {
            "total_runs": len(all_results),
            "best_threshold": best_f1.get("threshold", 0.0),
            "best_f1": best_f1.get("f1_score", 0.0),
            "best_precision": best_f1.get("precision", 0.0),
            "best_recall": best_f1.get("recall", 0.0),
            "avg_latency_p50_ms": sum(r["latency_p50_ms"] for r in all_results) / len(all_results) if all_results else 0,
            "avg_latency_p95_ms": sum(r["latency_p95_ms"] for r in all_results) / len(all_results) if all_results else 0,
        }

        return BenchmarkReport(
            timestamp=self.timestamp,
            dataset_name=self.dataset_name,
            thresholds=thresholds,
            results=all_results,
            best_f1=best_f1,
            summary=summary,
        )

    def save_json(self, output_path: Path):
        """Save report as JSON file.

        Args:
            output_path: Path to output JSON file
        """
        report = self.generate_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2)

    def save_html(self, output_path: Path):
        """Save report as HTML file.

        Args:
            output_path: Path to output HTML file
        """
        report = self.generate_report()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._generate_html(report)

        with open(output_path, "w") as f:
            f.write(html)

    def _generate_html(self, report: BenchmarkReport) -> str:
        """Generate HTML report.

        Args:
            report: Benchmark report data

        Returns:
            HTML string
        """
        # Create results table rows
        table_rows = []
        for result in report.results:
            table_rows.append(f"""
                <tr>
                    <td>{result['threshold']:.2f}</td>
                    <td>{result['precision']:.4f}</td>
                    <td>{result['recall']:.4f}</td>
                    <td><strong>{result['f1_score']:.4f}</strong></td>
                    <td>{result['accuracy']:.4f}</td>
                    <td>{result['dedup_ratio']:.2f}x</td>
                    <td>{result['latency_p50_ms']:.2f}</td>
                    <td>{result['latency_p95_ms']:.2f}</td>
                    <td>{result['storage_bytes'] / 1024 / 1024:.2f}</td>
                </tr>
            """)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Semantic Deduplication Benchmark - {report.dataset_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }}
        .metric-label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        .best-row {{
            background: #e7f3ff !important;
        }}
    </style>
</head>
<body>
    <h1>Semantic Deduplication Benchmark Report</h1>
    <p class="timestamp">Generated: {report.timestamp}</p>

    <div class="summary">
        <h2>Summary - {report.dataset_name.upper()}</h2>
        <div class="summary-grid">
            <div class="metric">
                <div class="metric-label">Best F1 Score</div>
                <div class="metric-value">{report.best_f1.get('f1_score', 0.0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Threshold</div>
                <div class="metric-value">{report.best_f1.get('threshold', 0.0):.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Precision</div>
                <div class="metric-value">{report.best_f1.get('precision', 0.0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best Recall</div>
                <div class="metric-value">{report.best_f1.get('recall', 0.0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Latency P50</div>
                <div class="metric-value">{report.summary['avg_latency_p50_ms']:.2f} ms</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Latency P95</div>
                <div class="metric-value">{report.summary['avg_latency_p95_ms']:.2f} ms</div>
            </div>
        </div>
    </div>

    <h2>Results by Threshold</h2>
    <table>
        <thead>
            <tr>
                <th>Threshold</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Accuracy</th>
                <th>Dedup Ratio</th>
                <th>P50 Latency (ms)</th>
                <th>P95 Latency (ms)</th>
                <th>Storage (MB)</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>

    <h2>Confusion Matrix (Best F1)</h2>
    <table style="max-width: 500px;">
        <tr>
            <th></th>
            <th>Predicted Duplicate</th>
            <th>Predicted Unique</th>
        </tr>
        <tr>
            <th>Actually Duplicate</th>
            <td>TP: {report.best_f1.get('tp', 0)}</td>
            <td>FN: {report.best_f1.get('fn', 0)}</td>
        </tr>
        <tr>
            <th>Actually Unique</th>
            <td>FP: {report.best_f1.get('fp', 0)}</td>
            <td>TN: {report.best_f1.get('tn', 0)}</td>
        </tr>
    </table>

</body>
</html>
"""
        return html


class BaselineComparator:
    """Compares current results against a baseline."""

    def __init__(self, current: BenchmarkReport, baseline: BenchmarkReport):
        """Initialize comparator.

        Args:
            current: Current benchmark report
            baseline: Baseline benchmark report
        """
        self.current = current
        self.baseline = baseline

    def detect_regressions(self, threshold_pct: float = 5.0) -> List[Dict]:
        """Detect metric regressions.

        Args:
            threshold_pct: Percentage threshold for regression detection

        Returns:
            List of regression records
        """
        regressions = []

        # Compare best F1
        current_f1 = self.current.best_f1.get("f1_score", 0.0)
        baseline_f1 = self.baseline.best_f1.get("f1_score", 0.0)

        if baseline_f1 > 0:
            pct_change = 100 * (current_f1 - baseline_f1) / baseline_f1

            if pct_change < -threshold_pct:
                regressions.append({
                    "metric": "best_f1",
                    "current": current_f1,
                    "baseline": baseline_f1,
                    "change_pct": pct_change,
                })

        # Compare latency (regression if slower)
        current_p95 = self.current.summary["avg_latency_p95_ms"]
        baseline_p95 = self.baseline.summary["avg_latency_p95_ms"]

        if baseline_p95 > 0:
            pct_change = 100 * (current_p95 - baseline_p95) / baseline_p95

            if pct_change > threshold_pct:  # Note: > for latency (higher is worse)
                regressions.append({
                    "metric": "latency_p95",
                    "current": current_p95,
                    "baseline": baseline_p95,
                    "change_pct": pct_change,
                })

        return regressions

    def generate_comparison_report(self) -> Dict:
        """Generate comparison report.

        Returns:
            Dictionary with comparison data
        """
        regressions = self.detect_regressions()

        return {
            "has_regressions": len(regressions) > 0,
            "regressions": regressions,
            "current_summary": self.current.summary,
            "baseline_summary": self.baseline.summary,
            "improvements": self._detect_improvements(),
        }

    def _detect_improvements(self) -> List[Dict]:
        """Detect metric improvements.

        Returns:
            List of improvement records
        """
        improvements = []

        # Check F1 improvement
        current_f1 = self.current.best_f1.get("f1_score", 0.0)
        baseline_f1 = self.baseline.best_f1.get("f1_score", 0.0)

        if baseline_f1 > 0:
            pct_change = 100 * (current_f1 - baseline_f1) / baseline_f1
            if pct_change > 1.0:  # At least 1% improvement
                improvements.append({
                    "metric": "best_f1",
                    "current": current_f1,
                    "baseline": baseline_f1,
                    "change_pct": pct_change,
                })

        return improvements


def load_report(path: Path) -> BenchmarkReport:
    """Load benchmark report from JSON file.

    Args:
        path: Path to JSON report file

    Returns:
        BenchmarkReport object
    """
    with open(path, "r") as f:
        data = json.load(f)

    return BenchmarkReport(**data)


def compare_reports(
    current_path: Path,
    baseline_path: Path,
    output_path: Optional[Path] = None
) -> Dict:
    """Compare two benchmark reports.

    Args:
        current_path: Path to current report JSON
        baseline_path: Path to baseline report JSON
        output_path: Optional path to save comparison JSON

    Returns:
        Comparison dictionary
    """
    current = load_report(current_path)
    baseline = load_report(baseline_path)

    comparator = BaselineComparator(current, baseline)
    comparison = comparator.generate_comparison_report()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)

    return comparison
