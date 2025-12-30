"""Command-line interface for semantic deduplication benchmarks."""

import sys
import json
from pathlib import Path
from typing import List, Optional

import click

# Handle both direct execution and module import
try:
    from .dataset_loader import list_available_datasets, get_dataset_config
    from .runner import SemanticDedupBenchmark, BenchmarkConfig
    from .report import ReportGenerator, compare_reports, load_report
except ImportError:
    # Direct execution - add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from benchmarks.semantic_dedup.dataset_loader import list_available_datasets, get_dataset_config
    from benchmarks.semantic_dedup.runner import SemanticDedupBenchmark, BenchmarkConfig
    from benchmarks.semantic_dedup.report import ReportGenerator, compare_reports, load_report


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Semantic deduplication benchmark harness for Prestige.

    Evaluate Prestige's semantic deduplication performance against labeled datasets.
    """
    pass


@cli.command()
@click.option(
    "--datasets",
    "-d",
    default="mrpc",
    help="Comma-separated dataset names (default: mrpc). Available: " + ", ".join(list_available_datasets()),
)
@click.option(
    "--thresholds",
    "-t",
    default="0.85,0.90,0.95",
    help="Comma-separated similarity thresholds (default: 0.85,0.90,0.95)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="./results",
    help="Output directory for results (default: ./results)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Cache directory for datasets and models (default: ~/.cache/prestige/benchmarks)",
)
@click.option(
    "--model",
    default="bge-small",
    help="Embedding model name (default: bge-small)",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Batch size for periodic flushes (default: 100)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress progress output",
)
@click.option(
    "--json-only",
    is_flag=True,
    help="Only generate JSON output (skip HTML)",
)
def run(
    datasets: str,
    thresholds: str,
    output: Path,
    cache_dir: Optional[Path],
    model: str,
    batch_size: int,
    quiet: bool,
    json_only: bool,
):
    """Run semantic deduplication benchmarks.

    Example:

        python cli.py run --datasets mrpc,qqp --thresholds 0.85,0.90,0.95 --output ./results
    """
    # Parse datasets
    dataset_names = [d.strip() for d in datasets.split(",")]

    # Parse thresholds
    try:
        threshold_values = [float(t.strip()) for t in thresholds.split(",")]
    except ValueError:
        click.echo("Error: Invalid threshold values. Must be comma-separated floats.", err=True)
        sys.exit(1)

    # Validate thresholds
    for t in threshold_values:
        if not 0.0 <= t <= 1.0:
            click.echo(f"Error: Threshold {t} out of range [0.0, 1.0]", err=True)
            sys.exit(1)

    # Setup cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "prestige" / "benchmarks"

    cache_dir = Path(cache_dir)
    output = Path(output)

    output.mkdir(parents=True, exist_ok=True)

    if not quiet:
        click.echo("=" * 60)
        click.echo("Semantic Deduplication Benchmark")
        click.echo("=" * 60)
        click.echo(f"Datasets: {', '.join(dataset_names)}")
        click.echo(f"Thresholds: {', '.join(map(str, threshold_values))}")
        click.echo(f"Output: {output}")
        click.echo(f"Model: {model}")
        click.echo("=" * 60)

    # Run benchmarks for each dataset
    all_reports = {}

    for dataset_name in dataset_names:
        try:
            dataset_config = get_dataset_config(dataset_name)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        if not quiet:
            click.echo(f"\n{'='*60}")
            click.echo(f"Dataset: {dataset_name.upper()}")
            click.echo(f"{'='*60}")

        # Create benchmark config
        config = BenchmarkConfig(
            dataset_config=dataset_config,
            thresholds=threshold_values,
            cache_dir=cache_dir,
            embedding_model=model,
            batch_size=batch_size,
            verbose=not quiet,
        )

        # Run benchmark
        try:
            benchmark = SemanticDedupBenchmark(config)
            results = benchmark.run()

            # Generate reports
            report_gen = ReportGenerator(dataset_name, results)

            # Save JSON
            json_path = output / f"{dataset_name}_results.json"
            report_gen.save_json(json_path)

            if not quiet:
                click.echo(f"\nJSON report saved: {json_path}")

            # Save HTML
            if not json_only:
                html_path = output / f"{dataset_name}_report.html"
                report_gen.save_html(html_path)

                if not quiet:
                    click.echo(f"HTML report saved: {html_path}")

            all_reports[dataset_name] = report_gen.generate_report()

        except Exception as e:
            click.echo(f"Error running benchmark for {dataset_name}: {e}", err=True)
            if not quiet:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    # Save combined summary
    summary_path = output / "summary.json"
    summary = {
        "datasets": {
            name: {
                "best_f1": report.best_f1.get("f1_score", 0.0),
                "best_threshold": report.best_f1.get("threshold", 0.0),
                "best_precision": report.best_f1.get("precision", 0.0),
                "best_recall": report.best_f1.get("recall", 0.0),
            }
            for name, report in all_reports.items()
        }
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if not quiet:
        click.echo(f"\n{'='*60}")
        click.echo("Benchmark Complete!")
        click.echo(f"{'='*60}")
        click.echo(f"Summary saved: {summary_path}")
        click.echo("\nResults:")
        for name, report in all_reports.items():
            best = report.best_f1
            click.echo(f"  {name.upper()}:")
            click.echo(f"    Best F1: {best.get('f1_score', 0.0):.4f} (threshold={best.get('threshold', 0.0):.2f})")


@cli.command()
@click.argument("current", type=click.Path(exists=True, path_type=Path))
@click.argument("baseline", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save comparison to JSON file",
)
@click.option(
    "--threshold",
    type=float,
    default=5.0,
    help="Regression threshold percentage (default: 5.0)",
)
def compare(current: Path, baseline: Path, output: Optional[Path], threshold: float):
    """Compare current results against baseline.

    Example:

        python cli.py compare results/current.json results/baseline.json --output comparison.json
    """
    try:
        comparison = compare_reports(current, baseline, output)

        click.echo("=" * 60)
        click.echo("Benchmark Comparison")
        click.echo("=" * 60)
        click.echo(f"Current:  {current}")
        click.echo(f"Baseline: {baseline}")
        click.echo("=" * 60)

        if comparison["has_regressions"]:
            click.echo("\n❌ REGRESSIONS DETECTED:\n", err=True)
            for reg in comparison["regressions"]:
                click.echo(
                    f"  {reg['metric']}: {reg['current']:.4f} → {reg['baseline']:.4f} "
                    f"({reg['change_pct']:+.2f}%)",
                    err=True
                )
            sys.exit(1)
        else:
            click.echo("\n✓ No regressions detected")

        if comparison["improvements"]:
            click.echo("\n✓ IMPROVEMENTS:\n")
            for imp in comparison["improvements"]:
                click.echo(
                    f"  {imp['metric']}: {imp['baseline']:.4f} → {imp['current']:.4f} "
                    f"({imp['change_pct']:+.2f}%)"
                )

        if output:
            click.echo(f"\nComparison saved: {output}")

    except Exception as e:
        click.echo(f"Error comparing reports: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_json", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output HTML file path (default: <input>.html)",
)
def report(input_json: Path, output: Optional[Path]):
    """Generate HTML report from JSON results.

    Example:

        python cli.py report results/mrpc_results.json --output report.html
    """
    try:
        # Load report
        benchmark_report = load_report(input_json)

        # Determine output path
        if output is None:
            output = input_json.with_suffix(".html")

        # Generate HTML
        from .report import ReportGenerator
        from .metrics import MetricsAggregator

        # Reconstruct MetricsAggregator from report data
        aggregator = MetricsAggregator()
        aggregator.threshold_metrics = benchmark_report.results

        # Generate report
        gen = ReportGenerator(benchmark_report.dataset_name, aggregator)
        gen.save_html(output)

        click.echo(f"HTML report generated: {output}")

    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_datasets():
    """List available benchmark datasets."""
    datasets = list_available_datasets()

    click.echo("Available datasets:")
    for name in datasets:
        try:
            config = get_dataset_config(name)
            click.echo(f"  {name}: {config.source}")
        except ValueError:
            pass


if __name__ == "__main__":
    cli()
