"""Command-line interface for dataloader benchmarks.

Usage:
    python -m benchmarks.dataloaders.cli run --all
    python -m benchmarks.dataloaders.cli run --category generalization
    python -m benchmarks.dataloaders.cli run --quick
    python -m benchmarks.dataloaders.cli report results.json --format html
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

try:
    import click
except ImportError:
    click = None


def main():
    """Main entry point."""
    if click is None:
        print("Click is required for CLI. Install with: pip install click")
        sys.exit(1)

    cli()


@click.group() if click else lambda: None
def cli():
    """Prestige Dataloaders Benchmark Suite.

    Run benchmarks focused on statistical and model performance
    for data scientists using deduplication dataloaders.
    """
    pass


@cli.command() if click else lambda: None
@click.option(
    "--all", "run_all", is_flag=True, help="Run all benchmark categories"
)
@click.option(
    "--category",
    "-c",
    type=click.Choice([
        "generalization",
        "contamination",
        "detection_quality",
        "mode_comparison",
        "statistical",
        "performance",
    ]),
    multiple=True,
    help="Benchmark category to run",
)
@click.option(
    "--quick", is_flag=True, help="Quick mode (fewer seeds, smaller datasets)"
)
@click.option(
    "--full", is_flag=True, help="Full evaluation (5 seeds, all metrics)"
)
@click.option(
    "--seeds", "-s", type=int, default=None, help="Number of random seeds"
)
@click.option(
    "--dataset", "-d", type=str, default=None, help="Dataset name to use"
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["exact", "semantic", "both"]),
    default="exact",
    help="Deduplication mode",
)
@click.option(
    "--threshold", "-t", type=float, default=0.9, help="Semantic threshold"
)
@click.option(
    "--output", "-o", type=click.Path(), default=None, help="Output file path"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(
    run_all: bool,
    category: tuple,
    quick: bool,
    full: bool,
    seeds: Optional[int],
    dataset: Optional[str],
    mode: str,
    threshold: float,
    output: Optional[str],
    verbose: bool,
):
    """Run benchmarks."""
    from .config import BenchmarkConfig, BenchmarkCategory, DedupMode, StatisticalConfig
    from .runner import BenchmarkRunner
    from .report import generate_summary_report, save_report

    # Build config
    if quick:
        config = BenchmarkConfig(quick_mode=True, verbose=verbose)
    elif full:
        config = BenchmarkConfig(
            statistical=StatisticalConfig(num_seeds=5),
            verbose=verbose,
        )
    else:
        config = BenchmarkConfig(verbose=verbose)

    # Set categories
    if run_all:
        config.categories = [BenchmarkCategory.ALL]
    elif category:
        config.categories = [BenchmarkCategory(c) for c in category]

    # Set dedup mode
    if mode == "semantic":
        config.dedup.mode = DedupMode.SEMANTIC
    elif mode == "exact":
        config.dedup.mode = DedupMode.EXACT
    else:
        config.dedup.mode = DedupMode.BOTH

    config.dedup.semantic_threshold = threshold

    # Set seeds
    if seeds is not None:
        config.statistical.num_seeds = seeds

    # Set dataset
    if dataset is not None:
        config.dataset.name = dataset

    # Run benchmarks
    click.echo("Starting benchmark run...")
    runner = BenchmarkRunner(config)
    results = runner.run()

    # Generate report
    report = generate_summary_report(results)
    click.echo("\n" + report)

    # Save results
    if output:
        output_path = Path(output)
        if output_path.suffix == ".json":
            runner.save_results(output_path)
        else:
            save_report(results, output_path)
        click.echo(f"\nResults saved to: {output_path}")
    else:
        # Auto-save to default location
        saved_path = runner.save_results()
        click.echo(f"\nResults saved to: {saved_path}")


@cli.command() if click else lambda: None
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["txt", "json", "html"]),
    default="txt",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file")
def report(results_file: str, format: str, output: Optional[str]):
    """Generate report from benchmark results."""
    from .metrics import BenchmarkResult, GeneralizationMetrics
    from .report import save_report, generate_summary_report

    # Load results
    with open(results_file) as f:
        data = json.load(f)

    # Reconstruct BenchmarkResult objects
    results = []
    for r in data.get("results", []):
        gen = None
        if "generalization" in r:
            gen = GeneralizationMetrics(
                test_accuracies=r["generalization"].get("test_accuracies", []),
                baseline_test_accuracies=r["generalization"].get("baseline_test_accuracies", []),
            )

        results.append(BenchmarkResult(
            benchmark_name=r.get("benchmark_name", "unknown"),
            dataset_name=r.get("dataset_name", "unknown"),
            dedup_mode=r.get("dedup_mode", "unknown"),
            threshold=r.get("threshold"),
            generalization=gen,
        ))

    if output:
        output_path = Path(output)
        save_report(results, output_path, format=format)
        click.echo(f"Report saved to: {output_path}")
    else:
        if format == "json":
            from .report import generate_json_report
            click.echo(json.dumps(generate_json_report(results), indent=2))
        else:
            click.echo(generate_summary_report(results))


@cli.command() if click else lambda: None
@click.argument("baseline", type=click.Path(exists=True))
@click.argument("current", type=click.Path(exists=True))
def compare(baseline: str, current: str):
    """Compare two benchmark result files."""
    from .report import compare_reports

    comparison = compare_reports(Path(baseline), Path(current))
    click.echo(comparison)


@cli.command() if click else lambda: None
def list_datasets():
    """List available datasets (synthetic and real)."""
    from .datasets import list_all_datasets_with_info, HF_AVAILABLE

    all_info = list_all_datasets_with_info()

    # Synthetic datasets
    click.echo("Synthetic Datasets (controlled experiments with ground truth):")
    click.echo("-" * 60)
    for name, info in all_info.items():
        if info["type"] == "synthetic":
            click.echo(f"  {name}")
            click.echo(f"    Size: {info['size']}")
            click.echo(f"    Use: {info['description']}")
            click.echo()

    # Real datasets
    click.echo("\nReal Datasets (credible real-world benchmarks):")
    click.echo("-" * 60)
    if not HF_AVAILABLE:
        click.echo("  [!] HuggingFace datasets not installed.")
        click.echo("      Install with: pip install datasets")
        click.echo()

    for name, info in all_info.items():
        if info["type"] == "real":
            status = "" if info["available"] else " [unavailable]"
            click.echo(f"  {name}{status}")
            click.echo(f"    Size: {info['size']} | Task: {info.get('task', 'N/A')}")
            click.echo(f"    Use: {info['description']}")
            click.echo()


@cli.command() if click else lambda: None
def list_models():
    """List available baseline models."""
    from .models import list_models as _list_models

    click.echo("Available baseline models:")
    for name in _list_models():
        click.echo(f"  - {name}")


if __name__ == "__main__":
    main()
