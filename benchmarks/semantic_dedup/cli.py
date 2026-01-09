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
@click.option(
    "--pooling",
    type=click.Choice(["mean", "cls"], case_sensitive=False),
    default="mean",
    help="Pooling strategy: mean or cls (default: mean)",
)
@click.option(
    "--sample",
    type=int,
    default=None,
    help="Sample N pairs from dataset (default: use all)",
)
@click.option(
    "--enable-reranker",
    is_flag=True,
    help="Enable two-stage retrieval with reranker for higher accuracy",
)
@click.option(
    "--reranker-model",
    default="bge-reranker-v2-m3",
    help="Reranker model name (default: bge-reranker-v2-m3)",
)
@click.option(
    "--reranker-threshold",
    type=float,
    default=0.8,
    help="Reranker score threshold [0.0-1.0] (default: 0.8)",
)
@click.option(
    "--reranker-top-k",
    type=int,
    default=100,
    help="Number of candidates for reranking (default: 100)",
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
    pooling: str,
    sample: Optional[int],
    enable_reranker: bool,
    reranker_model: str,
    reranker_threshold: float,
    reranker_top_k: int,
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
        click.echo(f"Embedding Model: {model}")
        if enable_reranker:
            click.echo(f"Reranker: {reranker_model}")
            click.echo(f"Reranker Threshold: {reranker_threshold}")
            click.echo(f"Reranker Top-K: {reranker_top_k}")
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
            pooling=pooling,
            sample_size=sample,
            enable_reranker=enable_reranker,
            reranker_model=reranker_model,
            reranker_threshold=reranker_threshold,
            reranker_top_k=reranker_top_k,
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


@cli.command()
@click.option(
    "--dataset", "-d", default="mrpc", help="Dataset to test (default: mrpc)"
)
@click.option(
    "--threshold", "-t", type=float, default=0.85, help="Similarity threshold (default: 0.85)"
)
@click.option(
    "--model", default="bge-small", help="Embedding model (default: bge-small)"
)
@click.option(
    "--reranker-threshold", type=float, default=0.8, help="Reranker threshold (default: 0.8)"
)
@click.option(
    "--cache-dir", type=click.Path(path_type=Path), default=None, help="Cache directory"
)
@click.option(
    "--sample", type=int, default=500, help="Number of pairs to sample (default: 500)"
)
def compare_reranker(
    dataset: str,
    threshold: float,
    model: str,
    reranker_threshold: float,
    cache_dir: Optional[Path],
    sample: int,
):
    """Compare embeddings-only vs. reranker performance.
    
    Runs the same benchmark twice - once with embeddings only, 
    once with reranker enabled - and compares the results.
    
    Example:
    
        python cli.py compare-reranker --dataset mrpc --sample 1000
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "prestige" / "benchmarks"
    
    try:
        dataset_config = get_dataset_config(dataset)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    click.echo("=" * 70)
    click.echo("Reranker Performance Comparison")
    click.echo("=" * 70)
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Sample size: {sample} pairs")
    click.echo(f"Embedding model: {model}")
    click.echo(f"Semantic threshold: {threshold}")
    click.echo(f"Reranker threshold: {reranker_threshold}")
    click.echo("=" * 70)
    
    results = {}
    
    # Test 1: Embeddings only
    click.echo("\n[1/2] Testing embeddings-only approach...")
    config_embeddings = BenchmarkConfig(
        dataset_config=dataset_config,
        thresholds=[threshold],
        cache_dir=cache_dir,
        embedding_model=model,
        sample_size=sample,
        verbose=False,
        enable_reranker=False,
    )
    
    benchmark = SemanticDedupBenchmark(config_embeddings)
    results_embeddings = benchmark.run()
    embeddings_result = results_embeddings.get_all_results()[0]
    
    # Test 2: With reranker
    click.echo("\n[2/2] Testing with reranker...")
    config_reranker = BenchmarkConfig(
        dataset_config=dataset_config,
        thresholds=[threshold], 
        cache_dir=cache_dir,
        embedding_model=model,
        sample_size=sample,
        verbose=False,
        enable_reranker=True,
        reranker_threshold=reranker_threshold,
        reranker_top_k=100,
    )
    
    benchmark = SemanticDedupBenchmark(config_reranker)
    results_reranker = benchmark.run()
    reranker_result = results_reranker.get_all_results()[0]
    
    # Compare results
    click.echo("\n" + "=" * 70)
    click.echo("COMPARISON RESULTS")
    click.echo("=" * 70)
    
    metrics = ["precision", "recall", "f1_score", "accuracy"]
    click.echo(f"{'Metric':<12} {'Embeddings':<12} {'Reranker':<12} {'Improvement':<12}")
    click.echo("-" * 50)
    
    for metric in metrics:
        emb_val = embeddings_result.get(metric, 0.0)
        rer_val = reranker_result.get(metric, 0.0)
        improvement = ((rer_val - emb_val) / emb_val * 100) if emb_val > 0 else 0
        
        click.echo(f"{metric:<12} {emb_val:<12.4f} {rer_val:<12.4f} {improvement:+.1f}%")
    
    # Performance comparison
    emb_latency = embeddings_result.get("latency_p50_ms", 0)
    rer_latency = reranker_result.get("latency_p50_ms", 0)
    latency_overhead = ((rer_latency - emb_latency) / emb_latency * 100) if emb_latency > 0 else 0
    
    click.echo(f"\nLatency (p50):   {emb_latency:.1f}ms → {rer_latency:.1f}ms ({latency_overhead:+.1f}%)")
    
    dedup_emb = embeddings_result.get("dedup_ratio", 1.0)
    dedup_rer = reranker_result.get("dedup_ratio", 1.0)
    click.echo(f"Dedup ratio:     {dedup_emb:.2f}x → {dedup_rer:.2f}x")
    
    # Recommendation
    f1_improvement = ((reranker_result.get("f1_score", 0) - embeddings_result.get("f1_score", 0)) 
                     / embeddings_result.get("f1_score", 1) * 100)
    
    click.echo(f"\n{'='*70}")
    if f1_improvement > 5:
        click.echo("💡 RECOMMENDATION: Reranker provides significant accuracy improvement!")
        click.echo(f"   F1 score improved by {f1_improvement:.1f}% with {latency_overhead:.1f}% latency overhead")
    elif f1_improvement > 1:
        click.echo("✅ RECOMMENDATION: Reranker provides moderate improvement")
        click.echo(f"   Consider enabling for high-accuracy use cases")
    else:
        click.echo("⚖️  RECOMMENDATION: Marginal improvement, embeddings-only may be sufficient")
        click.echo(f"   Reranker overhead ({latency_overhead:.1f}%) may not justify {f1_improvement:.1f}% gain")


if __name__ == "__main__":
    cli()
