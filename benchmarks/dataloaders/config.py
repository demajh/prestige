"""Configuration for dataloader benchmarks.

This module defines configuration dataclasses for benchmark execution,
including statistical parameters for rigorous evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""

    GENERALIZATION = "generalization"
    CONTAMINATION = "contamination"
    DETECTION_QUALITY = "detection_quality"
    MODE_COMPARISON = "mode_comparison"
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    ALL = "all"


class DedupMode(Enum):
    """Deduplication mode for benchmarks."""

    EXACT = "exact"
    SEMANTIC = "semantic"
    BOTH = "both"


@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis.

    Controls reproducibility and statistical rigor of benchmark results.
    """

    # Number of random seeds for multi-seed evaluation
    num_seeds: int = 5

    # Base random seed (seeds will be: base, base+1, ..., base+num_seeds-1)
    base_seed: int = 42

    # Confidence level for intervals (0.95 = 95% CI)
    confidence_level: float = 0.95

    # Minimum effect size to consider practically significant
    min_effect_size: float = 0.2  # Cohen's d = 0.2 (small effect)

    # Significance level for hypothesis tests
    alpha: float = 0.05

    def get_seeds(self) -> List[int]:
        """Get list of random seeds."""
        return list(range(self.base_seed, self.base_seed + self.num_seeds))


@dataclass
class ModelConfig:
    """Configuration for baseline model training."""

    # Model type: "logistic_regression", "mlp", "tfidf_svm"
    model_type: str = "logistic_regression"

    # Maximum training epochs (for iterative models)
    max_epochs: int = 100

    # Early stopping patience
    early_stopping_patience: int = 5

    # Batch size for training
    batch_size: int = 32

    # Learning rate (for gradient-based models)
    learning_rate: float = 0.001

    # Whether to use GPU if available
    use_gpu: bool = True

    # TF-IDF parameters (for tfidf_* models)
    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple = (1, 2)


@dataclass
class DatasetConfig:
    """Configuration for benchmark datasets."""

    # Dataset name or path
    name: str = "synth_classification"

    # Text column name
    text_column: str = "text"

    # Label column name
    label_column: str = "label"

    # Train/test split ratio
    test_size: float = 0.2

    # Validation split (from training data)
    val_size: float = 0.1

    # Maximum samples (None = use all)
    max_samples: Optional[int] = None

    # Stratify splits by label
    stratify: bool = True


@dataclass
class DedupConfig:
    """Configuration for deduplication settings."""

    # Deduplication mode
    mode: DedupMode = DedupMode.EXACT

    # Semantic similarity threshold (for semantic mode)
    semantic_threshold: float = 0.9

    # Thresholds to sweep for threshold optimization benchmarks
    threshold_sweep: List[float] = field(
        default_factory=lambda: [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
    )

    # Embedding model for semantic dedup
    embedding_model: str = "bge-small"

    # Device for embedding inference
    device: str = "auto"  # "auto", "cpu", "gpu"

    # Store path (None = temporary)
    store_path: Optional[Path] = None


@dataclass
class BenchmarkConfig:
    """Main configuration for benchmark execution.

    This is the primary configuration class that combines all settings.

    Example:
        >>> config = BenchmarkConfig(
        ...     categories=[BenchmarkCategory.GENERALIZATION],
        ...     statistical=StatisticalConfig(num_seeds=3),
        ...     quick_mode=True,
        ... )
        >>> runner = BenchmarkRunner(config)
        >>> results = runner.run()
    """

    # Benchmark categories to run
    categories: List[BenchmarkCategory] = field(
        default_factory=lambda: [BenchmarkCategory.ALL]
    )

    # Statistical configuration
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Deduplication configuration
    dedup: DedupConfig = field(default_factory=DedupConfig)

    # Quick mode (fewer seeds, smaller datasets)
    quick_mode: bool = False

    # Verbose output
    verbose: bool = True

    # Output directory for results
    output_dir: Optional[Path] = None

    # Cache directory for datasets and models
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "prestige" / "benchmarks"
    )

    def __post_init__(self):
        """Apply quick mode settings if enabled."""
        if self.quick_mode:
            self.statistical.num_seeds = 2
            if self.dataset.max_samples is None:
                self.dataset.max_samples = 1000
            self.dedup.threshold_sweep = [0.85, 0.92, 0.98]

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set output directory default
        if self.output_dir is None:
            self.output_dir = self.cache_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Preset configurations for common use cases
def quick_config() -> BenchmarkConfig:
    """Configuration for quick validation runs."""
    return BenchmarkConfig(quick_mode=True)


def full_config() -> BenchmarkConfig:
    """Configuration for full statistical evaluation."""
    return BenchmarkConfig(
        statistical=StatisticalConfig(num_seeds=5),
        quick_mode=False,
    )


def generalization_config() -> BenchmarkConfig:
    """Configuration focused on generalization benchmarks."""
    return BenchmarkConfig(
        categories=[BenchmarkCategory.GENERALIZATION],
        statistical=StatisticalConfig(num_seeds=5),
    )


def contamination_config() -> BenchmarkConfig:
    """Configuration focused on contamination detection benchmarks."""
    return BenchmarkConfig(
        categories=[BenchmarkCategory.CONTAMINATION],
    )
