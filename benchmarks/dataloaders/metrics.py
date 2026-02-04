"""Statistical metrics for dataloader benchmarks.

This module provides rigorous statistical analysis tools including:
- Confidence intervals
- Effect sizes (Cohen's d)
- Hypothesis testing
- Class distribution metrics
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Confidence interval with point estimate."""

    mean: float
    lower: float
    upper: float
    std: float
    confidence_level: float = 0.95
    n_samples: int = 0

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def contains(self, value: float) -> bool:
        """Check if value is within the confidence interval."""
        return self.lower <= value <= self.upper

    def margin_of_error(self) -> float:
        """Calculate margin of error."""
        return (self.upper - self.lower) / 2


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Compute confidence interval for a list of values.

    Uses t-distribution for small samples, normal for large samples.

    Args:
        values: List of measurements
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        ConfidenceInterval with mean and bounds
    """
    if len(values) == 0:
        return ConfidenceInterval(
            mean=0.0, lower=0.0, upper=0.0, std=0.0,
            confidence_level=confidence, n_samples=0
        )

    if len(values) == 1:
        v = values[0]
        return ConfidenceInterval(
            mean=v, lower=v, upper=v, std=0.0,
            confidence_level=confidence, n_samples=1
        )

    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # Sample std
    se = std / math.sqrt(n)

    # Use t-distribution for small samples
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin = t_crit * se
    lower = mean - margin
    upper = mean + margin

    return ConfidenceInterval(
        mean=float(mean),
        lower=float(lower),
        upper=float(upper),
        std=float(std),
        confidence_level=confidence,
        n_samples=n,
    )


@dataclass
class EffectSize:
    """Effect size metrics."""

    cohens_d: float
    interpretation: str  # "negligible", "small", "medium", "large"
    pooled_std: float

    @classmethod
    def compute(
        cls,
        group1: List[float],
        group2: List[float],
    ) -> "EffectSize":
        """Compute Cohen's d effect size between two groups.

        Args:
            group1: Control group (e.g., original data accuracy)
            group2: Treatment group (e.g., deduped data accuracy)

        Returns:
            EffectSize with Cohen's d and interpretation
        """
        if len(group1) < 2 or len(group2) < 2:
            return cls(cohens_d=0.0, interpretation="insufficient_data", pooled_std=0.0)

        arr1 = np.array(group1)
        arr2 = np.array(group2)

        n1, n2 = len(arr1), len(arr2)
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        var1 = np.var(arr1, ddof=1)
        var2 = np.var(arr2, ddof=1)

        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (mean2 - mean1) / pooled_std

        # Interpret effect size
        d_abs = abs(cohens_d)
        if d_abs < 0.2:
            interpretation = "negligible"
        elif d_abs < 0.5:
            interpretation = "small"
        elif d_abs < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return cls(
            cohens_d=float(cohens_d),
            interpretation=interpretation,
            pooled_std=float(pooled_std),
        )


@dataclass
class HypothesisTest:
    """Results of a hypothesis test."""

    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    test_name: str = "t-test"

    def __str__(self) -> str:
        sig = "**" if self.p_value < 0.01 else "*" if self.is_significant else ""
        return f"p={self.p_value:.4f}{sig}"


def paired_t_test(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTest:
    """Perform paired t-test between two groups.

    Use when comparing same model trained on original vs deduped data
    across multiple random seeds.

    Args:
        group1: Control measurements (e.g., original data accuracy per seed)
        group2: Treatment measurements (e.g., deduped data accuracy per seed)
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"

    Returns:
        HypothesisTest with statistic and p-value
    """
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired t-test")

    if len(group1) < 2:
        return HypothesisTest(
            statistic=0.0, p_value=1.0, is_significant=False,
            alpha=alpha, test_name="paired_t_test"
        )

    result = stats.ttest_rel(group2, group1, alternative=alternative)

    return HypothesisTest(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        is_significant=result.pvalue < alpha,
        alpha=alpha,
        test_name="paired_t_test",
    )


def independent_t_test(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTest:
    """Perform independent samples t-test.

    Use when comparing independent measurements.

    Args:
        group1: First group measurements
        group2: Second group measurements
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"

    Returns:
        HypothesisTest with statistic and p-value
    """
    if len(group1) < 2 or len(group2) < 2:
        return HypothesisTest(
            statistic=0.0, p_value=1.0, is_significant=False,
            alpha=alpha, test_name="independent_t_test"
        )

    result = stats.ttest_ind(group2, group1, alternative=alternative)

    return HypothesisTest(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        is_significant=result.pvalue < alpha,
        alpha=alpha,
        test_name="independent_t_test",
    )


@dataclass
class ClassDistribution:
    """Class distribution statistics."""

    counts: Dict[int, int]
    proportions: Dict[int, float]
    total: int
    num_classes: int

    @classmethod
    def from_labels(cls, labels: List[int]) -> "ClassDistribution":
        """Compute class distribution from labels.

        Args:
            labels: List of class labels

        Returns:
            ClassDistribution with counts and proportions
        """
        counts: Dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

        total = len(labels)
        proportions = {k: v / total for k, v in counts.items()} if total > 0 else {}

        return cls(
            counts=counts,
            proportions=proportions,
            total=total,
            num_classes=len(counts),
        )


def kl_divergence(
    dist1: ClassDistribution,
    dist2: ClassDistribution,
    epsilon: float = 1e-10,
) -> float:
    """Compute KL divergence between two class distributions.

    KL(dist1 || dist2) measures how much dist1 diverges from dist2.
    Lower values indicate more similar distributions.

    Args:
        dist1: First distribution (e.g., original)
        dist2: Second distribution (e.g., after dedup)
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence (non-negative, 0 = identical)
    """
    all_classes = set(dist1.proportions.keys()) | set(dist2.proportions.keys())

    kl = 0.0
    for cls in all_classes:
        p = dist1.proportions.get(cls, epsilon)
        q = dist2.proportions.get(cls, epsilon)
        if p > epsilon:
            kl += p * math.log(p / q)

    return max(0.0, kl)


def minority_class_retention(
    original_dist: ClassDistribution,
    deduped_dist: ClassDistribution,
    minority_threshold: float = 0.1,
) -> Dict[int, float]:
    """Calculate retention rate for minority classes.

    Args:
        original_dist: Distribution before deduplication
        deduped_dist: Distribution after deduplication
        minority_threshold: Classes with proportion < this are minority

    Returns:
        Dictionary mapping minority class -> retention rate
    """
    retention = {}
    for cls, prop in original_dist.proportions.items():
        if prop < minority_threshold:
            orig_count = original_dist.counts[cls]
            dedup_count = deduped_dist.counts.get(cls, 0)
            retention[cls] = dedup_count / orig_count if orig_count > 0 else 0.0
    return retention


@dataclass
class DedupQualityMetrics:
    """Metrics for deduplication quality evaluation."""

    # True positives: Correctly identified duplicates
    true_positives: int = 0
    # False positives: Unique items incorrectly marked as duplicates
    false_positives: int = 0
    # True negatives: Unique items correctly kept
    true_negatives: int = 0
    # False negatives: Duplicates missed
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / total."""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN)."""
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    def update(self, predicted_duplicate: bool, is_true_duplicate: bool) -> None:
        """Update metrics with a prediction.

        Args:
            predicted_duplicate: Whether the system marked as duplicate
            is_true_duplicate: Ground truth duplicate status
        """
        if is_true_duplicate and predicted_duplicate:
            self.true_positives += 1
        elif is_true_duplicate and not predicted_duplicate:
            self.false_negatives += 1
        elif not is_true_duplicate and predicted_duplicate:
            self.false_positives += 1
        else:
            self.true_negatives += 1


@dataclass
class GeneralizationMetrics:
    """Metrics for model generalization evaluation."""

    # Accuracy measurements across seeds
    train_accuracies: List[float] = field(default_factory=list)
    test_accuracies: List[float] = field(default_factory=list)

    # Baseline (original data) measurements
    baseline_train_accuracies: List[float] = field(default_factory=list)
    baseline_test_accuracies: List[float] = field(default_factory=list)

    # Epochs to convergence (if applicable)
    epochs_to_convergence: List[int] = field(default_factory=list)
    baseline_epochs_to_convergence: List[int] = field(default_factory=list)

    def test_accuracy_improvement(self) -> ConfidenceInterval:
        """Compute improvement in test accuracy."""
        if not self.test_accuracies or not self.baseline_test_accuracies:
            return ConfidenceInterval(0, 0, 0, 0, 0.95, 0)

        improvements = [
            t - b for t, b in zip(self.test_accuracies, self.baseline_test_accuracies)
        ]
        return compute_confidence_interval(improvements)

    def train_test_gap_reduction(self) -> ConfidenceInterval:
        """Compute reduction in train/test gap (overfitting measure)."""
        if not self.train_accuracies or not self.test_accuracies:
            return ConfidenceInterval(0, 0, 0, 0, 0.95, 0)

        # Gap = train_acc - test_acc (positive means overfitting)
        dedup_gaps = [
            tr - te for tr, te in zip(self.train_accuracies, self.test_accuracies)
        ]
        baseline_gaps = [
            tr - te for tr, te in zip(self.baseline_train_accuracies, self.baseline_test_accuracies)
        ]

        # Reduction = baseline_gap - dedup_gap (positive means less overfitting)
        reductions = [bg - dg for bg, dg in zip(baseline_gaps, dedup_gaps)]
        return compute_confidence_interval(reductions)

    def effect_size(self) -> EffectSize:
        """Compute Cohen's d for test accuracy improvement."""
        return EffectSize.compute(
            self.baseline_test_accuracies,
            self.test_accuracies,
        )

    def hypothesis_test(self, alpha: float = 0.05) -> HypothesisTest:
        """Test if improvement is statistically significant."""
        return paired_t_test(
            self.baseline_test_accuracies,
            self.test_accuracies,
            alpha=alpha,
            alternative="greater",  # Testing if dedup improves accuracy
        )

    def sample_efficiency(self) -> Optional[float]:
        """Compute sample efficiency improvement.

        Returns accuracy per training sample ratio (dedup / baseline).
        """
        # This requires knowing the dataset sizes, which should be passed separately
        return None


@dataclass
class ContaminationMetrics:
    """Metrics for contamination detection."""

    # Number of test samples found in training data
    contaminated_count: int = 0
    # Total test samples checked
    total_test_samples: int = 0
    # Indices of contaminated samples
    contaminated_indices: List[int] = field(default_factory=list)

    @property
    def contamination_rate(self) -> float:
        """Fraction of test set that is contaminated."""
        if self.total_test_samples == 0:
            return 0.0
        return self.contaminated_count / self.total_test_samples

    @property
    def leakage_severity(self) -> str:
        """Classify leakage severity."""
        rate = self.contamination_rate
        if rate < 0.001:
            return "none"
        elif rate < 0.01:
            return "low"
        elif rate < 0.05:
            return "medium"
        else:
            return "high"


@dataclass
class BenchmarkResult:
    """Complete benchmark result with all metrics."""

    # Identification
    benchmark_name: str
    dataset_name: str
    dedup_mode: str
    threshold: Optional[float] = None

    # Primary metrics
    generalization: Optional[GeneralizationMetrics] = None
    contamination: Optional[ContaminationMetrics] = None
    dedup_quality: Optional[DedupQualityMetrics] = None

    # Class distribution
    original_distribution: Optional[ClassDistribution] = None
    deduped_distribution: Optional[ClassDistribution] = None

    # Performance (secondary)
    throughput_samples_per_sec: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    processing_time_sec: Optional[float] = None

    # Statistical significance
    effect_size: Optional[EffectSize] = None
    hypothesis_test: Optional[HypothesisTest] = None

    # Metadata
    config_hash: Optional[str] = None
    timestamp: Optional[str] = None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Dataset: {self.dataset_name}",
            f"Mode: {self.dedup_mode}",
        ]

        if self.threshold is not None:
            lines.append(f"Threshold: {self.threshold}")

        if self.generalization:
            ci = self.generalization.test_accuracy_improvement()
            lines.append(f"\nTest Accuracy Improvement: {ci}")

            gap = self.generalization.train_test_gap_reduction()
            lines.append(f"Train/Test Gap Reduction: {gap}")

            es = self.generalization.effect_size()
            lines.append(f"Effect Size (Cohen's d): {es.cohens_d:.3f} ({es.interpretation})")

            ht = self.generalization.hypothesis_test()
            lines.append(f"Statistical Significance: {ht}")

        if self.contamination:
            lines.append(f"\nContamination Rate: {self.contamination.contamination_rate:.2%}")
            lines.append(f"Leakage Severity: {self.contamination.leakage_severity}")

        if self.dedup_quality:
            lines.append(f"\nDedup Precision: {self.dedup_quality.precision:.4f}")
            lines.append(f"Dedup Recall: {self.dedup_quality.recall:.4f}")
            lines.append(f"Dedup F1: {self.dedup_quality.f1_score:.4f}")

        return "\n".join(lines)
