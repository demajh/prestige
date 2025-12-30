"""Metrics calculation for semantic deduplication benchmarks."""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification."""
    tp: int = 0  # True positives
    fp: int = 0  # False positives
    tn: int = 0  # True negatives
    fn: int = 0  # False negatives

    def update(self, predicted: bool, ground_truth: bool):
        """Update confusion matrix with a new prediction.

        Args:
            predicted: Predicted label (did deduplicate)
            ground_truth: Ground truth label (should deduplicate)
        """
        if ground_truth and predicted:
            self.tp += 1
        elif ground_truth and not predicted:
            self.fn += 1
        elif not ground_truth and predicted:
            self.fp += 1
        else:
            self.tn += 1

    def total(self) -> int:
        """Total number of samples."""
        return self.tp + self.fp + self.tn + self.fn


@dataclass
class BenchmarkMetrics:
    """Computed metrics from confusion matrix."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @classmethod
    def from_confusion_matrix(cls, cm: ConfusionMatrix) -> "BenchmarkMetrics":
        """Compute metrics from confusion matrix.

        Args:
            cm: Confusion matrix

        Returns:
            Computed metrics
        """
        precision = compute_precision(cm.tp, cm.fp)
        recall = compute_recall(cm.tp, cm.fn)
        f1 = compute_f1(precision, recall)
        accuracy = compute_accuracy(cm.tp, cm.tn, cm.fp, cm.fn)

        return cls(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            true_positives=cm.tp,
            false_positives=cm.fp,
            true_negatives=cm.tn,
            false_negatives=cm.fn,
        )


def compute_precision(tp: int, fp: int) -> float:
    """Compute precision = TP / (TP + FP).

    Args:
        tp: True positives
        fp: False positives

    Returns:
        Precision score (0.0-1.0)
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def compute_recall(tp: int, fn: int) -> float:
    """Compute recall = TP / (TP + FN).

    Args:
        tp: True positives
        fn: False negatives

    Returns:
        Recall score (0.0-1.0)
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score = 2 * (precision * recall) / (precision + recall).

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score (0.0-1.0)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute accuracy = (TP + TN) / (TP + TN + FP + FN).

    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives

    Returns:
        Accuracy score (0.0-1.0)
    """
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total


class MetricsAggregator:
    """Aggregates metrics across multiple thresholds."""

    def __init__(self):
        """Initialize metrics aggregator."""
        self.threshold_metrics = []

    def add_threshold_result(
        self,
        threshold: float,
        confusion_matrix: ConfusionMatrix,
        latencies: List[float],
        dedup_ratio: float,
        storage_bytes: int,
        unique_objects: int,
        total_keys: int,
    ):
        """Add results for a specific threshold.

        Args:
            threshold: Threshold value
            confusion_matrix: Confusion matrix for this threshold
            latencies: List of operation latencies (seconds)
            dedup_ratio: Deduplication ratio (keys per unique value)
            storage_bytes: Total storage used
            unique_objects: Number of unique objects
            total_keys: Total number of keys
        """
        metrics = BenchmarkMetrics.from_confusion_matrix(confusion_matrix)

        result = {
            "threshold": threshold,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "accuracy": metrics.accuracy,
            "tp": metrics.true_positives,
            "fp": metrics.false_positives,
            "tn": metrics.true_negatives,
            "fn": metrics.false_negatives,
            "dedup_ratio": dedup_ratio,
            "latency_p50_ms": np.percentile(latencies, 50) * 1000 if latencies else 0,
            "latency_p95_ms": np.percentile(latencies, 95) * 1000 if latencies else 0,
            "latency_p99_ms": np.percentile(latencies, 99) * 1000 if latencies else 0,
            "storage_bytes": storage_bytes,
            "unique_objects": unique_objects,
            "total_keys": total_keys,
        }

        self.threshold_metrics.append(result)

    def get_best_f1(self) -> dict:
        """Get threshold with best F1 score.

        Returns:
            Metrics dict for best F1 threshold
        """
        if not self.threshold_metrics:
            return {}

        return max(self.threshold_metrics, key=lambda x: x["f1_score"])

    def get_all_results(self) -> List[dict]:
        """Get all threshold results.

        Returns:
            List of metrics dicts
        """
        return self.threshold_metrics
