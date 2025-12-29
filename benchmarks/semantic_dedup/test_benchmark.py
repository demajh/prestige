"""Tests for semantic deduplication benchmark harness."""

import unittest
import tempfile
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

from .metrics import (
    ConfusionMatrix,
    BenchmarkMetrics,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_accuracy,
    compute_auc,
    MetricsAggregator,
)
from .dataset_loader import DatasetConfig, DatasetLoader
from .report import ReportGenerator, BaselineComparator, BenchmarkReport


class TestMetrics(unittest.TestCase):
    """Test metric calculations."""

    def test_confusion_matrix_update(self):
        """Test confusion matrix updates correctly."""
        cm = ConfusionMatrix()

        # True positive
        cm.update(predicted=True, ground_truth=True)
        self.assertEqual(cm.tp, 1)

        # False positive
        cm.update(predicted=True, ground_truth=False)
        self.assertEqual(cm.fp, 1)

        # True negative
        cm.update(predicted=False, ground_truth=False)
        self.assertEqual(cm.tn, 1)

        # False negative
        cm.update(predicted=False, ground_truth=True)
        self.assertEqual(cm.fn, 1)

        self.assertEqual(cm.total(), 4)

    def test_precision_calculation(self):
        """Test precision calculation."""
        self.assertEqual(compute_precision(10, 5), 10 / 15)
        self.assertEqual(compute_precision(0, 0), 0.0)
        self.assertEqual(compute_precision(10, 0), 1.0)

    def test_recall_calculation(self):
        """Test recall calculation."""
        self.assertEqual(compute_recall(10, 5), 10 / 15)
        self.assertEqual(compute_recall(0, 0), 0.0)
        self.assertEqual(compute_recall(10, 0), 1.0)

    def test_f1_calculation(self):
        """Test F1 score calculation."""
        precision = 0.8
        recall = 0.6
        expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        self.assertAlmostEqual(compute_f1(precision, recall), expected_f1)

        self.assertEqual(compute_f1(0.0, 0.0), 0.0)

    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        self.assertEqual(compute_accuracy(10, 20, 5, 5), (10 + 20) / 40)
        self.assertEqual(compute_accuracy(0, 0, 0, 0), 0.0)

    def test_auc_calculation(self):
        """Test AUC calculation with trapezoidal rule."""
        # Perfect classifier: (0,0) -> (0,1) -> (1,1)
        x = [0.0, 0.0, 1.0]
        y = [0.0, 1.0, 1.0]
        auc = compute_auc(x, y)
        self.assertAlmostEqual(auc, 1.0)

        # Random classifier: (0,0) -> (1,1)
        x = [0.0, 1.0]
        y = [0.0, 1.0]
        auc = compute_auc(x, y)
        self.assertAlmostEqual(auc, 0.5)

    def test_benchmark_metrics_from_confusion_matrix(self):
        """Test BenchmarkMetrics creation from confusion matrix."""
        cm = ConfusionMatrix(tp=10, fp=2, tn=15, fn=3)
        metrics = BenchmarkMetrics.from_confusion_matrix(cm)

        self.assertAlmostEqual(metrics.precision, 10 / 12)
        self.assertAlmostEqual(metrics.recall, 10 / 13)
        self.assertEqual(metrics.true_positives, 10)
        self.assertEqual(metrics.false_positives, 2)


class TestMetricsAggregator(unittest.TestCase):
    """Test metrics aggregation."""

    def test_add_threshold_result(self):
        """Test adding threshold results."""
        agg = MetricsAggregator()

        cm = ConfusionMatrix(tp=10, fp=2, tn=15, fn=3)
        agg.add_threshold_result(
            threshold=0.9,
            confusion_matrix=cm,
            latencies=[0.001, 0.002, 0.003],
            dedup_ratio=1.5,
            storage_bytes=1000,
            unique_objects=100,
            total_keys=150,
        )

        results = agg.get_all_results()
        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result["threshold"], 0.9)
        self.assertAlmostEqual(result["precision"], 10 / 12)
        self.assertEqual(result["tp"], 10)

    def test_get_best_f1(self):
        """Test getting best F1 threshold."""
        agg = MetricsAggregator()

        # Add results with different F1 scores
        for threshold, f1 in [(0.85, 0.7), (0.90, 0.85), (0.95, 0.75)]:
            # Calculate TP/FP to achieve desired F1 (approximately)
            tp = int(f1 * 100)
            fp = int((100 - f1 * 100) / 2)
            fn = int((100 - f1 * 100) / 2)
            tn = 100

            cm = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
            agg.add_threshold_result(
                threshold=threshold,
                confusion_matrix=cm,
                latencies=[0.001],
                dedup_ratio=1.0,
                storage_bytes=1000,
                unique_objects=100,
                total_keys=100,
            )

        best = agg.get_best_f1()
        # Threshold 0.90 should have highest F1
        self.assertEqual(best["threshold"], 0.90)


class TestDatasetLoader(unittest.TestCase):
    """Test dataset loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)
        self.loader = DatasetLoader(self.cache_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_normalize_columns(self):
        """Test column normalization."""
        df = pd.DataFrame({
            "sentence1": ["Hello world", "Foo bar"],
            "sentence2": ["Hi there", "Baz qux"],
            "label": [1, 0],
        })

        config = DatasetConfig(
            name="test",
            source="test",
            text_columns=["sentence1", "sentence2"],
            label_column="label",
            positive_label=1,
        )

        result = self.loader._normalize_columns(df, config)

        self.assertIn("text1", result.columns)
        self.assertIn("text2", result.columns)
        self.assertIn("is_duplicate", result.columns)

        self.assertEqual(result["is_duplicate"].iloc[0], True)
        self.assertEqual(result["is_duplicate"].iloc[1], False)

    def test_stratified_sample(self):
        """Test stratified sampling preserves class balance."""
        # Create unbalanced dataset
        df = pd.DataFrame({
            "text1": ["text"] * 1000,
            "text2": ["text"] * 1000,
            "is_duplicate": [True] * 700 + [False] * 300,
        })

        result = self.loader._stratified_sample(df, 100)

        self.assertEqual(len(result), 100)

        # Check class balance is approximately preserved
        pos_ratio = result["is_duplicate"].sum() / len(result)
        expected_ratio = 0.7
        self.assertAlmostEqual(pos_ratio, expected_ratio, delta=0.1)


class TestReportGeneration(unittest.TestCase):
    """Test report generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_report_generation(self):
        """Test generating a report."""
        agg = MetricsAggregator()

        cm = ConfusionMatrix(tp=10, fp=2, tn=15, fn=3)
        agg.add_threshold_result(
            threshold=0.9,
            confusion_matrix=cm,
            latencies=[0.001, 0.002, 0.003],
            dedup_ratio=1.5,
            storage_bytes=1000,
            unique_objects=100,
            total_keys=150,
        )

        gen = ReportGenerator("test_dataset", agg)
        report = gen.generate_report()

        self.assertEqual(report.dataset_name, "test_dataset")
        self.assertEqual(len(report.results), 1)
        self.assertGreater(report.best_f1["f1_score"], 0)

    def test_json_save(self):
        """Test saving JSON report."""
        agg = MetricsAggregator()

        cm = ConfusionMatrix(tp=10, fp=2, tn=15, fn=3)
        agg.add_threshold_result(
            threshold=0.9,
            confusion_matrix=cm,
            latencies=[0.001],
            dedup_ratio=1.5,
            storage_bytes=1000,
            unique_objects=100,
            total_keys=150,
        )

        gen = ReportGenerator("test", agg)
        output_path = Path(self.temp_dir) / "report.json"
        gen.save_json(output_path)

        self.assertTrue(output_path.exists())

        # Verify can load back
        from .report import load_report
        loaded = load_report(output_path)
        self.assertEqual(loaded.dataset_name, "test")

    def test_html_generation(self):
        """Test HTML report generation."""
        agg = MetricsAggregator()

        cm = ConfusionMatrix(tp=10, fp=2, tn=15, fn=3)
        agg.add_threshold_result(
            threshold=0.9,
            confusion_matrix=cm,
            latencies=[0.001],
            dedup_ratio=1.5,
            storage_bytes=1000,
            unique_objects=100,
            total_keys=150,
        )

        gen = ReportGenerator("test", agg)
        output_path = Path(self.temp_dir) / "report.html"
        gen.save_html(output_path)

        self.assertTrue(output_path.exists())

        # Verify HTML contains key elements
        html_content = output_path.read_text()
        self.assertIn("Semantic Deduplication Benchmark", html_content)
        self.assertIn("test", html_content.lower())


class TestBaselineComparison(unittest.TestCase):
    """Test baseline comparison."""

    def test_regression_detection(self):
        """Test regression detection."""
        # Create baseline with good F1
        baseline = BenchmarkReport(
            timestamp="2025-01-01T00:00:00Z",
            dataset_name="test",
            thresholds=[0.9],
            results=[],
            best_f1={"f1_score": 0.9, "threshold": 0.9},
            roc_auc=0.95,
            pr_auc=0.93,
            summary={"avg_latency_p95_ms": 5.0},
        )

        # Create current with worse F1 (regression)
        current = BenchmarkReport(
            timestamp="2025-01-02T00:00:00Z",
            dataset_name="test",
            thresholds=[0.9],
            results=[],
            best_f1={"f1_score": 0.8, "threshold": 0.9},
            roc_auc=0.90,
            pr_auc=0.88,
            summary={"avg_latency_p95_ms": 5.0},
        )

        comparator = BaselineComparator(current, baseline)
        regressions = comparator.detect_regressions(threshold_pct=5.0)

        # Should detect F1 and ROC_AUC regressions
        self.assertGreater(len(regressions), 0)

        regression_metrics = [r["metric"] for r in regressions]
        self.assertIn("best_f1", regression_metrics)

    def test_improvement_detection(self):
        """Test improvement detection."""
        baseline = BenchmarkReport(
            timestamp="2025-01-01T00:00:00Z",
            dataset_name="test",
            thresholds=[0.9],
            results=[],
            best_f1={"f1_score": 0.8, "threshold": 0.9},
            roc_auc=0.85,
            pr_auc=0.83,
            summary={"avg_latency_p95_ms": 5.0},
        )

        # Improved F1
        current = BenchmarkReport(
            timestamp="2025-01-02T00:00:00Z",
            dataset_name="test",
            thresholds=[0.9],
            results=[],
            best_f1={"f1_score": 0.9, "threshold": 0.9},
            roc_auc=0.90,
            pr_auc=0.88,
            summary={"avg_latency_p95_ms": 5.0},
        )

        comparator = BaselineComparator(current, baseline)
        improvements = comparator._detect_improvements()

        self.assertGreater(len(improvements), 0)
        self.assertEqual(improvements[0]["metric"], "best_f1")


if __name__ == "__main__":
    unittest.main()
