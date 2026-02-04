"""Benchmark implementations for dataloader evaluation."""

from .generalization import (
    bench_test_accuracy_with_dedup,
    bench_overfitting_reduction,
    bench_convergence_speed,
    bench_generalization_by_threshold,
    bench_sample_efficiency,
    bench_cross_validation_variance,
    GeneralizationBenchmark,
)
from .contamination import (
    bench_contamination_rate,
    bench_metric_inflation_estimate,
    bench_contamination_by_threshold,
    bench_clean_test_performance,
    bench_cross_validation_leakage,
    ContaminationBenchmark,
)
from .detection_quality import (
    bench_precision_recall_curve,
    bench_false_positive_analysis,
    bench_threshold_sensitivity,
    bench_label_preservation,
    bench_rare_class_impact,
    DetectionQualityBenchmark,
)
from .mode_comparison import (
    bench_mode_accuracy_comparison,
    bench_semantic_catches_paraphrases,
    bench_paraphrase_impact_on_model,
    bench_threshold_tuning,
    ModeComparisonBenchmark,
)
from .statistical import (
    bench_multi_seed_variance,
    bench_confidence_intervals,
    bench_effect_size,
    StatisticalBenchmark,
)
from .performance import (
    bench_throughput,
    bench_memory_usage,
    bench_gpu_speedup,
    PerformanceBenchmark,
)

__all__ = [
    # Generalization (PRIMARY)
    "bench_test_accuracy_with_dedup",
    "bench_overfitting_reduction",
    "bench_convergence_speed",
    "bench_generalization_by_threshold",
    "bench_sample_efficiency",
    "bench_cross_validation_variance",
    "GeneralizationBenchmark",
    # Contamination
    "bench_contamination_rate",
    "bench_metric_inflation_estimate",
    "bench_contamination_by_threshold",
    "bench_clean_test_performance",
    "bench_cross_validation_leakage",
    "ContaminationBenchmark",
    # Detection Quality
    "bench_precision_recall_curve",
    "bench_false_positive_analysis",
    "bench_threshold_sensitivity",
    "bench_label_preservation",
    "bench_rare_class_impact",
    "DetectionQualityBenchmark",
    # Mode Comparison
    "bench_mode_accuracy_comparison",
    "bench_semantic_catches_paraphrases",
    "bench_paraphrase_impact_on_model",
    "bench_threshold_tuning",
    "ModeComparisonBenchmark",
    # Statistical
    "bench_multi_seed_variance",
    "bench_confidence_intervals",
    "bench_effect_size",
    "StatisticalBenchmark",
    # Performance (secondary)
    "bench_throughput",
    "bench_memory_usage",
    "bench_gpu_speedup",
    "PerformanceBenchmark",
]
