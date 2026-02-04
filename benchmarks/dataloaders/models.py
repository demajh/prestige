"""Baseline models for dataloader benchmark evaluation.

This module provides simple, reproducible baseline models for
measuring the impact of deduplication on model performance.

Models are intentionally simple to:
1. Minimize training time for benchmarks
2. Ensure reproducibility across environments
3. Focus on data quality effects, not model architecture
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Try importing sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import cross_val_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class TrainingResult:
    """Results from model training."""

    train_accuracy: float
    val_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    train_f1: Optional[float] = None
    test_f1: Optional[float] = None
    epochs_to_converge: Optional[int] = None
    training_time_sec: float = 0.0
    model_params: Dict[str, Any] = field(default_factory=dict)


class BaselineModel(ABC):
    """Abstract base class for benchmark models."""

    @abstractmethod
    def fit(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> TrainingResult:
        """Train the model.

        Args:
            texts: Training texts
            labels: Training labels
            val_texts: Optional validation texts
            val_labels: Optional validation labels

        Returns:
            TrainingResult with training metrics
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> List[int]:
        """Predict labels for texts.

        Args:
            texts: Input texts

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            texts: Test texts
            labels: True labels

        Returns:
            Dictionary with accuracy, f1, etc.
        """
        pass


class TfidfLogisticRegression(BaselineModel):
    """TF-IDF + Logistic Regression baseline.

    Simple and fast baseline that works well for text classification.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """Initialize model.

        Args:
            max_features: Maximum TF-IDF vocabulary size
            ngram_range: N-gram range for TF-IDF
            max_iter: Maximum iterations for logistic regression
            random_state: Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TfidfLogisticRegression")

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> TrainingResult:
        import time

        start_time = time.time()

        # Fit TF-IDF
        X_train = self.vectorizer.fit_transform(texts)
        y_train = np.array(labels)

        # Fit classifier
        self.classifier.fit(X_train, y_train)
        self._is_fitted = True

        training_time = time.time() - start_time

        # Compute training accuracy
        train_preds = self.classifier.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, average="weighted")

        # Compute validation accuracy if provided
        val_acc = None
        if val_texts is not None and val_labels is not None:
            X_val = self.vectorizer.transform(val_texts)
            val_preds = self.classifier.predict(X_val)
            val_acc = accuracy_score(val_labels, val_preds)

        return TrainingResult(
            train_accuracy=float(train_acc),
            train_f1=float(train_f1),
            val_accuracy=float(val_acc) if val_acc is not None else None,
            training_time_sec=training_time,
            model_params={
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "max_iter": self.max_iter,
            },
        )

    def predict(self, texts: List[str]) -> List[int]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X).tolist()

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        X = self.vectorizer.transform(texts)
        y = np.array(labels)
        preds = self.classifier.predict(X)

        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1_weighted": float(f1_score(y, preds, average="weighted")),
            "f1_macro": float(f1_score(y, preds, average="macro")),
        }


class TfidfSVM(BaselineModel):
    """TF-IDF + Linear SVM baseline."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TfidfSVM")

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.classifier = LinearSVC(
            max_iter=max_iter,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> TrainingResult:
        import time

        start_time = time.time()

        X_train = self.vectorizer.fit_transform(texts)
        y_train = np.array(labels)

        self.classifier.fit(X_train, y_train)
        self._is_fitted = True

        training_time = time.time() - start_time

        train_preds = self.classifier.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, average="weighted")

        val_acc = None
        if val_texts is not None and val_labels is not None:
            X_val = self.vectorizer.transform(val_texts)
            val_preds = self.classifier.predict(X_val)
            val_acc = accuracy_score(val_labels, val_preds)

        return TrainingResult(
            train_accuracy=float(train_acc),
            train_f1=float(train_f1),
            val_accuracy=float(val_acc) if val_acc is not None else None,
            training_time_sec=training_time,
            model_params={
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "max_iter": self.max_iter,
            },
        )

    def predict(self, texts: List[str]) -> List[int]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X).tolist()

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        X = self.vectorizer.transform(texts)
        y = np.array(labels)
        preds = self.classifier.predict(X)

        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1_weighted": float(f1_score(y, preds, average="weighted")),
            "f1_macro": float(f1_score(y, preds, average="macro")),
        }


class TfidfMLP(BaselineModel):
    """TF-IDF + Multi-Layer Perceptron baseline."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        max_iter: int = 200,
        early_stopping: bool = True,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TfidfMLP")

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.classifier = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> TrainingResult:
        import time

        start_time = time.time()

        X_train = self.vectorizer.fit_transform(texts)
        y_train = np.array(labels)

        self.classifier.fit(X_train, y_train)
        self._is_fitted = True

        training_time = time.time() - start_time

        train_preds = self.classifier.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds, average="weighted")

        # Get epochs to converge
        epochs = self.classifier.n_iter_

        val_acc = None
        if val_texts is not None and val_labels is not None:
            X_val = self.vectorizer.transform(val_texts)
            val_preds = self.classifier.predict(X_val)
            val_acc = accuracy_score(val_labels, val_preds)

        return TrainingResult(
            train_accuracy=float(train_acc),
            train_f1=float(train_f1),
            val_accuracy=float(val_acc) if val_acc is not None else None,
            epochs_to_converge=int(epochs),
            training_time_sec=training_time,
            model_params={
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "hidden_layer_sizes": self.hidden_layer_sizes,
                "max_iter": self.max_iter,
            },
        )

    def predict(self, texts: List[str]) -> List[int]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X).tolist()

    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        X = self.vectorizer.transform(texts)
        y = np.array(labels)
        preds = self.classifier.predict(X)

        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1_weighted": float(f1_score(y, preds, average="weighted")),
            "f1_macro": float(f1_score(y, preds, average="macro")),
        }


# Model registry
MODEL_REGISTRY = {
    "logistic_regression": TfidfLogisticRegression,
    "tfidf_lr": TfidfLogisticRegression,
    "svm": TfidfSVM,
    "tfidf_svm": TfidfSVM,
    "mlp": TfidfMLP,
    "tfidf_mlp": TfidfMLP,
}


def get_model(
    model_type: str,
    random_state: int = 42,
    **kwargs,
) -> BaselineModel:
    """Get a baseline model by type.

    Args:
        model_type: Model type (e.g., "logistic_regression", "svm", "mlp")
        random_state: Random seed for reproducibility
        **kwargs: Additional model-specific parameters

    Returns:
        BaselineModel instance

    Raises:
        KeyError: If model type is not found
    """
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model type: {model_type}. Available: {available}")

    model_class = MODEL_REGISTRY[model_type]
    return model_class(random_state=random_state, **kwargs)


def cross_validate_model(
    model: BaselineModel,
    texts: List[str],
    labels: List[int],
    cv: int = 5,
) -> Dict[str, Any]:
    """Perform cross-validation on a model.

    Args:
        model: BaselineModel to evaluate
        texts: All texts
        labels: All labels
        cv: Number of folds

    Returns:
        Dictionary with CV scores and statistics
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for cross-validation")

    from sklearn.pipeline import Pipeline

    # Create a fresh pipeline for CV
    if isinstance(model, TfidfLogisticRegression):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=model.max_features,
                ngram_range=model.ngram_range,
            )),
            ("clf", LogisticRegression(
                max_iter=model.max_iter,
                random_state=model.random_state,
                n_jobs=-1,
            )),
        ])
    elif isinstance(model, TfidfSVM):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=model.max_features,
                ngram_range=model.ngram_range,
            )),
            ("clf", LinearSVC(
                max_iter=model.max_iter,
                random_state=model.random_state,
            )),
        ])
    elif isinstance(model, TfidfMLP):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=model.max_features,
                ngram_range=model.ngram_range,
            )),
            ("clf", MLPClassifier(
                hidden_layer_sizes=model.hidden_layer_sizes,
                max_iter=model.max_iter,
                random_state=model.random_state,
            )),
        ])
    else:
        raise ValueError(f"Cross-validation not supported for {type(model)}")

    scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="accuracy")

    return {
        "cv_scores": scores.tolist(),
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "cv_min": float(np.min(scores)),
        "cv_max": float(np.max(scores)),
        "num_folds": cv,
    }


def list_models() -> List[str]:
    """List available model types."""
    return list(MODEL_REGISTRY.keys())
