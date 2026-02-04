"""Dataset loading for dataloader benchmarks.

This module provides:
1. Real NLP datasets (MRPC, QQP, PAWS, STS-B, SST-2) via HuggingFace
2. Synthetic datasets with controlled properties for testing

Real datasets provide credible benchmarks on actual data.
Synthetic datasets enable controlled experiments with known ground truth.
"""

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json

# Try importing HuggingFace datasets
try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class SyntheticSample:
    """A single synthetic sample with metadata."""

    text: str
    label: int
    original_idx: Optional[int] = None  # Index of the original (if this is a dup)
    is_duplicate: bool = False
    is_paraphrase: bool = False
    duplicate_group_id: Optional[str] = None  # ID linking duplicates together


@dataclass
class SyntheticDataset:
    """Synthetic dataset with ground truth annotations."""

    samples: List[SyntheticSample]
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def texts(self) -> List[str]:
        """Get all text samples."""
        return [s.text for s in self.samples]

    @property
    def labels(self) -> List[int]:
        """Get all labels."""
        return [s.label for s in self.samples]

    @property
    def duplicate_mask(self) -> List[bool]:
        """Get mask indicating which samples are duplicates."""
        return [s.is_duplicate for s in self.samples]

    @property
    def num_duplicates(self) -> int:
        """Count of duplicate samples."""
        return sum(1 for s in self.samples if s.is_duplicate)

    @property
    def num_unique(self) -> int:
        """Count of unique (non-duplicate) samples."""
        return sum(1 for s in self.samples if not s.is_duplicate)

    @property
    def duplicate_rate(self) -> float:
        """Fraction of samples that are duplicates."""
        return self.num_duplicates / len(self.samples) if self.samples else 0.0

    def get_duplicate_groups(self) -> Dict[str, List[int]]:
        """Get mapping from duplicate group ID to sample indices."""
        groups: Dict[str, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            if sample.duplicate_group_id:
                if sample.duplicate_group_id not in groups:
                    groups[sample.duplicate_group_id] = []
                groups[sample.duplicate_group_id].append(idx)
        return groups

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries (for PyTorch/HF compatibility)."""
        return [
            {
                "text": s.text,
                "label": s.label,
                "_is_duplicate": s.is_duplicate,
                "_is_paraphrase": s.is_paraphrase,
                "_duplicate_group_id": s.duplicate_group_id,
                "_original_idx": s.original_idx,
            }
            for s in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SyntheticSample:
        return self.samples[idx]


# Template-based sentence generation
SENTENCE_TEMPLATES = {
    0: [  # Technology
        "The {adj} {tech} revolutionized the {field} industry.",
        "{tech} has become essential for modern {field}.",
        "Experts predict {tech} will transform {field} in the coming years.",
        "Companies are investing heavily in {tech} for {field} applications.",
    ],
    1: [  # Science
        "Researchers discovered a {adj} connection between {topic1} and {topic2}.",
        "The study of {topic1} reveals insights about {topic2}.",
        "Scientists are exploring how {topic1} affects {topic2}.",
        "New findings suggest {topic1} plays a role in {topic2}.",
    ],
    2: [  # Business
        "The {adj} market for {product} is expected to grow significantly.",
        "{product} sales have increased due to {adj} consumer demand.",
        "Industry analysts predict {adj} growth in {product} sector.",
        "Companies are launching {adj} {product} to meet market needs.",
    ],
    3: [  # Health
        "A {adj} approach to {health_topic} shows promising results.",
        "Studies indicate {health_topic} is linked to {outcome}.",
        "Medical experts recommend {adj} practices for {health_topic}.",
        "New research on {health_topic} suggests {outcome}.",
    ],
    4: [  # Environment
        "The {adj} impact of {env_factor} on {env_system} is significant.",
        "Scientists warn about {env_factor} affecting {env_system}.",
        "Conservation efforts focus on protecting {env_system} from {env_factor}.",
        "Environmental studies show {env_factor} changes in {env_system}.",
    ],
}

TEMPLATE_FILLERS = {
    "adj": ["innovative", "groundbreaking", "remarkable", "significant", "emerging", "revolutionary"],
    "tech": ["artificial intelligence", "blockchain", "cloud computing", "machine learning", "IoT"],
    "field": ["healthcare", "finance", "education", "manufacturing", "retail"],
    "topic1": ["climate patterns", "neural activity", "genetic markers", "quantum states", "ecosystem dynamics"],
    "topic2": ["human behavior", "disease progression", "evolution", "material properties", "energy systems"],
    "product": ["smart devices", "electric vehicles", "renewable energy", "biotechnology", "software services"],
    "health_topic": ["nutrition", "exercise", "sleep quality", "mental wellness", "preventive care"],
    "outcome": ["improved longevity", "better outcomes", "reduced risk", "enhanced performance", "positive effects"],
    "env_factor": ["pollution", "deforestation", "temperature rise", "biodiversity loss", "ocean acidification"],
    "env_system": ["marine ecosystems", "forest habitats", "agricultural systems", "urban environments", "wildlife populations"],
}

# Paraphrase patterns for semantic duplicates
PARAPHRASE_PATTERNS = [
    # Synonym replacements
    ("revolutionized", "transformed"),
    ("essential", "crucial"),
    ("investing heavily", "allocating significant resources"),
    ("discovered", "found"),
    ("indicates", "suggests"),
    ("significant", "substantial"),
    ("promising", "encouraging"),
    # Structural changes
    ("The study of", "Studying"),
    ("Scientists are exploring", "Research is underway on"),
    ("Experts predict", "It is predicted that"),
    ("New research suggests", "Recent studies indicate"),
]


def _generate_sentence(label: int, seed: int, add_noise: bool = True) -> str:
    """Generate a sentence for a given label using templates.

    Args:
        label: Class label
        seed: Random seed
        add_noise: If True, sometimes use templates from other classes (label noise)
    """
    rng = random.Random(seed)

    # With some probability, use a template from a different class (label noise)
    # This makes classification harder and more realistic
    effective_label = label
    if add_noise and rng.random() < 0.15:  # 15% label noise
        effective_label = rng.randint(0, len(SENTENCE_TEMPLATES) - 1)

    templates = SENTENCE_TEMPLATES.get(effective_label, SENTENCE_TEMPLATES[0])
    template = rng.choice(templates)

    # Fill in placeholders
    result = template
    for key, options in TEMPLATE_FILLERS.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, rng.choice(options), 1)

    # Add random filler words to increase overlap between classes
    if add_noise and rng.random() < 0.3:  # 30% chance
        fillers = ["Additionally,", "Furthermore,", "However,", "Indeed,", "Moreover,"]
        result = rng.choice(fillers) + " " + result.lower()

    return result


def _create_paraphrase(text: str, seed: int) -> str:
    """Create a paraphrased version of the text."""
    rng = random.Random(seed)
    result = text

    # Apply some random paraphrase patterns
    patterns = list(PARAPHRASE_PATTERNS)
    rng.shuffle(patterns)

    for old, new in patterns[:2]:  # Apply up to 2 patterns
        if old in result:
            result = result.replace(old, new)

    # Sometimes add/remove minor words
    if rng.random() < 0.3:
        result = result.replace(" is ", " has become ")
    if rng.random() < 0.3:
        result = result.replace(" the ", " this ")

    return result


def _hash_text(text: str) -> str:
    """Create a short hash for duplicate group identification."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def generate_classification_dataset(
    size: int = 10000,
    num_classes: int = 5,
    duplicate_rate: float = 0.3,
    paraphrase_rate: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate a balanced classification dataset with duplicates.

    Args:
        size: Total number of samples (including duplicates)
        num_classes: Number of classes
        duplicate_rate: Fraction of samples that are exact duplicates
        paraphrase_rate: Fraction of samples that are paraphrase duplicates
        seed: Random seed for reproducibility

    Returns:
        SyntheticDataset with ground truth annotations
    """
    rng = random.Random(seed)
    samples: List[SyntheticSample] = []

    # Calculate how many of each type
    num_exact_dups = int(size * duplicate_rate)
    num_paraphrases = int(size * paraphrase_rate)
    num_unique = size - num_exact_dups - num_paraphrases

    # Generate unique samples
    unique_samples: List[SyntheticSample] = []
    for i in range(num_unique):
        label = i % num_classes
        text = _generate_sentence(label, seed=seed + i)
        group_id = _hash_text(text)

        unique_samples.append(SyntheticSample(
            text=text,
            label=label,
            is_duplicate=False,
            is_paraphrase=False,
            duplicate_group_id=group_id,
        ))

    samples.extend(unique_samples)

    # Generate exact duplicates
    for i in range(num_exact_dups):
        original = rng.choice(unique_samples)
        samples.append(SyntheticSample(
            text=original.text,  # Exact copy
            label=original.label,
            original_idx=unique_samples.index(original),
            is_duplicate=True,
            is_paraphrase=False,
            duplicate_group_id=original.duplicate_group_id,
        ))

    # Generate paraphrase duplicates
    for i in range(num_paraphrases):
        original = rng.choice(unique_samples)
        paraphrased_text = _create_paraphrase(original.text, seed=seed + num_unique + i)
        samples.append(SyntheticSample(
            text=paraphrased_text,
            label=original.label,
            original_idx=unique_samples.index(original),
            is_duplicate=True,
            is_paraphrase=True,
            duplicate_group_id=original.duplicate_group_id,
        ))

    # Shuffle samples
    rng.shuffle(samples)

    return SyntheticDataset(
        samples=samples,
        name="synth_classification",
        metadata={
            "size": size,
            "num_classes": num_classes,
            "duplicate_rate": duplicate_rate,
            "paraphrase_rate": paraphrase_rate,
            "seed": seed,
            "num_exact_duplicates": num_exact_dups,
            "num_paraphrases": num_paraphrases,
            "num_unique": num_unique,
        },
    )


def generate_imbalanced_dataset(
    size: int = 10000,
    class_proportions: Optional[List[float]] = None,
    duplicate_rate: float = 0.3,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate an imbalanced classification dataset.

    Args:
        size: Total number of samples
        class_proportions: Proportion for each class (default: [0.90, 0.05, 0.03, 0.015, 0.005])
        duplicate_rate: Fraction of samples that are duplicates
        seed: Random seed

    Returns:
        SyntheticDataset with imbalanced class distribution
    """
    if class_proportions is None:
        class_proportions = [0.90, 0.05, 0.03, 0.015, 0.005]

    # Normalize proportions
    total = sum(class_proportions)
    class_proportions = [p / total for p in class_proportions]

    rng = random.Random(seed)
    samples: List[SyntheticSample] = []
    unique_samples: List[SyntheticSample] = []

    num_dups = int(size * duplicate_rate)
    num_unique = size - num_dups

    # Generate unique samples with imbalanced distribution
    for i in range(num_unique):
        # Sample class according to proportions
        r = rng.random()
        cumsum = 0
        label = 0
        for cls_idx, prop in enumerate(class_proportions):
            cumsum += prop
            if r <= cumsum:
                label = cls_idx
                break

        text = _generate_sentence(label, seed=seed + i)
        group_id = _hash_text(text)

        unique_samples.append(SyntheticSample(
            text=text,
            label=label,
            is_duplicate=False,
            duplicate_group_id=group_id,
        ))

    samples.extend(unique_samples)

    # Generate duplicates (respecting class distribution of originals)
    for i in range(num_dups):
        original = rng.choice(unique_samples)
        samples.append(SyntheticSample(
            text=original.text,
            label=original.label,
            original_idx=unique_samples.index(original),
            is_duplicate=True,
            duplicate_group_id=original.duplicate_group_id,
        ))

    rng.shuffle(samples)

    return SyntheticDataset(
        samples=samples,
        name="synth_imbalanced",
        metadata={
            "size": size,
            "class_proportions": class_proportions,
            "duplicate_rate": duplicate_rate,
            "seed": seed,
        },
    )


def generate_contaminated_dataset(
    train_size: int = 8000,
    test_size: int = 2000,
    contamination_rate: float = 0.05,
    num_classes: int = 3,
    seed: int = 42,
) -> Tuple[SyntheticDataset, SyntheticDataset]:
    """Generate train/test split with controlled contamination.

    Args:
        train_size: Number of training samples
        test_size: Number of test samples
        contamination_rate: Fraction of test samples that appear in train
        num_classes: Number of classes
        seed: Random seed

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    rng = random.Random(seed)

    # Generate test set first (all unique)
    # Add unique suffix to ensure no natural duplicates between train/test
    test_samples: List[SyntheticSample] = []
    for i in range(test_size):
        label = i % num_classes
        base_text = _generate_sentence(label, seed=seed + i, add_noise=False)
        # Add unique identifier to ensure test texts are distinct from train
        text = f"{base_text} [ref:{seed}:test:{i}]"
        group_id = _hash_text(text)

        test_samples.append(SyntheticSample(
            text=text,
            label=label,
            is_duplicate=False,
            duplicate_group_id=group_id,
        ))

    # Calculate contamination
    num_contaminated = int(test_size * contamination_rate)
    contaminated_test_indices = rng.sample(range(test_size), num_contaminated)

    # Generate training set
    train_samples: List[SyntheticSample] = []

    # First, add clean training samples with unique identifiers
    clean_train_size = train_size - num_contaminated
    for i in range(clean_train_size):
        label = i % num_classes
        base_text = _generate_sentence(label, seed=seed + test_size + i, add_noise=False)
        # Add unique identifier to ensure train texts are distinct from test
        text = f"{base_text} [ref:{seed}:train:{i}]"
        group_id = _hash_text(text)

        train_samples.append(SyntheticSample(
            text=text,
            label=label,
            is_duplicate=False,
            duplicate_group_id=group_id,
        ))

    # Add contaminated samples (copies from test set)
    for test_idx in contaminated_test_indices:
        original = test_samples[test_idx]
        train_samples.append(SyntheticSample(
            text=original.text,
            label=original.label,
            original_idx=test_idx,
            is_duplicate=True,  # Marked as duplicate (of test)
            duplicate_group_id=original.duplicate_group_id,
        ))

    rng.shuffle(train_samples)

    train_dataset = SyntheticDataset(
        samples=train_samples,
        name="synth_contaminated_train",
        metadata={
            "size": train_size,
            "num_classes": num_classes,
            "contaminated_indices": contaminated_test_indices,
            "contamination_rate": contamination_rate,
            "seed": seed,
        },
    )

    test_dataset = SyntheticDataset(
        samples=test_samples,
        name="synth_contaminated_test",
        metadata={
            "size": test_size,
            "num_classes": num_classes,
            "seed": seed,
        },
    )

    return train_dataset, test_dataset


def generate_paraphrase_dataset(
    size: int = 5000,
    num_classes: int = 3,
    paraphrase_rate: float = 0.4,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate dataset with high paraphrase rate for semantic dedup evaluation.

    Args:
        size: Total number of samples
        num_classes: Number of classes
        paraphrase_rate: Fraction of samples that are paraphrases
        seed: Random seed

    Returns:
        SyntheticDataset focused on paraphrase duplicates
    """
    rng = random.Random(seed)
    samples: List[SyntheticSample] = []

    num_paraphrases = int(size * paraphrase_rate)
    num_unique = size - num_paraphrases

    # Generate unique samples
    unique_samples: List[SyntheticSample] = []
    for i in range(num_unique):
        label = i % num_classes
        text = _generate_sentence(label, seed=seed + i)
        group_id = _hash_text(text)

        unique_samples.append(SyntheticSample(
            text=text,
            label=label,
            is_duplicate=False,
            is_paraphrase=False,
            duplicate_group_id=group_id,
        ))

    samples.extend(unique_samples)

    # Generate paraphrase duplicates
    for i in range(num_paraphrases):
        original = rng.choice(unique_samples)
        paraphrased_text = _create_paraphrase(original.text, seed=seed + num_unique + i)

        samples.append(SyntheticSample(
            text=paraphrased_text,
            label=original.label,
            original_idx=unique_samples.index(original),
            is_duplicate=True,
            is_paraphrase=True,
            duplicate_group_id=original.duplicate_group_id,
        ))

    rng.shuffle(samples)

    return SyntheticDataset(
        samples=samples,
        name="synth_paraphrases",
        metadata={
            "size": size,
            "num_classes": num_classes,
            "paraphrase_rate": paraphrase_rate,
            "seed": seed,
            "num_paraphrases": num_paraphrases,
            "num_unique": num_unique,
        },
    )


# ==============================================================================
# REAL DATASETS (HuggingFace)
# ==============================================================================

@dataclass
class RealDataset:
    """A real dataset loaded from HuggingFace.

    Compatible with SyntheticDataset interface but without ground truth
    duplicate annotations (since we don't know true duplicates in real data).
    """

    texts: List[str]
    labels: List[int]
    name: str
    split: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For paraphrase datasets, store pairs
    text_pairs: Optional[List[Tuple[str, str]]] = None
    pair_labels: Optional[List[int]] = None  # 1 = paraphrase/duplicate, 0 = not

    @property
    def samples(self) -> List[Dict[str, Any]]:
        """Get samples as list of dicts for compatibility."""
        return [{"text": t, "label": l} for t, l in zip(self.texts, self.labels)]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries."""
        return self.samples


def load_mrpc(
    split: str = "train",
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load Microsoft Research Paraphrase Corpus (MRPC).

    MRPC contains sentence pairs labeled as paraphrase (1) or not (0).
    Good for evaluating semantic deduplication.

    Args:
        split: "train", "validation", or "test"
        cache_dir: Cache directory for dataset

    Returns:
        RealDataset with paraphrase pairs
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Install: pip install datasets")

    dataset = hf_load_dataset("glue", "mrpc", split=split, cache_dir=cache_dir)

    texts = []
    labels = []
    text_pairs = []
    pair_labels = []

    for item in dataset:
        # Use sentence1 as primary text for dedup benchmarks
        texts.append(item["sentence1"])
        labels.append(item["label"])

        # Store pairs for paraphrase detection benchmarks
        text_pairs.append((item["sentence1"], item["sentence2"]))
        pair_labels.append(item["label"])

    return RealDataset(
        texts=texts,
        labels=labels,
        name="mrpc",
        split=split,
        text_pairs=text_pairs,
        pair_labels=pair_labels,
        metadata={
            "source": "glue/mrpc",
            "size": len(texts),
            "num_paraphrases": sum(pair_labels),
            "paraphrase_rate": sum(pair_labels) / len(pair_labels) if pair_labels else 0,
            "description": "Microsoft Research Paraphrase Corpus - sentence pairs labeled as paraphrases or not",
        },
    )


def load_qqp(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load Quora Question Pairs (QQP).

    QQP contains question pairs labeled as duplicate (1) or not (0).
    Large dataset with real-world duplicates.

    Args:
        split: "train" or "validation"
        max_samples: Maximum samples to load (QQP is large)
        cache_dir: Cache directory

    Returns:
        RealDataset with question pairs
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Install: pip install datasets")

    dataset = hf_load_dataset("glue", "qqp", split=split, cache_dir=cache_dir)

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    texts = []
    labels = []
    text_pairs = []
    pair_labels = []

    for item in dataset:
        texts.append(item["question1"])
        labels.append(item["label"])
        text_pairs.append((item["question1"], item["question2"]))
        pair_labels.append(item["label"])

    return RealDataset(
        texts=texts,
        labels=labels,
        name="qqp",
        split=split,
        text_pairs=text_pairs,
        pair_labels=pair_labels,
        metadata={
            "source": "glue/qqp",
            "size": len(texts),
            "num_duplicates": sum(pair_labels),
            "duplicate_rate": sum(pair_labels) / len(pair_labels) if pair_labels else 0,
            "description": "Quora Question Pairs - question pairs labeled as duplicates or not",
        },
    )


def load_paws(
    split: str = "train",
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load PAWS (Paraphrase Adversaries from Word Scrambling).

    PAWS contains adversarial sentence pairs that are lexically similar
    but may have different meanings. Good stress test for semantic dedup.

    Args:
        split: "train", "validation", or "test"
        cache_dir: Cache directory

    Returns:
        RealDataset with adversarial pairs
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Install: pip install datasets")

    dataset = hf_load_dataset("paws", "labeled_final", split=split, cache_dir=cache_dir)

    texts = []
    labels = []
    text_pairs = []
    pair_labels = []

    for item in dataset:
        texts.append(item["sentence1"])
        labels.append(item["label"])
        text_pairs.append((item["sentence1"], item["sentence2"]))
        pair_labels.append(item["label"])

    return RealDataset(
        texts=texts,
        labels=labels,
        name="paws",
        split=split,
        text_pairs=text_pairs,
        pair_labels=pair_labels,
        metadata={
            "source": "paws/labeled_final",
            "size": len(texts),
            "num_paraphrases": sum(pair_labels),
            "description": "PAWS - adversarial paraphrase pairs (high lexical overlap, may differ semantically)",
        },
    )


def load_stsb(
    split: str = "train",
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load STS-B (Semantic Textual Similarity Benchmark).

    STS-B contains sentence pairs with similarity scores from 0 to 5.
    Useful for threshold tuning experiments.

    Args:
        split: "train", "validation", or "test"
        cache_dir: Cache directory

    Returns:
        RealDataset with similarity scores
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Install: pip install datasets")

    dataset = hf_load_dataset("glue", "stsb", split=split, cache_dir=cache_dir)

    texts = []
    labels = []  # Discretized similarity
    text_pairs = []
    similarity_scores = []

    for item in dataset:
        texts.append(item["sentence1"])
        # Discretize similarity: 0-2.5 = 0 (not similar), 2.5-5 = 1 (similar)
        labels.append(1 if item["label"] >= 2.5 else 0)
        text_pairs.append((item["sentence1"], item["sentence2"]))
        similarity_scores.append(item["label"])

    return RealDataset(
        texts=texts,
        labels=labels,
        name="stsb",
        split=split,
        text_pairs=text_pairs,
        pair_labels=[1 if s >= 2.5 else 0 for s in similarity_scores],
        metadata={
            "source": "glue/stsb",
            "size": len(texts),
            "similarity_scores": similarity_scores,
            "description": "STS-B - sentence pairs with graded similarity (0-5)",
        },
    )


def load_sst2(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load SST-2 (Stanford Sentiment Treebank).

    SST-2 is a binary sentiment classification dataset.
    Good baseline for classification benchmarks.

    Args:
        split: "train", "validation", or "test"
        max_samples: Maximum samples to load
        cache_dir: Cache directory

    Returns:
        RealDataset for sentiment classification
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Install: pip install datasets")

    dataset = hf_load_dataset("glue", "sst2", split=split, cache_dir=cache_dir)

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    texts = [item["sentence"] for item in dataset]
    labels = [item["label"] for item in dataset]

    return RealDataset(
        texts=texts,
        labels=labels,
        name="sst2",
        split=split,
        metadata={
            "source": "glue/sst2",
            "size": len(texts),
            "num_positive": sum(labels),
            "num_negative": len(labels) - sum(labels),
            "description": "SST-2 - binary sentiment classification",
        },
    )


def load_ag_news(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load AG News topic classification dataset.

    AG News has 4 classes: World, Sports, Business, Sci/Tech.
    Good for multi-class classification benchmarks.

    Args:
        split: "train" or "test"
        max_samples: Maximum samples to load
        cache_dir: Cache directory

    Returns:
        RealDataset for topic classification
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Install: pip install datasets")

    dataset = hf_load_dataset("ag_news", split=split, cache_dir=cache_dir)

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    texts = [item["text"] for item in dataset]
    labels = [item["label"] for item in dataset]

    return RealDataset(
        texts=texts,
        labels=labels,
        name="ag_news",
        split=split,
        metadata={
            "source": "ag_news",
            "size": len(texts),
            "num_classes": 4,
            "class_names": ["World", "Sports", "Business", "Sci/Tech"],
            "description": "AG News - 4-class topic classification",
        },
    )


def load_imdb(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load IMDB movie review sentiment dataset.

    IMDB has longer texts than SST-2, good for document-level dedup.

    Args:
        split: "train" or "test"
        max_samples: Maximum samples to load
        cache_dir: Cache directory

    Returns:
        RealDataset for sentiment classification
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets required. Install: pip install datasets")

    dataset = hf_load_dataset("imdb", split=split, cache_dir=cache_dir)

    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    texts = [item["text"] for item in dataset]
    labels = [item["label"] for item in dataset]

    return RealDataset(
        texts=texts,
        labels=labels,
        name="imdb",
        split=split,
        metadata={
            "source": "imdb",
            "size": len(texts),
            "avg_length": sum(len(t.split()) for t in texts) / len(texts),
            "description": "IMDB - long-form movie review sentiment (good for document dedup)",
        },
    )


# Real dataset loaders registry
REAL_DATASET_LOADERS = {
    "mrpc": load_mrpc,
    "qqp": load_qqp,
    "paws": load_paws,
    "stsb": load_stsb,
    "sst2": load_sst2,
    "ag_news": load_ag_news,
    "imdb": load_imdb,
}

# Dataset info for documentation
REAL_DATASET_INFO = {
    "mrpc": {
        "size": "~5.8k",
        "task": "Paraphrase detection",
        "use_case": "Semantic dedup evaluation - known paraphrase pairs",
    },
    "qqp": {
        "size": "~400k",
        "task": "Duplicate question detection",
        "use_case": "Large-scale real-world duplicates",
    },
    "paws": {
        "size": "~108k",
        "task": "Adversarial paraphrase detection",
        "use_case": "Stress test - hard negatives with high lexical overlap",
    },
    "stsb": {
        "size": "~8.6k",
        "task": "Semantic similarity (graded 0-5)",
        "use_case": "Threshold tuning with graded similarity",
    },
    "sst2": {
        "size": "~70k",
        "task": "Sentiment classification",
        "use_case": "Classification baseline",
    },
    "ag_news": {
        "size": "~120k",
        "task": "Topic classification (4 classes)",
        "use_case": "Multi-class classification with potential duplicates",
    },
    "imdb": {
        "size": "~50k",
        "task": "Sentiment classification (long text)",
        "use_case": "Document-level deduplication",
    },
}


def load_real_dataset(
    name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> RealDataset:
    """Load a real dataset by name.

    Args:
        name: Dataset name (mrpc, qqp, paws, stsb, sst2, ag_news, imdb)
        split: Data split
        max_samples: Maximum samples (for large datasets)
        cache_dir: Cache directory

    Returns:
        RealDataset

    Raises:
        KeyError: If dataset not found
        ImportError: If HuggingFace datasets not installed
    """
    if name not in REAL_DATASET_LOADERS:
        available = ", ".join(REAL_DATASET_LOADERS.keys())
        raise KeyError(f"Unknown real dataset: {name}. Available: {available}")

    loader = REAL_DATASET_LOADERS[name]

    # Handle max_samples for loaders that support it
    if name in ["qqp", "sst2", "ag_news", "imdb"] and max_samples:
        return loader(split=split, max_samples=max_samples, cache_dir=cache_dir)
    else:
        return loader(split=split, cache_dir=cache_dir)


def list_real_datasets() -> List[str]:
    """List available real datasets."""
    return list(REAL_DATASET_LOADERS.keys())


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Get information about a dataset."""
    if name in REAL_DATASET_INFO:
        return REAL_DATASET_INFO[name]
    elif name.startswith("synth"):
        return {"size": "configurable", "task": "synthetic", "use_case": "Controlled testing"}
    else:
        return {"size": "unknown", "task": "unknown", "use_case": "unknown"}


# ==============================================================================
# SYNTHETIC DATASETS
# ==============================================================================

# Dataset registry for easy access
SYNTHETIC_DATASETS = {
    "synth_classification": lambda seed=42: generate_classification_dataset(
        size=10000, num_classes=5, duplicate_rate=0.3, seed=seed
    ),
    "synth_small": lambda seed=42: generate_classification_dataset(
        size=1000, num_classes=5, duplicate_rate=0.2, seed=seed
    ),
    "synth_large": lambda seed=42: generate_classification_dataset(
        size=100000, num_classes=10, duplicate_rate=0.25, seed=seed
    ),
    "synth_imbalanced": lambda seed=42: generate_imbalanced_dataset(
        size=10000, duplicate_rate=0.3, seed=seed
    ),
    "synth_paraphrases": lambda seed=42: generate_paraphrase_dataset(
        size=5000, num_classes=3, paraphrase_rate=0.4, seed=seed
    ),
    "synth_high_dup": lambda seed=42: generate_classification_dataset(
        size=10000, num_classes=5, duplicate_rate=0.8, seed=seed
    ),
    "synth_low_dup": lambda seed=42: generate_classification_dataset(
        size=10000, num_classes=5, duplicate_rate=0.05, seed=seed
    ),
}


def get_dataset(
    name: str,
    seed: int = 42,
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Union[SyntheticDataset, RealDataset]:
    """Get a dataset by name (synthetic or real).

    Args:
        name: Dataset name (e.g., "synth_classification", "mrpc", "qqp")
        seed: Random seed (for synthetic datasets)
        split: Data split (for real datasets)
        max_samples: Maximum samples (for large real datasets)
        cache_dir: Cache directory (for real datasets)

    Returns:
        SyntheticDataset or RealDataset

    Raises:
        KeyError: If dataset name is not found
    """
    # Check synthetic datasets first
    if name in SYNTHETIC_DATASETS:
        return SYNTHETIC_DATASETS[name](seed=seed)

    # Check real datasets
    if name in REAL_DATASET_LOADERS:
        return load_real_dataset(
            name,
            split=split,
            max_samples=max_samples,
            cache_dir=cache_dir,
        )

    # Not found
    all_available = list(SYNTHETIC_DATASETS.keys()) + list(REAL_DATASET_LOADERS.keys())
    raise KeyError(f"Unknown dataset: {name}. Available: {', '.join(all_available)}")


def get_contaminated_dataset(seed: int = 42) -> Tuple[SyntheticDataset, SyntheticDataset]:
    """Get the contaminated train/test dataset pair.

    Args:
        seed: Random seed

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    return generate_contaminated_dataset(
        train_size=8000,
        test_size=2000,
        contamination_rate=0.05,
        num_classes=3,
        seed=seed,
    )


def list_datasets(include_real: bool = True) -> List[str]:
    """List available dataset names.

    Args:
        include_real: If True, include real datasets (requires HuggingFace)

    Returns:
        List of dataset names
    """
    names = list(SYNTHETIC_DATASETS.keys())
    if include_real:
        names.extend(list(REAL_DATASET_LOADERS.keys()))
    return names


def list_all_datasets_with_info() -> Dict[str, Dict[str, Any]]:
    """List all datasets with metadata.

    Returns:
        Dict mapping dataset name to info dict with:
        - type: "synthetic" or "real"
        - size: Dataset size description
        - description: What the dataset is for
        - available: Whether the dataset can be loaded
    """
    result = {}

    # Synthetic datasets
    for name in SYNTHETIC_DATASETS:
        info = get_dataset_info(name)
        result[name] = {
            "type": "synthetic",
            "size": info.get("size", "configurable"),
            "description": info.get("use_case", "Controlled testing with ground truth"),
            "available": True,
        }

    # Real datasets
    for name in REAL_DATASET_LOADERS:
        info = REAL_DATASET_INFO.get(name, {})
        result[name] = {
            "type": "real",
            "size": info.get("size", "unknown"),
            "task": info.get("task", "unknown"),
            "description": info.get("use_case", "Real-world data"),
            "available": HF_AVAILABLE,
        }

    return result
