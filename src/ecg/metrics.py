"""Evaluation metrics: per-class scores, macro-F1 and a bootstrap confidence interval."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .data import AAMI_CLASSES


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(AAMI_CLASSES)))
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
        for i, name in enumerate(AAMI_CLASSES)
    }


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(AAMI_CLASSES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """95% confidence interval for macro-F1 by resampling test beats with replacement."""
    rng = np.random.default_rng(seed)
    labels = list(range(len(AAMI_CLASSES)))
    n, scores = len(y_true), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        scores.append(f1_score(y_true[i], y_pred[i], labels=labels, average="macro", zero_division=0))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
