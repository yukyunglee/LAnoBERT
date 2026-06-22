"""Evaluation metrics for anomaly detection.

`abnormal_score` convention: higher = more anomalous.
`labels`: 1 = anomaly, 0 = normal.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np


def evaluate(
    scores: Sequence[float],
    labels: Sequence[int],
    result_dir: Optional[str] = None,
    tag: str = "eval",
) -> Dict[str, float]:
    """Compute AUROC and the best-threshold F1 / classification report.

    Returns a dict with auroc, best_f1, best_threshold, precision, recall.
    Optionally saves an ROC curve png and a text report to `result_dir`.
    """
    from sklearn.metrics import (
        auc,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = float(auc(fpr, tpr))

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # f1 per threshold; thresholds has len = len(precision) - 1
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    best_f1 = float(f1[best_idx])

    preds = (scores >= best_threshold).astype(int)
    report = classification_report(labels, preds, digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds)

    out = {
        "auroc": auroc,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }

    print(f"[eval:{tag}] AUROC={auroc:.4f}  best_F1={best_f1:.4f}  thr={best_threshold:.4g}")
    print(report)

    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"{tag}_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"AUROC: {auroc:.6f}\n")
            f.write(f"best_F1: {best_f1:.6f}\n")
            f.write(f"best_threshold: {best_threshold:.6g}\n\n")
            f.write("confusion_matrix:\n")
            f.write(np.array2string(cm) + "\n\n")
            f.write(report + "\n")
        _save_roc(fpr, tpr, auroc, os.path.join(result_dir, f"{tag}_roc.png"))

    return out


def _save_roc(fpr, tpr, auroc, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", label="random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:  # plotting is best-effort
        print(f"[eval] could not save ROC plot: {e}")
