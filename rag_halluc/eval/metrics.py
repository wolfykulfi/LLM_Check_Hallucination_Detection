"""Metrics for unsupervised detection scores.

Convention used everywhere in this project: label 1 = hallucinated.

An unsupervised score (perplexity, attention log-det, ...) has no built-in
direction: for some scores *higher* means "more likely hallucinated", for
others *lower* does. We therefore report:

  auroc      AUROC of the raw score with label 1 = hallucinated
             (0.5 = chance; below 0.5 means the score works "upside down")
  direction  "+" if higher score -> hallucinated, "-" otherwise
  auroc_best max(auroc, 1 - auroc), i.e. AUROC after choosing the direction

Choosing the direction on the same data you evaluate on is a small optimistic
bias. It is fine for Phase 0 (reproduction and sanity checks); from Phase 2
on, directions are fixed on the validation split and only then applied to test.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    """True-positive rate at the largest threshold whose FPR stays below max_fpr
    (same definition as LLM-Check's get_roc_scores)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = np.where(fpr < max_fpr)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def balanced_accuracy_best(labels: np.ndarray, scores: np.ndarray) -> float:
    """Best balanced accuracy over all thresholds: max_t (TPR + TNR) / 2.
    LLM-Check reports this as "Acc"."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(1 - (fpr + (1 - tpr)) / 2))


def score_report(labels, scores) -> Dict[str, float]:
    """All Phase 0 metrics for one score column."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    keep = np.isfinite(scores)
    labels, scores = labels[keep], scores[keep]
    if len(np.unique(labels)) < 2:
        return {"n": int(len(labels)), "auroc": float("nan"), "direction": "", "auroc_best": float("nan"),
                "bal_acc": float("nan"), "tpr_at_5fpr": float("nan")}
    auroc = roc_auc_score(labels, scores)
    sign = 1.0 if auroc >= 0.5 else -1.0
    return {
        "n": int(len(labels)),
        "auroc": float(auroc),
        "direction": "+" if sign > 0 else "-",
        "auroc_best": float(max(auroc, 1 - auroc)),
        "bal_acc": balanced_accuracy_best(labels, sign * scores),
        "tpr_at_5fpr": tpr_at_fpr(labels, sign * scores),
    }
