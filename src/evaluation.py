"""Evaluation utilities with permutation testing and statistical comparison.

Provides comprehensive evaluation metrics: accuracy, Cohen's kappa,
permutation-based p-values, and pairwise significance tests.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score

logger = logging.getLogger(__name__)


def full_evaluation(
    clf: BaseEstimator,
    X: NDArray,
    y: NDArray,
    n_perms: int = 1000,
) -> dict[str, Any]:
    """Comprehensive evaluation with permutation testing.

    Parameters
    ----------
    clf : BaseEstimator
        Sklearn-compatible classifier or pipeline.
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    n_perms : int
        Number of permutations for significance testing.

    Returns
    -------
    dict
        Keys: acc, kappa, p_value, balanced_acc, confusion_matrix,
              classification_report, adjusted_chance_level, fold_scores.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validated accuracy
    fold_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    acc = fold_scores.mean()

    # Permutation test for significance
    logger.info("Running permutation test (%d permutations)...", n_perms)
    score, perm_scores, p_value = permutation_test_score(
        clf, X, y, cv=cv, n_permutations=n_perms, scoring="accuracy",
        random_state=42, n_jobs=-1,
    )

    # Fit on full data for confusion matrix / report
    clf_fitted = clf.fit(X, y)
    y_pred = clf_fitted.predict(X)  # resubstitution (for confusion matrix display)

    # For a fairer confusion matrix, use cross-val predictions
    from sklearn.model_selection import cross_val_predict
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv)

    kappa = cohen_kappa_score(y, y_pred_cv)
    bal_acc = balanced_accuracy_score(y, y_pred_cv)
    cm = confusion_matrix(y, y_pred_cv)
    report = classification_report(y, y_pred_cv)

    # Adjusted chance level (binomial distribution, alpha=0.05)
    n_trials = len(y)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes
    adjusted_chance = scipy_stats.binom.ppf(0.95, n_trials, chance) / n_trials

    # Print formatted results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy:          {acc:.4f} ± {fold_scores.std():.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Cohen's Kappa:     {kappa:.4f}")
    print(f"  Permutation p:     {p_value:.4f}")
    print(f"  Chance level:      {chance:.4f}")
    print(f"  Adjusted chance:   {adjusted_chance:.4f} (alpha=0.05)")
    print(f"  Significant:       {'YES' if p_value < 0.05 else 'NO'}")
    print("-" * 50)
    print("Confusion Matrix (CV):")
    print(cm)
    print("-" * 50)
    print("Classification Report (CV):")
    print(report)
    print("=" * 50)

    return {
        "acc": acc,
        "kappa": kappa,
        "p_value": p_value,
        "balanced_acc": bal_acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "adjusted_chance_level": adjusted_chance,
        "fold_scores": fold_scores.tolist(),
    }


def compare_to_baseline(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add significance column by paired t-test vs CSP+LDA baseline.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_all() with 'pipeline' and 'fold_scores' columns.

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'vs_baseline_p' and 'significant' columns.
    """
    df = results_df.copy()

    baseline_row = df[df["pipeline"] == "CSP+LDA"]
    if baseline_row.empty:
        logger.warning("CSP+LDA not found in results — cannot compare")
        df["vs_baseline_p"] = np.nan
        df["significant"] = False
        return df

    baseline_scores = np.array(baseline_row.iloc[0]["fold_scores"])

    p_values: list[float] = []
    for _, row in df.iterrows():
        if row["pipeline"] == "CSP+LDA":
            p_values.append(1.0)
            continue

        fold_scores = np.array(row["fold_scores"])
        if len(fold_scores) == 0 or len(fold_scores) != len(baseline_scores):
            p_values.append(np.nan)
            continue

        _, p = scipy_stats.ttest_rel(fold_scores, baseline_scores)
        p_values.append(float(p))

    df["vs_baseline_p"] = p_values
    df["significant"] = df["vs_baseline_p"] < 0.05

    return df
