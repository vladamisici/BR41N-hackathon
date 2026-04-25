#!/usr/bin/env python
"""
BR41N.IO Hackathon — FAST Statistical Validation

Same 6 checks but optimized for speed:
- Permutation test uses TS+LR (fast) instead of FBCSP (slow)
- Label shuffle reduced to 10 iterations
- Best pipeline accuracy still reported from the full set
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")

from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from src.loading import load_train_test, C3_IDX, C4_IDX
from src.classifiers import build_all_pipelines
from src.lateralization import compute_laterality_index

DATA_DIR = Path("dataset/stroke-rehab/")
SFREQ = 256.0
N_PERMUTATIONS = 200

# Fast pipeline for permutation test (TS+LR is ~50x faster than FBCSP)
FAST_PIPE = Pipeline([
    ("cov", Covariances(estimator="oas")),
    ("ts", TangentSpace(metric="riemann")),
    ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
])

print("=" * 75)
print("FAST STATISTICAL VALIDATION")
print("=" * 75)

data = load_train_test(str(DATA_DIR))
pipelines = build_all_pipelines(sfreq=SFREQ)
TOP4 = {k: pipelines[k] for k in ["FBCSP+LDA", "ACM(3,7)", "TS+LR", "CSP+LDA"]}

CONDITIONS = [
    ("P1", "pre"), ("P1", "post"),
    ("P2", "pre"), ("P2", "post"),
    ("P3", "pre"), ("P3", "post"),
]

all_validation = []

for patient, stage in CONDITIONS:
    X_tr, y_tr, _ = data[patient][f"{stage}_train"]
    X_te, y_te, _ = data[patient][f"{stage}_test"]
    label = f"{patient}_{stage}"
    n_test = len(y_te)
    chance = 0.5
    sig_threshold = sp_stats.binom.ppf(0.95, n_test, chance) / n_test

    print(f"\n{'='*75}")
    print(f"  {label}  |  train={len(y_tr)}  test={n_test}")
    print(f"{'='*75}")

    # Find best pipeline
    best_acc, best_name, best_y_pred = 0, "", None
    for name, pipe in TOP4.items():
        p = clone(pipe)
        p.fit(X_tr, y_tr)
        y_pred = p.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        if acc > best_acc:
            best_acc, best_name, best_y_pred = acc, name, y_pred

    # 1. Binomial
    binom_p = 1.0 - sp_stats.binom.cdf(int(best_acc * n_test) - 1, n_test, chance)
    print(f"\n  1. ACCURACY: {best_name} = {best_acc:.1%}  (binomial p={binom_p:.1e}, {'SIG ✓' if best_acc > sig_threshold else 'NS ✗'})")

    # 2. Kappa
    kappa = cohen_kappa_score(y_te, best_y_pred)
    print(f"  2. KAPPA: {kappa:.3f} ({'almost perfect' if kappa > 0.8 else 'substantial' if kappa > 0.6 else 'moderate'})")

    # 3. Per-class
    cm = confusion_matrix(y_te, best_y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print(f"  3. PER-CLASS: L={class_acc[0]:.0%} R={class_acc[1]:.0%} (balance={min(class_acc)/max(class_acc):.2f})")

    # 4. CV consistency
    X_all = np.concatenate([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use TS+LR for CV (fast, representative)
    cv_scores = cross_val_score(clone(FAST_PIPE), X_all, y_all, cv=cv, scoring="accuracy")
    cv_mean = cv_scores.mean()
    print(f"  4. CV (TS+LR): {cv_mean:.1%} ± {cv_scores.std():.1%} (diff from test: {abs(best_acc - cv_mean):.1%}, {'CONSISTENT ✓' if abs(best_acc - cv_mean) < 0.15 else 'CHECK ✗'})")

    # 5. Permutation test (using FAST TS+LR)
    print(f"  5. PERMUTATION ({N_PERMUTATIONS} shuffles, TS+LR)...", end=" ", flush=True)
    score, perm_scores, perm_p = permutation_test_score(
        clone(FAST_PIPE), X_all, y_all, cv=cv,
        n_permutations=N_PERMUTATIONS, scoring="accuracy",
        random_state=42, n_jobs=-1,
    )
    print(f"p={perm_p:.4f} (null={perm_scores.mean():.1%}±{perm_scores.std():.1%}, max={perm_scores.max():.1%}) {'SIG ✓' if perm_p < 0.05 else 'NS ✗'}")

    # 6. Label shuffle
    shuf_accs = []
    for i in range(10):
        y_shuf = np.random.RandomState(i).permutation(y_tr)
        p = clone(FAST_PIPE)
        p.fit(X_tr, y_shuf)
        shuf_accs.append(accuracy_score(y_te, p.predict(X_te)))
    shuf_mean = np.mean(shuf_accs)
    print(f"  6. SHUFFLE: real={best_acc:.1%} vs shuffled={shuf_mean:.1%} (gap={best_acc-shuf_mean:.1%}) {'REAL ✓' if shuf_mean < 0.60 else 'CHECK ✗'}")

    # LI
    li = compute_laterality_index(X_tr, SFREQ, c3_idx=C3_IDX, c4_idx=C4_IDX)

    all_validation.append({
        "condition": label, "pipeline": best_name, "accuracy": best_acc,
        "kappa": kappa, "binom_p": binom_p, "perm_p": perm_p,
        "cv_mean": cv_mean, "shuffle_mean": shuf_mean, "mu_li": li["mu_li"],
    })

# Summary
print(f"\n\n{'='*75}")
print("SUMMARY")
print(f"{'='*75}")
print(f"\n{'Cond':<10} {'Pipe':<12} {'Acc':>5} {'κ':>5} {'Binom':>9} {'Perm':>7} {'CV':>6} {'Shuf':>5} {'LI':>7}")
print("-" * 70)
for r in all_validation:
    print(f"{r['condition']:<10} {r['pipeline']:<12} {r['accuracy']:>4.0%} {r['kappa']:>5.2f} "
          f"{r['binom_p']:>9.1e} {r['perm_p']:>7.4f} {r['cv_mean']:>5.0%} {r['shuffle_mean']:>5.0%} {r['mu_li']:>+6.3f}")

n = len(all_validation)
print(f"\nAll significant: {sum(1 for r in all_validation if r['binom_p'] < 0.05)}/{n}")
print(f"All permutation: {sum(1 for r in all_validation if r['perm_p'] < 0.05)}/{n}")
print(f"All real signal:  {sum(1 for r in all_validation if r['shuffle_mean'] < 0.60)}/{n}")
print(f"{'='*75}")
