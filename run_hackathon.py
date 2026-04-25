#!/usr/bin/env python
"""
BR41N.IO Hackathon — Full analysis on real stroke data.
Run this on the EC2 instance after uploading the dataset.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")

from src.loading import load_train_test, C3_IDX, C4_IDX
from src.classifiers import build_all_pipelines, evaluate_all, evaluate_train_test
from src.lateralization import compute_laterality_index

# ── Configuration ────────────────────────────────────────────────
DATA_DIR = "dataset/stroke-rehab/"
SFREQ = 256.0  # confirmed from .mat files

print("=" * 70)
print("BR41N.IO HACKATHON — Stroke Rehab Data Analysis")
print("=" * 70)

# ── 1. Load all data ────────────────────────────────────────────
print("\n[1/4] Loading data...")
data = load_train_test(DATA_DIR)
for patient, splits in sorted(data.items()):
    for key, (X, y, _) in sorted(splits.items()):
        print(f"  {patient}/{key}: {X.shape[0]} epochs, left={sum(y==0)}, right={sum(y==1)}")

# ── 2. Build pipelines ─────────────────────────────────────────
print("\n[2/4] Building pipelines (sfreq=256 Hz)...")
pipelines = build_all_pipelines(sfreq=SFREQ)
print(f"  Pipelines: {list(pipelines.keys())}")

# ── 3. Evaluate: train/test split (hackathon protocol) ─────────
print("\n[3/4] Train/test evaluation (hackathon protocol)...")
print("=" * 70)

all_tt_results = {}
for patient in sorted(data.keys()):
    for stage in ["pre", "post"]:
        train_key = f"{stage}_train"
        test_key = f"{stage}_test"
        if train_key not in data[patient] or test_key not in data[patient]:
            continue

        X_tr, y_tr, _ = data[patient][train_key]
        X_te, y_te, _ = data[patient][test_key]

        label = f"{patient}_{stage}"
        print(f"\n--- {label} (train={len(y_tr)}, test={len(y_te)}) ---")
        df = evaluate_train_test(X_tr, y_tr, X_te, y_te, pipelines)
        all_tt_results[label] = df
        print(df[["pipeline", "accuracy"]].to_string(index=False))

# ── 4. Lateralization index ─────────────────────────────────────
print("\n\n[4/4] Lateralization Index...")
print("=" * 70)

li_results = {}
for patient in sorted(data.keys()):
    for stage in ["pre", "post"]:
        train_key = f"{stage}_train"
        if train_key not in data[patient]:
            continue
        X, y, epochs = data[patient][train_key]
        li = compute_laterality_index(X, SFREQ, c3_idx=C3_IDX, c4_idx=C4_IDX)
        label = f"{patient}_{stage}"
        li_results[label] = li
        pattern = "Bilateral" if abs(li["mu_li"]) < 0.1 else ("L-dom" if li["mu_li"] > 0 else "R-dom")
        print(f"  {label}: Mu LI={li['mu_li']:+.4f}, Beta LI={li['beta_li']:+.4f} ({pattern})")

# ── 5. Summary ──────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Hackathon baselines from slides
baselines = {
    "P1_pre": {"CSP+LDA": 0.771, "PCA+TVLDA": 0.929},
    "P1_post": {"CSP+LDA": 0.939, "PCA+TVLDA": 0.970},
    "P2_pre": {"CSP+LDA": 0.684, "PCA+TVLDA": 0.724},
    "P2_post": {"CSP+LDA": 0.961, "PCA+TVLDA": 0.974},
    "P3_pre": {"CSP+LDA": 0.744, "PCA+TVLDA": 0.936},
    "P3_post": {"CSP+LDA": 0.797, "PCA+TVLDA": 1.000},
}

print(f"\n{'Condition':<12} {'Best Pipeline':<14} {'Acc':>6} {'vs CSP+LDA':>11} {'vs PCA+TVLDA':>13} {'Mu LI':>8}")
print("-" * 70)

for label in sorted(all_tt_results.keys()):
    df = all_tt_results[label]
    best = df.iloc[0]
    li = li_results.get(label, {})
    mu_li = li.get("mu_li", float("nan"))

    bl = baselines.get(label, {})
    bl_csp = bl.get("CSP+LDA", float("nan"))
    bl_pca = bl.get("PCA+TVLDA", float("nan"))

    diff_csp = best["accuracy"] - bl_csp
    diff_pca = best["accuracy"] - bl_pca

    print(f"{label:<12} {best['pipeline']:<14} {best['accuracy']:>5.1%} {diff_csp:>+10.1%} {diff_pca:>+12.1%} {mu_li:>+8.4f}")

print("\n" + "=" * 70)
print("DONE — Copy these results into your presentation slides")
print("=" * 70)
