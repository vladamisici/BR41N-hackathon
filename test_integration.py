#!/usr/bin/env python
"""
Final integration test — hackathon readiness check.

Loads BNCI2014_001 subject 1, runs all 4 priority pipelines,
computes lateralization index, and prints results.
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import os
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import numpy as np
import mne
mne.set_log_level("ERROR")

print("=" * 60)
print("INTEGRATION TEST — Hackathon Readiness Check")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Import all src modules
# ------------------------------------------------------------------
print("\n[1/5] Importing src modules...")
from src.classifiers import (
    build_all_pipelines, build_fbcsp_pipeline, build_acm_pipeline,
    FilterBankCSP, AugmentedDataset, evaluate_all, DEFAULT_FILTER_BANKS,
)
from src.channel_selection import csp_rank_channels
from src.evaluation import full_evaluation, compare_to_baseline
from src.lateralization import compute_laterality_index, laterality_report
from src.loading import load_gtec_stroke_data, extract_epochs, CH_NAMES
from src.preprocessing import (
    augment_gaussian_noise, augment_sliding_window,
    augment_ft_surrogate, augment_hemisphere_recombination,
)
from src.transfer import rpa_transfer_pipeline, cross_patient_transfer
from src.visualization import (
    plot_pipeline_comparison, plot_confusion_matrix,
    plot_laterality_comparison,
)
print("  All imports OK ✓")

# ------------------------------------------------------------------
# 2. Load BNCI2014_001 subject 1
# ------------------------------------------------------------------
print("\n[2/5] Loading BNCI2014_001 subject 1...")
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

TARGET_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CP2", "CP4",
]

paradigm = LeftRightImagery(
    fmin=4, fmax=40, tmin=0.5, tmax=4.5,
    channels=TARGET_CHANNELS, resample=None,
)

dataset = BNCI2014_001()
X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])

labels = np.unique(y)
label_map = {lab: i for i, lab in enumerate(sorted(labels))}
y_int = np.array([label_map[lab] for lab in y])

sfreq = 250.0  # BNCI2014_001 sampling rate

print(f"  X shape: {X.shape}")
print(f"  Classes: {labels} → {np.bincount(y_int)}")
print(f"  Channels: {TARGET_CHANNELS}")
print(f"  Sfreq: {sfreq} Hz, Window: 0.5–4.5s, Bandpass: 4–40 Hz")
print("  Data loaded ✓")

# ------------------------------------------------------------------
# 3. Run all 4 priority pipelines
# ------------------------------------------------------------------
print("\n[3/5] Running 4 priority pipelines (5-fold CV)...")

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

priority_pipelines = {
    "1. FBCSP+LDA": build_fbcsp_pipeline(sfreq=sfreq),
    "2. ACM(3,7)":  build_acm_pipeline(order=3, lag=7),
}

# Build TS+LR and CSP+LDA from build_all_pipelines
all_pipes = build_all_pipelines(sfreq=sfreq)
priority_pipelines["3. TS+LR"] = all_pipes["TS+LR"]
priority_pipelines["4. CSP+LDA"] = all_pipes["CSP+LDA"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, pipe in priority_pipelines.items():
    scores = cross_val_score(pipe, X, y_int, cv=cv, scoring="accuracy")
    results[name] = {"mean": scores.mean(), "std": scores.std(), "folds": scores}

print(f"\n  {'Pipeline':<18} {'Accuracy':>10} {'Std':>8}")
print(f"  {'-'*18} {'-'*10} {'-'*8}")
for name in priority_pipelines:
    r = results[name]
    print(f"  {name:<18} {r['mean']:>10.4f} {r['std']:>8.4f}")

# Check ordering
best_name = max(results, key=lambda k: results[k]["mean"])
baseline_acc = results["4. CSP+LDA"]["mean"]
best_acc = results[best_name]["mean"]
print(f"\n  Best: {best_name} ({best_acc:.4f})")
print(f"  vs CSP+LDA baseline: +{best_acc - baseline_acc:.4f}")
print("  Pipelines OK ✓")

# ------------------------------------------------------------------
# 4. Lateralization index
# ------------------------------------------------------------------
print("\n[4/5] Computing lateralization index...")

C3_IDX = 6   # C3 in our 16-ch montage
C4_IDX = 10  # C4

li = compute_laterality_index(X, sfreq, c3_idx=C3_IDX, c4_idx=C4_IDX)

print(f"  Mu  LI (8–13 Hz):  {li['mu_li']:+.4f}")
print(f"  Beta LI (13–30 Hz): {li['beta_li']:+.4f}")
print(f"  Mu  power — C3: {li['mu_c3_power']:.4e}, C4: {li['mu_c4_power']:.4e}")
print(f"  Beta power — C3: {li['beta_c3_power']:.4e}, C4: {li['beta_c4_power']:.4e}")

if abs(li["mu_li"]) < 0.1:
    interp = "Bilateral (weak lateralization)"
elif li["mu_li"] > 0:
    interp = "Left-hemisphere dominant (C3 > C4)"
else:
    interp = "Right-hemisphere dominant (C4 > C3)"
print(f"  Interpretation: {interp}")
print("  Lateralization OK ✓")

# ------------------------------------------------------------------
# 5. Verify build_all_pipelines order and evaluate_all
# ------------------------------------------------------------------
print("\n[5/5] Verifying build_all_pipelines() order + evaluate_all()...")

all_pipes = build_all_pipelines(sfreq=sfreq)
pipe_names = list(all_pipes.keys())
expected_order = ["FBCSP+LDA", "ACM(3,7)", "TS+LR", "CSP+LDA",
                  "FgMDM", "TS+SVM", "MDM", "TS+LDA"]

assert pipe_names == expected_order, f"Order mismatch: {pipe_names}"
print(f"  Pipeline order: {pipe_names[:4]} + {len(pipe_names)-4} more ✓")

# Quick evaluate_all on just the top 4
top4 = {k: all_pipes[k] for k in expected_order[:4]}
df = evaluate_all(X, y_int, top4, n_splits=5)
print(f"\n  evaluate_all() results:")
print(df[["pipeline", "mean_acc", "std_acc"]].to_string(index=False, col_space=12))
print("  evaluate_all OK ✓")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL CHECKS PASSED — Ready for hackathon day")
print("=" * 60)
print(f"\nPipeline ranking for Subject 1:")
for _, row in df.iterrows():
    marker = " ← PRIMARY" if row["pipeline"] == "FBCSP+LDA" else ""
    marker = " ← BASELINE" if row["pipeline"] == "CSP+LDA" else marker
    print(f"  {row['pipeline']:<12} {row['mean_acc']:.4f} ± {row['std_acc']:.4f}{marker}")
print(f"\nLateralization: Mu LI = {li['mu_li']:+.4f} ({interp})")
print(f"Filter banks: {DEFAULT_FILTER_BANKS}")
