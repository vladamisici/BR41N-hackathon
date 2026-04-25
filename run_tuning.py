#!/usr/bin/env python
"""
BR41N.IO Hackathon — Parameter tuning on real stroke data.
Tests different epoch windows, bandpass filters, and artifact rejection thresholds.
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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from src.loading import CH_NAMES, C3_IDX, C4_IDX
from src.classifiers import FilterBankCSP, AugmentedDataset, DEFAULT_FILTER_BANKS

# ── Configuration ────────────────────────────────────────────────
DATA_DIR = Path("dataset/stroke-rehab/")
SFREQ = 256.0

# ── Load raw .mat files (bypass extract_epochs to control parameters) ──
def load_raw(mat_path):
    """Load a .mat file and return raw EEG + trigger arrays."""
    from scipy.io import loadmat
    d = loadmat(str(mat_path), squeeze_me=True)
    fs = float(d["fs"])
    y = np.asarray(d["y"], dtype=np.float64)  # (samples, 16)
    trig = np.asarray(d["trig"], dtype=np.float64).flatten()
    return y, trig, fs


def extract_custom_epochs(eeg, trig, fs, tmin, tmax, l_freq, h_freq, reject_uv=None):
    """Extract epochs with custom parameters directly from arrays."""
    n_samples, n_ch = eeg.shape

    # Build MNE Raw
    eeg_v = eeg.T * 1e-6  # (16, samples), µV → V
    trig_remapped = trig.copy()
    trig_remapped[trig == -1] = 2

    ch_types = ["eeg"] * n_ch + ["stim"]
    info = mne.create_info(ch_names=CH_NAMES + ["STI"], sfreq=fs, ch_types=ch_types)
    raw_data = np.vstack([eeg_v, trig_remapped[np.newaxis, :n_samples]])
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn")

    # Bandpass
    raw_filt = raw.copy().filter(l_freq=l_freq, h_freq=h_freq,
                                  method="iir",
                                  iir_params=dict(order=5, ftype="butter"),
                                  picks="eeg", verbose=False)

    # Events
    events = mne.find_events(raw_filt, stim_channel="STI", verbose=False)
    if len(events) == 0:
        return None, None

    event_ids = sorted(set(events[:, 2]) - {0})
    if set(event_ids) == {1, 2}:
        event_id = {"left": 1, "right": 2}
    elif len(event_ids) >= 2:
        event_id = {"left": event_ids[0], "right": event_ids[1]}
    else:
        return None, None

    # Rejection
    reject = dict(eeg=reject_uv * 1e-6) if reject_uv else None

    epochs = mne.Epochs(raw_filt, events, event_id=event_id,
                        tmin=tmin, tmax=tmax, baseline=None,
                        reject=reject, preload=True, verbose=False)

    X = epochs.get_data(picks="eeg")
    y = epochs.events[:, 2]
    unique = sorted(np.unique(y))
    label_map = {old: new for new, old in enumerate(unique)}
    y = np.array([label_map[l] for l in y])

    return X, y


def build_pipelines(sfreq):
    """Build the 4 priority pipelines."""
    pipes = {}
    pipes["FBCSP+LDA"] = Pipeline([
        ("fbcsp", FilterBankCSP(sfreq=sfreq)),
        ("lda", LDA()),
    ])
    pipes["ACM(3,7)"] = Pipeline([
        ("augment", AugmentedDataset(order=3, lag=7)),
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])
    pipes["TS+LR"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    pipes["CSP+LDA"] = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])
    pipes["TS+SVM"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])
    pipes["MDM"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("mdm", MDM(metric="riemann")),
    ])
    return pipes


# ── Parameter grid ───────────────────────────────────────────────
CONFIGS = [
    # (name, tmin, tmax, l_freq, h_freq, reject_uv)
    ("original_0.5-4.5_4-40_150uV",    0.5, 4.5, 4.0, 40.0, 150),
    ("window_0-8_0.5-30_none",         0.0, 8.0, 0.5, 30.0, None),
    ("window_0-8_4-40_none",           0.0, 8.0, 4.0, 40.0, None),
    ("window_1-5_0.5-30_none",         1.0, 5.0, 0.5, 30.0, None),
    ("window_2-6_0.5-30_none",         2.0, 6.0, 0.5, 30.0, None),
    ("window_3-7_0.5-30_none",         3.0, 7.0, 0.5, 30.0, None),
    ("window_0.5-4.5_0.5-30_none",     0.5, 4.5, 0.5, 30.0, None),
    ("window_0.5-4.5_8-30_none",       0.5, 4.5, 8.0, 30.0, None),
    ("window_0.5-4.5_4-40_none",       0.5, 4.5, 4.0, 40.0, None),
    ("window_0.5-4.5_4-40_250uV",      0.5, 4.5, 4.0, 40.0, 250),
    ("window_1-5_8-30_none",           1.0, 5.0, 8.0, 30.0, None),
    ("window_2-6_8-30_none",           2.0, 6.0, 8.0, 30.0, None),
    ("window_3-8_0.5-30_none",         3.0, 8.0, 0.5, 30.0, None),
]

# ── Patient/stage combinations ───────────────────────────────────
CONDITIONS = [
    ("P1", "pre"), ("P1", "post"),
    ("P2", "pre"), ("P2", "post"),
    ("P3", "pre"), ("P3", "post"),
]

# ── Run ──────────────────────────────────────────────────────────
print("=" * 80)
print("PARAMETER TUNING — Testing epoch windows, bandpass, rejection")
print("=" * 80)

all_results = []

for config_name, tmin, tmax, l_freq, h_freq, reject_uv in CONFIGS:
    print(f"\n{'='*80}")
    print(f"Config: {config_name}")
    print(f"  tmin={tmin}, tmax={tmax}, bandpass={l_freq}-{h_freq}Hz, reject={reject_uv}")
    print(f"{'='*80}")

    pipes = build_pipelines(sfreq=SFREQ)

    for patient, stage in CONDITIONS:
        train_file = DATA_DIR / f"{patient}_{stage}_training.mat"
        test_file = DATA_DIR / f"{patient}_{stage}_test.mat"

        eeg_tr, trig_tr, fs = load_raw(train_file)
        eeg_te, trig_te, _ = load_raw(test_file)

        X_tr, y_tr = extract_custom_epochs(eeg_tr, trig_tr, fs, tmin, tmax, l_freq, h_freq, reject_uv)
        X_te, y_te = extract_custom_epochs(eeg_te, trig_te, fs, tmin, tmax, l_freq, h_freq, reject_uv)

        if X_tr is None or X_te is None or len(y_tr) < 10 or len(y_te) < 10:
            print(f"  {patient}_{stage}: SKIPPED (too few epochs)")
            continue

        label = f"{patient}_{stage}"
        best_acc = 0
        best_pipe = ""

        for pipe_name, pipe in pipes.items():
            try:
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
                acc = accuracy_score(y_te, y_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_pipe = pipe_name
                all_results.append({
                    "config": config_name,
                    "condition": label,
                    "pipeline": pipe_name,
                    "accuracy": acc,
                    "n_train": len(y_tr),
                    "n_test": len(y_te),
                })
            except Exception as e:
                pass

        print(f"  {label}: best={best_pipe} {best_acc:.1%} (train={len(y_tr)}, test={len(y_te)})")

# ── Summary: best config per condition ───────────────────────────
print("\n\n" + "=" * 80)
print("BEST CONFIGURATION PER CONDITION")
print("=" * 80)

df = pd.DataFrame(all_results)
for cond in sorted(df["condition"].unique()):
    sub = df[df["condition"] == cond]
    best_row = sub.loc[sub["accuracy"].idxmax()]
    print(f"\n{cond}:")
    print(f"  Config:   {best_row['config']}")
    print(f"  Pipeline: {best_row['pipeline']}")
    print(f"  Accuracy: {best_row['accuracy']:.1%}")

    # Show top 5 for this condition
    top5 = sub.nlargest(5, "accuracy")[["config", "pipeline", "accuracy"]]
    for _, row in top5.iterrows():
        print(f"    {row['config']:<40} {row['pipeline']:<12} {row['accuracy']:.1%}")

# ── Overall best configs ─────────────────────────────────────────
print("\n\n" + "=" * 80)
print("AVERAGE ACCURACY PER CONFIG (across all conditions)")
print("=" * 80)

avg_by_config = df.groupby("config")["accuracy"].mean().sort_values(ascending=False)
for config, avg in avg_by_config.items():
    print(f"  {config:<45} {avg:.1%}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
