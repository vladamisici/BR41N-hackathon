#!/usr/bin/env python
"""
MOABB BNCI2014_001 Analysis Script
Steps 2-7: Download data, evaluate pipelines, FBCSP+LDA, ACM, lateralization, summary.
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import os
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import numpy as np
import pandas as pd
import mne
mne.set_log_level("ERROR")
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM

from scipy.signal import welch

# ============================================================
# STEP 2: Download BNCI2014_001, filter to LEFT_HAND vs RIGHT_HAND,
#          select 16 channels, extract epochs 0.5-4.5s, bandpass 4-40Hz
# ============================================================
print("=" * 70)
print("STEP 2: Loading BNCI2014_001 from MOABB")
print("=" * 70)

from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

dataset = BNCI2014_001()

# The 16 channels closest to our g.tec montage
TARGET_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CP2", "CP4",
]

# Subjects to analyze
SUBJECTS = [1, 3, 7]

# Hemisphere indices in our 16-channel montage
# FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CP2, CP4
#  0    1    2    3    4    5   6   7   8   9  10  11   12   13   14   15
LEFT_IDX = [0, 1, 5, 6, 7, 12, 13]    # FC3, FC1, C5, C3, C1, CP3, CP1
RIGHT_IDX = [3, 4, 9, 10, 11, 14, 15]  # FC2, FC4, C2, C4, C6, CP2, CP4
MIDLINE_IDX = [2, 8]                    # FCz, Cz

# C3 and C4 indices in our 16-channel montage
C3_IDX = 6   # C3
C4_IDX = 10  # C4

# Use LeftRightImagery paradigm: filters to left_hand vs right_hand
paradigm = LeftRightImagery(
    fmin=4, fmax=40,
    tmin=0.5, tmax=4.5,
    channels=TARGET_CHANNELS,
    resample=None,
)

# Load data for our 3 subjects
subject_data = {}
for subj in SUBJECTS:
    print(f"\nLoading subject {subj}...")
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
    # Encode labels as integers
    labels = np.unique(y)
    label_map = {lab: i for i, lab in enumerate(sorted(labels))}
    y_int = np.array([label_map[lab] for lab in y])
    subject_data[subj] = (X, y_int, meta)
    print(f"  Subject {subj}: X={X.shape}, classes={labels}, "
          f"n_left={np.sum(y_int==0)}, n_right={np.sum(y_int==1)}")

print(f"\nChannel montage: {TARGET_CHANNELS}")
print(f"Epoch window: 0.5-4.5s, Bandpass: 4-40 Hz")
print(f"Subjects loaded: {SUBJECTS}")


# ============================================================
# STEP 3: Run evaluate_all on subjects 1, 3, 7
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: evaluate_all on subjects 1, 3, 7")
print("=" * 70)

def build_all_pipelines():
    """Build all classification pipelines."""
    pipelines = {}

    # 1. CSP + LDA baseline
    pipelines["CSP+LDA"] = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])

    # 2. Tangent Space + Logistic Regression (Riemannian)
    pipelines["TS+LR"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])

    # 3. Minimum Distance to Mean (Riemannian)
    pipelines["MDM"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("mdm", MDM(metric="riemann")),
    ])

    # 4. Fisher Geodesic MDM (Riemannian)
    pipelines["FgMDM"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("fgmdm", FgMDM(metric="riemann")),
    ])

    # 5. Tangent Space + SVM (Riemannian)
    pipelines["TS+SVM"] = Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("ts", TangentSpace(metric="riemann")),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])

    # 6. Tangent Space + LDA (Riemannian)
    pipelines["TS+LDA"] = Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("ts", TangentSpace(metric="riemann")),
        ("lda", LDA()),
    ])

    return pipelines


def evaluate_all(X, y, pipelines, n_splits=5):
    """Evaluate all pipelines with stratified k-fold CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for name, pipe in pipelines.items():
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            results.append({
                "pipeline": name,
                "mean_acc": scores.mean(),
                "std_acc": scores.std(),
                "fold_scores": scores.tolist(),
            })
        except Exception as exc:
            print(f"  {name} FAILED: {exc}")
            results.append({
                "pipeline": name,
                "mean_acc": np.nan,
                "std_acc": np.nan,
                "fold_scores": [],
            })
    df = pd.DataFrame(results).sort_values("mean_acc", ascending=False)
    return df.reset_index(drop=True)


pipelines = build_all_pipelines()
all_results = {}

for subj in SUBJECTS:
    X, y, meta = subject_data[subj]
    print(f"\n--- Subject {subj} ({X.shape[0]} epochs) ---")
    df = evaluate_all(X, y, pipelines)
    all_results[subj] = df
    print(df[["pipeline", "mean_acc", "std_acc"]].to_string(index=False))


# ============================================================
# STEP 4: Add and run FBCSP+LDA pipeline
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: FBCSP+LDA pipeline")
print("=" * 70)


class FilterBank(BaseEstimator, TransformerMixin):
    """Filter bank: splits signal into sub-bands, applies CSP to each,
    and concatenates the CSP features.

    Parameters
    ----------
    bands : list of (low, high) tuples
        Frequency bands for the filter bank.
    sfreq : float
        Sampling frequency.
    n_components : int
        Number of CSP components per band.
    """

    def __init__(self, bands=None, sfreq=250.0, n_components=4):
        self.bands = bands or [
            (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30)
        ]
        self.sfreq = sfreq
        self.n_components = n_components
        self.csps_ = []

    def fit(self, X, y):
        self.csps_ = []
        for low, high in self.bands:
            X_filt = mne.filter.filter_data(
                X.astype(np.float64), self.sfreq, low, high,
                method="iir",
                iir_params=dict(order=5, ftype="butter"),
                verbose=False,
            )
            csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)
            csp.fit(X_filt, y)
            self.csps_.append((low, high, csp))
        return self

    def transform(self, X):
        features = []
        for low, high, csp in self.csps_:
            X_filt = mne.filter.filter_data(
                X.astype(np.float64), self.sfreq, low, high,
                method="iir",
                iir_params=dict(order=5, ftype="butter"),
                verbose=False,
            )
            features.append(csp.transform(X_filt))
        return np.hstack(features)


# Determine sfreq from the data (BNCI2014_001 is 250 Hz)
sfreq = 250.0

fbcsp_pipeline = Pipeline([
    ("fbcsp", FilterBank(
        bands=[(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30)],
        sfreq=sfreq,
        n_components=4,
    )),
    ("lda", LDA()),
])

fbcsp_results = {}
for subj in SUBJECTS:
    X, y, meta = subject_data[subj]
    print(f"\n--- Subject {subj} ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(fbcsp_pipeline, X, y, cv=cv, scoring="accuracy")
    fbcsp_results[subj] = {"mean_acc": scores.mean(), "std_acc": scores.std(),
                           "fold_scores": scores.tolist()}
    print(f"  FBCSP+LDA: {scores.mean():.4f} ± {scores.std():.4f}")
    # Add to all_results for this subject
    new_row = pd.DataFrame([{
        "pipeline": "FBCSP+LDA",
        "mean_acc": scores.mean(),
        "std_acc": scores.std(),
        "fold_scores": scores.tolist(),
    }])
    all_results[subj] = pd.concat([all_results[subj], new_row], ignore_index=True)
    all_results[subj] = all_results[subj].sort_values("mean_acc", ascending=False).reset_index(drop=True)


# ============================================================
# STEP 5: Run ACM(3,7) on the same 3 subjects. Compare to TS+LR.
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: ACM(order=3, lag=7) vs TS+LR")
print("=" * 70)


class AugmentedDataset(BaseEstimator, TransformerMixin):
    """Takens delay embedding for ACM."""

    def __init__(self, order=3, lag=7):
        self.order = order
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_epochs, n_channels, n_times = X.shape
        max_delay = (self.order - 1) * self.lag
        if max_delay >= n_times:
            raise ValueError(
                f"Delay too large: (order-1)*lag={max_delay} >= n_times={n_times}"
            )
        trimmed_len = n_times - max_delay
        augmented = np.zeros((n_epochs, n_channels * self.order, trimmed_len))
        for k in range(self.order):
            offset = k * self.lag
            augmented[:, k * n_channels:(k + 1) * n_channels, :] = \
                X[:, :, offset:offset + trimmed_len]
        return augmented


acm_pipeline = Pipeline([
    ("augment", AugmentedDataset(order=3, lag=7)),
    ("cov", Covariances(estimator="oas")),
    ("ts", TangentSpace(metric="riemann")),
    ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
])

acm_results = {}
for subj in SUBJECTS:
    X, y, meta = subject_data[subj]
    print(f"\n--- Subject {subj} ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(acm_pipeline, X, y, cv=cv, scoring="accuracy")
    acm_results[subj] = {"mean_acc": scores.mean(), "std_acc": scores.std(),
                         "fold_scores": scores.tolist()}
    ts_lr_acc = all_results[subj].loc[
        all_results[subj]["pipeline"] == "TS+LR", "mean_acc"
    ].values[0]
    diff = scores.mean() - ts_lr_acc
    beat = "YES ✓" if diff > 0 else "NO ✗"
    print(f"  ACM(3,7):  {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  TS+LR:     {ts_lr_acc:.4f}")
    print(f"  ACM beats TS+LR? {beat} (Δ = {diff:+.4f})")

    # Add ACM to all_results
    new_row = pd.DataFrame([{
        "pipeline": "ACM(3,7)",
        "mean_acc": scores.mean(),
        "std_acc": scores.std(),
        "fold_scores": scores.tolist(),
    }])
    all_results[subj] = pd.concat([all_results[subj], new_row], ignore_index=True)
    all_results[subj] = all_results[subj].sort_values("mean_acc", ascending=False).reset_index(drop=True)


# ============================================================
# STEP 6: Compute lateralization index for each subject
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Lateralization Index")
print("=" * 70)


def compute_laterality_index(epochs_data, sfreq, c3_idx=6, c4_idx=10):
    """Compute laterality index for mu and beta bands."""
    c3_data = epochs_data[:, c3_idx, :]
    c4_data = epochs_data[:, c4_idx, :]

    nperseg = min(int(sfreq * 2), c3_data.shape[1])
    freqs, psd_c3 = welch(c3_data, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_c4 = welch(c4_data, fs=sfreq, nperseg=nperseg, axis=-1)

    psd_c3_mean = psd_c3.mean(axis=0)
    psd_c4_mean = psd_c4.mean(axis=0)

    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)

    mu_c3 = psd_c3_mean[mu_mask].mean()
    mu_c4 = psd_c4_mean[mu_mask].mean()
    beta_c3 = psd_c3_mean[beta_mask].mean()
    beta_c4 = psd_c4_mean[beta_mask].mean()

    eps = 1e-12
    mu_li = (mu_c3 - mu_c4) / (mu_c3 + mu_c4 + eps)
    beta_li = (beta_c3 - beta_c4) / (beta_c3 + beta_c4 + eps)

    return {
        "mu_li": float(mu_li),
        "beta_li": float(beta_li),
        "mu_c3_power": float(mu_c3),
        "mu_c4_power": float(mu_c4),
        "beta_c3_power": float(beta_c3),
        "beta_c4_power": float(beta_c4),
    }


li_results = {}
for subj in SUBJECTS:
    X, y, meta = subject_data[subj]
    li = compute_laterality_index(X, sfreq, c3_idx=C3_IDX, c4_idx=C4_IDX)
    li_results[subj] = li

    if abs(li["mu_li"]) < 0.1:
        interp = "Bilateral (weak lateralization)"
    elif li["mu_li"] > 0:
        interp = "Left-hemisphere dominant (C3 > C4)"
    else:
        interp = "Right-hemisphere dominant (C4 > C3)"

    print(f"\nSubject {subj}:")
    print(f"  Mu  LI (8-13 Hz):  {li['mu_li']:+.4f}")
    print(f"  Beta LI (13-30 Hz): {li['beta_li']:+.4f}")
    print(f"  Mu  C3={li['mu_c3_power']:.4e}  C4={li['mu_c4_power']:.4e}")
    print(f"  Beta C3={li['beta_c3_power']:.4e}  C4={li['beta_c4_power']:.4e}")
    print(f"  Interpretation: {interp}")


# ============================================================
# STEP 7: Final summary
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: FINAL SUMMARY")
print("=" * 70)

print("\n--- Full Results Per Subject (all pipelines) ---\n")
for subj in SUBJECTS:
    print(f"Subject {subj}:")
    df = all_results[subj]
    print(df[["pipeline", "mean_acc", "std_acc"]].to_string(index=False))
    winner = df.iloc[0]
    print(f"  → WINNER: {winner['pipeline']} ({winner['mean_acc']:.4f})")
    print()

print("\n--- Best Pipeline Per Subject ---\n")
summary_rows = []
for subj in SUBJECTS:
    df = all_results[subj]
    winner = df.iloc[0]
    li = li_results[subj]
    summary_rows.append({
        "Subject": subj,
        "Best Pipeline": winner["pipeline"],
        "Accuracy": f"{winner['mean_acc']:.4f}",
        "Mu LI": f"{li['mu_li']:+.4f}",
        "Beta LI": f"{li['beta_li']:+.4f}",
        "LI Pattern": "Bilateral" if abs(li["mu_li"]) < 0.1
                       else ("L-dom" if li["mu_li"] > 0 else "R-dom"),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

print("\n--- Does LI Predict Which Pipeline Works Best? ---\n")
for subj in SUBJECTS:
    li = li_results[subj]
    winner = all_results[subj].iloc[0]["pipeline"]
    mu_li = li["mu_li"]

    # Check if Riemannian methods win for bilateral/weak lateralization
    riemannian_methods = {"TS+LR", "TS+SVM", "TS+LDA", "MDM", "FgMDM", "ACM(3,7)"}
    csp_methods = {"CSP+LDA", "FBCSP+LDA"}

    is_riemannian = winner in riemannian_methods
    is_csp = winner in csp_methods
    is_bilateral = abs(mu_li) < 0.1

    print(f"Subject {subj}: LI={mu_li:+.4f} ({'bilateral' if is_bilateral else 'lateralized'}), "
          f"Winner={winner} ({'Riemannian' if is_riemannian else 'CSP-based'})")

    if is_bilateral and is_riemannian:
        print(f"  → Consistent: bilateral LI + Riemannian winner "
              f"(Riemannian handles diffuse patterns better)")
    elif not is_bilateral and is_csp:
        print(f"  → Consistent: lateralized LI + CSP winner "
              f"(CSP exploits clear spatial patterns)")
    elif not is_bilateral and is_riemannian:
        print(f"  → Riemannian wins even with lateralized LI "
              f"(Riemannian robust across conditions)")
    elif is_bilateral and is_csp:
        print(f"  → Surprising: bilateral LI but CSP wins "
              f"(may indicate strong frequency-domain features)")
    else:
        print(f"  → Mixed pattern")

print("\n--- Key Takeaways ---\n")
# Check if ACM beats TS+LR across subjects
acm_wins = 0
for subj in SUBJECTS:
    acm_acc = acm_results[subj]["mean_acc"]
    ts_lr_acc = all_results[subj].loc[
        all_results[subj]["pipeline"] == "TS+LR", "mean_acc"
    ].values[0]
    if acm_acc > ts_lr_acc:
        acm_wins += 1

print(f"1. ACM(3,7) beats TS+LR in {acm_wins}/{len(SUBJECTS)} subjects")

# Check FBCSP performance
for subj in SUBJECTS:
    fbcsp_acc = fbcsp_results[subj]["mean_acc"]
    csp_acc = all_results[subj].loc[
        all_results[subj]["pipeline"] == "CSP+LDA", "mean_acc"
    ].values[0]
    diff = fbcsp_acc - csp_acc
    print(f"2. Subject {subj}: FBCSP+LDA vs CSP+LDA: Δ = {diff:+.4f} "
          f"({'FBCSP better' if diff > 0 else 'CSP better'})")

# LI summary
print(f"\n3. Lateralization patterns:")
for subj in SUBJECTS:
    li = li_results[subj]
    print(f"   Subject {subj}: Mu LI={li['mu_li']:+.4f}, Beta LI={li['beta_li']:+.4f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
