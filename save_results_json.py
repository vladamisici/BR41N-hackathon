#!/usr/bin/env python
"""Save all hackathon results as JSON for the Streamlit dashboard."""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"

import json
import numpy as np
import mne
mne.set_log_level("ERROR")

from sklearn.base import clone
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from src.loading import load_train_test, C3_IDX, C4_IDX
from src.classifiers import build_all_pipelines
from src.lateralization import compute_laterality_index

DATA_DIR = "dataset/stroke-rehab/"
SFREQ = 256.0

data = load_train_test(DATA_DIR)
pipelines = build_all_pipelines(sfreq=SFREQ)

results = {}

for patient in sorted(data.keys()):
    results[patient] = {}
    for stage in ["pre", "post"]:
        train_key = f"{stage}_train"
        test_key = f"{stage}_test"
        if train_key not in data[patient]:
            continue

        X_tr, y_tr, _ = data[patient][train_key]
        X_te, y_te, _ = data[patient][test_key]

        label = f"{stage}"
        results[patient][label] = {
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "pipelines": {},
        }

        # Run all pipelines
        for name, pipe in pipelines.items():
            p = clone(pipe)
            try:
                p.fit(X_tr, y_tr)
                y_pred = p.predict(X_te)
                acc = float(accuracy_score(y_te, y_pred))
                kappa = float(cohen_kappa_score(y_te, y_pred))
                cm = confusion_matrix(y_te, y_pred).tolist()
                results[patient][label]["pipelines"][name] = {
                    "accuracy": acc,
                    "kappa": kappa,
                    "confusion_matrix": cm,
                }
            except Exception as e:
                results[patient][label]["pipelines"][name] = {
                    "accuracy": None,
                    "kappa": None,
                    "confusion_matrix": None,
                    "error": str(e),
                }

        # Lateralization
        li = compute_laterality_index(X_tr, SFREQ, c3_idx=C3_IDX, c4_idx=C4_IDX)
        results[patient][label]["lateralization"] = {
            k: float(v) for k, v in li.items()
        }

# Baselines from hackathon slides
results["baselines"] = {
    "P1_pre": {"CSP+LDA": 0.771, "PCA+TVLDA": 0.929},
    "P1_post": {"CSP+LDA": 0.939, "PCA+TVLDA": 0.970},
    "P2_pre": {"CSP+LDA": 0.684, "PCA+TVLDA": 0.724},
    "P2_post": {"CSP+LDA": 0.961, "PCA+TVLDA": 0.974},
    "P3_pre": {"CSP+LDA": 0.744, "PCA+TVLDA": 0.936},
    "P3_post": {"CSP+LDA": 0.797, "PCA+TVLDA": 1.000},
}

with open("dashboard_data.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to dashboard_data.json")
print(f"Patients: {[k for k in results if k != 'baselines']}")
for p in sorted(k for k in results if k != "baselines"):
    for s in ["pre", "post"]:
        if s in results[p]:
            best = max(results[p][s]["pipelines"].items(),
                      key=lambda x: x[1]["accuracy"] or 0)
            print(f"  {p}_{s}: {best[0]} = {best[1]['accuracy']:.1%}")
