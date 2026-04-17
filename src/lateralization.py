"""Lateralization Index computation for stroke motor imagery.

LI = (ERD_contra - ERD_ipsi) / (ERD_contra + ERD_ipsi)

Key clinical biomarker correlating with Fugl-Meyer motor scores (r=0.57–0.61).
"""

from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

logger = logging.getLogger(__name__)


def compute_laterality_index(
    epochs_data: NDArray,
    sfreq: float,
    c3_idx: int = 6,
    c4_idx: int = 10,
) -> dict[str, float]:
    """Compute laterality index for mu and beta bands.

    Calculates Event-Related Desynchronization (ERD) power in C3 and C4
    channels and derives the laterality index.

    Parameters
    ----------
    epochs_data : NDArray, shape (n_epochs, n_channels, n_times)
        Epoch data.
    sfreq : float
        Sampling frequency in Hz.
    c3_idx : int
        Channel index for C3 (default 6 in the 16-ch montage).
    c4_idx : int
        Channel index for C4 (default 10 in the 16-ch montage).

    Returns
    -------
    dict[str, float]
        Keys: mu_li (8–13 Hz laterality index),
              beta_li (13–30 Hz laterality index),
              mu_c3_power, mu_c4_power,
              beta_c3_power, beta_c4_power.

    Notes
    -----
    Positive LI → contralateral dominance (healthy pattern).
    LI near 0 → bilateral activation (common in stroke).
    Negative LI → ipsilateral dominance (compensatory).
    """
    c3_data = epochs_data[:, c3_idx, :]  # (n_epochs, n_times)
    c4_data = epochs_data[:, c4_idx, :]

    # Compute PSD using Welch's method
    nperseg = min(int(sfreq * 2), c3_data.shape[1])

    freqs, psd_c3 = welch(c3_data, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_c4 = welch(c4_data, fs=sfreq, nperseg=nperseg, axis=-1)

    # Average PSD across epochs
    psd_c3_mean = psd_c3.mean(axis=0)
    psd_c4_mean = psd_c4.mean(axis=0)

    # Band extraction
    mu_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)

    mu_c3 = psd_c3_mean[mu_mask].mean()
    mu_c4 = psd_c4_mean[mu_mask].mean()
    beta_c3 = psd_c3_mean[beta_mask].mean()
    beta_c4 = psd_c4_mean[beta_mask].mean()

    # Laterality index: (contra - ipsi) / (contra + ipsi)
    # For right-hand MI: C3 is contralateral, C4 is ipsilateral
    # For left-hand MI: C4 is contralateral, C3 is ipsilateral
    # Here we compute a general LI (C3 vs C4 power ratio)
    eps = 1e-12  # avoid division by zero
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


def laterality_report(
    patient_data_dict: dict[str, tuple[NDArray, NDArray, mne.Epochs]],
) -> str:
    """Generate a formatted laterality report for all patients.

    Parameters
    ----------
    patient_data_dict : dict
        Mapping of patient_id → (X, y, epochs).

    Returns
    -------
    str
        Formatted multi-line report with per-patient LI values.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("LATERALIZATION INDEX REPORT")
    lines.append("=" * 60)
    lines.append("")

    all_results: dict[str, dict[str, float]] = {}

    for patient_id in sorted(patient_data_dict.keys()):
        X, y, epochs = patient_data_dict[patient_id]
        sfreq = epochs.info["sfreq"]

        li = compute_laterality_index(X, sfreq)
        all_results[patient_id] = li

        lines.append(f"Patient: {patient_id}")
        lines.append(f"  Epochs: {X.shape[0]}")
        lines.append(f"  Mu  LI (8–13 Hz):  {li['mu_li']:+.4f}")
        lines.append(f"  Beta LI (13–30 Hz): {li['beta_li']:+.4f}")
        lines.append(f"  Mu  C3 power: {li['mu_c3_power']:.6e}")
        lines.append(f"  Mu  C4 power: {li['mu_c4_power']:.6e}")
        lines.append(f"  Beta C3 power: {li['beta_c3_power']:.6e}")
        lines.append(f"  Beta C4 power: {li['beta_c4_power']:.6e}")

        # Interpretation
        if abs(li["mu_li"]) < 0.1:
            interp = "Bilateral (weak lateralization)"
        elif li["mu_li"] > 0:
            interp = "Left-hemisphere dominant (C3 > C4)"
        else:
            interp = "Right-hemisphere dominant (C4 > C3)"
        lines.append(f"  Interpretation: {interp}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("CLINICAL NOTES:")
    lines.append("  - Positive LI → contralateral dominance (healthy pattern)")
    lines.append("  - LI near 0 → bilateral (common post-stroke)")
    lines.append("  - Negative LI → ipsilateral (compensatory)")
    lines.append("  - LI correlates with Fugl-Meyer scores (r=0.57–0.61)")
    lines.append("=" * 60)

    report = "\n".join(lines)
    logger.info("Generated laterality report for %d patients", len(all_results))
    return report
