"""Data loading utilities for g.tec recoveriX .mat files.

Handles both scipy.io.loadmat (MATLAB v5/v7) and h5py (MATLAB v7.3+).
Builds MNE Raw objects with proper montage and stimulus channels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# 16-channel sensorimotor montage (10/20)
CH_NAMES: list[str] = [
    "FC5", "FC1", "FCz", "FC2", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP1", "CP2", "CP6",
]

SFREQ: float = 500.0

# Hemisphere channel indices
LEFT_IDX: list[int] = [0, 1, 5, 6, 7, 12, 13]
RIGHT_IDX: list[int] = [3, 4, 9, 10, 11, 14, 15]
MIDLINE_IDX: list[int] = [2, 8]


def _try_scipy_load(mat_path: str | Path) -> dict[str, Any] | None:
    """Attempt loading with scipy.io.loadmat."""
    from scipy.io import loadmat

    try:
        return loadmat(str(mat_path), squeeze_me=True)
    except Exception as exc:
        logger.debug("scipy.io.loadmat failed: %s", exc)
        return None


def _try_h5py_load(mat_path: str | Path) -> dict[str, Any] | None:
    """Attempt loading with h5py for MATLAB v7.3+ files."""
    try:
        import h5py
    except ImportError:
        logger.warning("h5py not installed — cannot read v7.3 .mat files")
        return None

    try:
        with h5py.File(str(mat_path), "r") as f:
            return {key: np.array(f[key]) for key in f.keys()}
    except Exception as exc:
        logger.debug("h5py load failed: %s", exc)
        return None


def _detect_field(data: dict[str, Any], candidates: list[str]) -> NDArray:
    """Auto-detect a field from a list of candidate names.

    Parameters
    ----------
    data : dict
        Loaded .mat file contents.
    candidates : list[str]
        Field names to try in priority order.

    Returns
    -------
    NDArray
        The detected array.

    Raises
    ------
    KeyError
        If none of the candidates are found.
    """
    for name in candidates:
        if name in data:
            logger.info("Detected field '%s'", name)
            return np.asarray(data[name])
    available = [k for k in data.keys() if not k.startswith("__")]
    raise KeyError(
        f"None of {candidates} found in file. Available keys: {available}"
    )


def load_gtec_stroke_data(
    mat_path: str | Path,
) -> tuple[mne.io.RawArray, float]:
    """Load a g.tec recoveriX .mat file into an MNE RawArray.

    Tries scipy.io.loadmat first, falls back to h5py for v7.3 files.
    Auto-detects EEG and trigger field names.

    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file.

    Returns
    -------
    raw : mne.io.RawArray
        Raw EEG data with STI channel appended, standard_1020 montage.
    sfreq : float
        Sampling frequency (Hz).
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"File not found: {mat_path}")

    # Try scipy first, then h5py
    data = _try_scipy_load(mat_path)
    if data is None:
        data = _try_h5py_load(mat_path)
    if data is None:
        raise RuntimeError(f"Could not load {mat_path} with scipy or h5py")

    # Detect sampling frequency
    sfreq = SFREQ
    for fs_key in ("fs", "sfreq", "Fs", "SampleRate"):
        if fs_key in data:
            sfreq = float(np.squeeze(data[fs_key]))
            logger.info("Detected sfreq=%.1f from field '%s'", sfreq, fs_key)
            break

    # Detect EEG data — shape should be (n_samples, 16) or (16, n_samples)
    eeg_candidates = ["y", "data", "eeg", "EEG", "X", "signal"]
    eeg = _detect_field(data, eeg_candidates).astype(np.float64)

    # Ensure shape is (n_channels, n_samples)
    if eeg.ndim == 2:
        if eeg.shape[0] == len(CH_NAMES):
            pass  # already (16, n_samples)
        elif eeg.shape[1] == len(CH_NAMES):
            eeg = eeg.T  # was (n_samples, 16) → transpose
        else:
            raise ValueError(
                f"EEG shape {eeg.shape} does not match {len(CH_NAMES)} channels"
            )
    else:
        raise ValueError(f"Expected 2D EEG array, got shape {eeg.shape}")

    # Scale µV → V
    eeg = eeg * 1e-6

    # Detect trigger channel
    trig_candidates = ["trig", "trigger", "stim", "Trig", "markers", "events"]
    trig = _detect_field(data, trig_candidates).astype(np.float64)
    trig = trig.squeeze()
    if trig.ndim == 2:
        trig = trig.flatten()

    # Build MNE info and RawArray
    ch_types = ["eeg"] * len(CH_NAMES) + ["stim"]
    all_names = CH_NAMES + ["STI"]
    info = mne.create_info(ch_names=all_names, sfreq=sfreq, ch_types=ch_types)

    # Stack EEG + trigger
    all_data = np.vstack([eeg, trig[np.newaxis, :eeg.shape[1]]])
    raw = mne.io.RawArray(all_data, info, verbose=False)

    # Set standard 10/20 montage for EEG channels
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn")

    logger.info(
        "Loaded %s: %d channels, %d samples, sfreq=%.1f Hz",
        mat_path.name, len(CH_NAMES), eeg.shape[1], sfreq,
    )
    return raw, sfreq


def extract_epochs(
    raw: mne.io.RawArray,
    tmin: float = 0.5,
    tmax: float = 4.5,
    l_freq: float = 4.0,
    h_freq: float = 40.0,
) -> tuple[NDArray, NDArray, mne.Epochs]:
    """Extract and preprocess epochs from Raw data.

    Applies Butterworth IIR bandpass, finds events from STI channel,
    creates epochs with 150 µV artifact rejection.

    Parameters
    ----------
    raw : mne.io.RawArray
        Raw data with STI channel.
    tmin : float
        Epoch start relative to event (seconds).
    tmax : float
        Epoch end relative to event (seconds).
    l_freq : float
        Lower bandpass frequency (Hz).
    h_freq : float
        Upper bandpass frequency (Hz).

    Returns
    -------
    X : NDArray, shape (n_epochs, 16, n_times)
        Epoch data.
    y : NDArray, shape (n_epochs,)
        Integer class labels.
    epochs : mne.Epochs
        MNE Epochs object for further analysis.
    """
    # Bandpass filter (Butterworth IIR)
    raw_filt = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params=dict(order=5, ftype="butter"),
        picks="eeg",
        verbose=False,
    )

    # Find events from STI channel
    events = mne.find_events(raw_filt, stim_channel="STI", verbose=False)
    if len(events) == 0:
        raise ValueError("No events found in STI channel")

    # Get unique event IDs (excluding 0)
    event_ids = sorted(set(events[:, 2]) - {0})
    logger.info("Found %d events with IDs: %s", len(events), event_ids)

    # Build event_id mapping — assume first two non-zero IDs are left/right
    if len(event_ids) >= 2:
        event_id = {"left": event_ids[0], "right": event_ids[1]}
    elif len(event_ids) == 1:
        event_id = {"class_1": event_ids[0]}
    else:
        raise ValueError(f"Unexpected event IDs: {event_ids}")

    # Create epochs with artifact rejection
    reject = dict(eeg=150e-6)  # 150 µV threshold
    epochs = mne.Epochs(
        raw_filt,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject=reject,
        preload=True,
        verbose=False,
    )

    n_dropped = len(events) - len(epochs)
    if n_dropped > 0:
        logger.info("Rejected %d/%d epochs (150 µV threshold)", n_dropped, len(events))

    X = epochs.get_data(picks="eeg")  # (n_epochs, 16, n_times)
    y = epochs.events[:, 2]

    # Remap labels to 0/1 integers
    unique_labels = sorted(np.unique(y))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])

    logger.info("Extracted %d epochs: shape %s, classes %s", len(y), X.shape, np.unique(y))
    return X, y, epochs


def load_all_patients(
    data_dir: str | Path,
) -> dict[str, tuple[NDArray, NDArray, mne.Epochs]]:
    """Load all patient .mat files from a directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .mat files.

    Returns
    -------
    dict
        Mapping of patient_id (filename stem) → (X, y, epochs).
    """
    data_dir = Path(data_dir)
    mat_files = sorted(data_dir.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")

    patient_data: dict[str, tuple[NDArray, NDArray, mne.Epochs]] = {}

    for mat_file in mat_files:
        patient_id = mat_file.stem
        logger.info("Loading patient: %s", patient_id)
        try:
            raw, sfreq = load_gtec_stroke_data(mat_file)
            X, y, epochs = extract_epochs(raw)
            patient_data[patient_id] = (X, y, epochs)
            logger.info("  → %d epochs, %d channels", X.shape[0], X.shape[1])
        except Exception as exc:
            logger.error("Failed to load %s: %s", patient_id, exc)
            raise

    return patient_data
