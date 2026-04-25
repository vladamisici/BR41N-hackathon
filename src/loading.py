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

# 16-channel sensorimotor montage (g.tec recoveriX, confirmed from hackathon slides)
#  #1  FC3    #2  FCz    #3  FC4
#  #4  C5     #5  C3     #6  C1     #7  Cz     #8  C2     #9  C4     #10 C6
#  #11 CP3    #12 CP1    #13 CPz    #14 CP2    #15 CP4
#  #16 Pz
CH_NAMES: list[str] = [
    "FC3", "FCz", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "Pz",
]

SFREQ: float = 500.0

# Hemisphere channel indices (0-indexed into CH_NAMES)
# Left:  FC3(0), C5(3), C3(4), C1(5), CP3(10), CP1(11)
# Right: FC4(2), C2(7), C4(8), C6(9), CP2(13), CP4(14)
# Midline: FCz(1), Cz(6), CPz(12), Pz(15)
LEFT_IDX: list[int] = [0, 3, 4, 5, 10, 11]
RIGHT_IDX: list[int] = [2, 7, 8, 9, 13, 14]
MIDLINE_IDX: list[int] = [1, 6, 12, 15]

# Key channel indices for lateralization
C3_IDX: int = 4   # C3 in the 16-ch montage
C4_IDX: int = 8   # C4 in the 16-ch montage

# Trigger codes (from hackathon slides)
TRIG_LEFT: int = 1    # +1 = left movement
TRIG_RIGHT: int = 2   # -1 in raw → remapped to 2 for MNE compatibility


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

    # Remap trigger codes for MNE compatibility:
    # Raw data uses +1 (left) and -1 (right).
    # MNE find_events expects positive integers, so remap -1 → 2.
    trig_remapped = trig.copy()
    trig_remapped[trig == -1] = 2
    logger.info(
        "Trigger codes: left(+1)=%d, right(-1→2)=%d, zero=%d",
        int(np.sum(trig == 1)),
        int(np.sum(trig == -1)),
        int(np.sum(trig == 0)),
    )

    # Build MNE info and RawArray
    ch_types = ["eeg"] * len(CH_NAMES) + ["stim"]
    all_names = CH_NAMES + ["STI"]
    info = mne.create_info(ch_names=all_names, sfreq=sfreq, ch_types=ch_types)

    # Stack EEG + trigger (use remapped triggers)
    all_data = np.vstack([eeg, trig_remapped[np.newaxis, :eeg.shape[1]]])
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
    tmin: float = 3.0,
    tmax: float = 7.0,
    l_freq: float = 0.5,
    h_freq: float = 30.0,
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

    # Build event_id mapping
    # recoveriX data: 1 = left hand, 2 = right hand (remapped from -1)
    # MOABB/other data: auto-detect first two non-zero IDs
    if set(event_ids) == {1, 2}:
        event_id = {"left": 1, "right": 2}
    elif len(event_ids) >= 2:
        event_id = {"left": event_ids[0], "right": event_ids[1]}
    elif len(event_ids) == 1:
        event_id = {"class_1": event_ids[0]}
    else:
        raise ValueError(f"Unexpected event IDs: {event_ids}")

    # Create epochs — no artifact rejection for stroke data
    # (stroke patients have higher amplitude signals, 150µV threshold
    #  was discarding too many valid trials)
    epochs = mne.Epochs(
        raw_filt,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject=None,
        preload=True,
        verbose=False,
    )

    n_total = len(events)
    n_kept = len(epochs)
    if n_kept < n_total:
        logger.info("Kept %d/%d epochs", n_kept, n_total)

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


def load_train_test(
    data_dir: str | Path,
) -> dict[str, dict[str, tuple[NDArray, NDArray, mne.Epochs]]]:
    """Load hackathon data organized by patient → stage → train/test split.

    Expects files named: P{n}_{stage}_{split}.mat
    e.g. P1_pre_training.mat, P1_pre_test.mat, P1_post_training.mat, etc.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing .mat files.

    Returns
    -------
    dict
        Nested mapping: patient_id → {
            "pre_train": (X, y, epochs),
            "pre_test": (X, y, epochs),
            "post_train": (X, y, epochs),
            "post_test": (X, y, epochs),
        }
    """
    data_dir = Path(data_dir)
    mat_files = sorted(data_dir.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")

    result: dict[str, dict[str, tuple[NDArray, NDArray, mne.Epochs]]] = {}

    for mat_file in mat_files:
        stem = mat_file.stem  # e.g. "P1_pre_training"
        parts = stem.split("_")

        if len(parts) >= 3:
            patient = parts[0]           # "P1"
            stage = parts[1]             # "pre" or "post"
            split = parts[2]             # "training" or "test"
        else:
            # Fallback: load as generic
            patient = stem
            stage = "unknown"
            split = "unknown"

        key = f"{stage}_{'train' if split == 'training' else split}"

        if patient not in result:
            result[patient] = {}

        logger.info("Loading %s → %s/%s", mat_file.name, patient, key)
        try:
            raw, sfreq = load_gtec_stroke_data(mat_file)
            X, y, epochs = extract_epochs(raw)
            result[patient][key] = (X, y, epochs)
            logger.info("  → %d epochs, %d channels", X.shape[0], X.shape[1])
        except Exception as exc:
            logger.error("Failed to load %s: %s", mat_file.name, exc)
            raise

    return result
