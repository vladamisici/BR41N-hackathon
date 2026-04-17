"""Transfer learning for cross-patient BCI using Riemannian Procrustes Analysis.

Uses pyriemann.transfer to align covariance matrices across patients
and enable cross-subject classification.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.transfer import TLCenter, TLRotate, TLStretch, TLClassifier

import mne

logger = logging.getLogger(__name__)


def rpa_transfer_pipeline(
    X_source: NDArray,
    y_source: NDArray,
    X_target: NDArray,
    y_target: NDArray,
) -> dict[str, Any]:
    """Apply Riemannian Procrustes Analysis for cross-patient transfer.

    Pipeline: TLCenter → TLScale → TLRotate → TLClassifier(MDM).
    Trains on source domain, evaluates on target domain.

    Parameters
    ----------
    X_source : NDArray, shape (n_source, n_channels, n_times)
        Source patient epochs.
    y_source : NDArray, shape (n_source,)
        Source patient labels.
    X_target : NDArray, shape (n_target, n_channels, n_times)
        Target patient epochs.
    y_target : NDArray, shape (n_target,)
        Target patient labels.

    Returns
    -------
    dict
        Keys: accuracy, predictions, pipeline.
    """
    # Compute covariance matrices
    cov_est = Covariances(estimator="oas")
    cov_source = cov_est.fit_transform(X_source)
    cov_target = cov_est.transform(X_target)

    # Build domain labels
    domain_source = np.zeros(len(y_source), dtype=int)
    domain_target = np.ones(len(y_target), dtype=int)

    # Stack source + target
    cov_all = np.concatenate([cov_source, cov_target], axis=0)
    y_all = np.concatenate([y_source, y_target], axis=0)
    domain_all = np.concatenate([domain_source, domain_target], axis=0)

    # RPA pipeline
    centering = TLCenter(target_domain="target_domain")
    scaling = TLStretch(target_domain="target_domain")
    rotation = TLRotate(target_domain="target_domain", metric="riemann")

    # Fit transforms on all data with domain info
    sample_weight = domain_all  # domain label used internally
    cov_centered = centering.fit_transform(cov_all, y_all, sample_domain=domain_all)
    cov_scaled = scaling.fit_transform(cov_centered, y_all, sample_domain=domain_all)
    cov_aligned = rotation.fit_transform(cov_scaled, y_all, sample_domain=domain_all)

    # Train MDM on source, predict target
    n_source = len(y_source)
    cov_source_aligned = cov_aligned[:n_source]
    cov_target_aligned = cov_aligned[n_source:]

    mdm = MDM(metric="riemann")
    mdm.fit(cov_source_aligned, y_source)
    predictions = mdm.predict(cov_target_aligned)
    accuracy = np.mean(predictions == y_target)

    logger.info("RPA transfer accuracy: %.3f", accuracy)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "pipeline": {"centering": centering, "scaling": scaling,
                      "rotation": rotation, "mdm": mdm},
    }


def cross_patient_transfer(
    patient_data_dict: dict[str, tuple[NDArray, NDArray, mne.Epochs]],
) -> pd.DataFrame:
    """Leave-one-patient-out transfer learning evaluation.

    For each target patient, uses all other patients as source domain.

    Parameters
    ----------
    patient_data_dict : dict
        Mapping of patient_id → (X, y, epochs).

    Returns
    -------
    pd.DataFrame
        Per-target-patient results with columns:
        target_patient, source_patients, rpa_accuracy, no_transfer_accuracy.
    """
    patient_ids = sorted(patient_data_dict.keys())
    results: list[dict[str, Any]] = []

    for target_id in patient_ids:
        X_target, y_target, _ = patient_data_dict[target_id]

        # Pool all other patients as source
        source_Xs: list[NDArray] = []
        source_ys: list[NDArray] = []
        source_ids: list[str] = []

        for src_id in patient_ids:
            if src_id == target_id:
                continue
            X_src, y_src, _ = patient_data_dict[src_id]
            source_Xs.append(X_src)
            source_ys.append(y_src)
            source_ids.append(src_id)

        X_source = np.concatenate(source_Xs, axis=0)
        y_source = np.concatenate(source_ys, axis=0)

        logger.info(
            "Transfer: %s (target, %d trials) ← %s (source, %d trials)",
            target_id, len(y_target), source_ids, len(y_source),
        )

        # RPA transfer
        try:
            rpa_result = rpa_transfer_pipeline(X_source, y_source, X_target, y_target)
            rpa_acc = rpa_result["accuracy"]
        except Exception as exc:
            logger.error("RPA transfer failed for %s: %s", target_id, exc)
            rpa_acc = np.nan

        # No-transfer baseline: train MDM on source, test on target directly
        try:
            cov_est = Covariances(estimator="oas")
            cov_source = cov_est.fit_transform(X_source)
            cov_target = cov_est.transform(X_target)
            mdm_baseline = MDM(metric="riemann")
            mdm_baseline.fit(cov_source, y_source)
            no_transfer_acc = np.mean(mdm_baseline.predict(cov_target) == y_target)
        except Exception as exc:
            logger.error("No-transfer baseline failed for %s: %s", target_id, exc)
            no_transfer_acc = np.nan

        results.append({
            "target_patient": target_id,
            "source_patients": ", ".join(source_ids),
            "rpa_accuracy": rpa_acc,
            "no_transfer_accuracy": no_transfer_acc,
        })

    return pd.DataFrame(results)
