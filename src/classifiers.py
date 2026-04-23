"""Classification pipelines for stroke MI-BCI.

Hackathon run order (validated on BNCI2014_001):
  1. FBCSP+LDA  — primary (filter-bank CSP, best on healthy + stroke)
  2. ACM(3,7)   — secondary (Takens delay embedding, captures temporal dynamics)
  3. TS+LR      — reliable backup (Riemannian tangent space, robust to noise)
  4. CSP+LDA    — baseline to beat

Additional Riemannian pipelines (MDM, FgMDM, TS+SVM, TS+LDA) included
for thorough comparison in the final report.
"""

from __future__ import annotations

import logging
from typing import Any

import mne
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

logger = logging.getLogger(__name__)

# Default FBCSP filter banks — covers theta through low-gamma,
# critical for stroke where ERD shifts to theta/low-alpha.
DEFAULT_FILTER_BANKS: list[tuple[float, float]] = [
    (4, 8),    # theta
    (8, 12),   # alpha / mu
    (12, 16),  # low beta
    (16, 20),  # mid beta
    (20, 24),  # high beta
    (24, 30),  # low gamma
]


# ---------------------------------------------------------------------------
# Filter Bank CSP
# ---------------------------------------------------------------------------

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter Bank Common Spatial Patterns.

    Splits the signal into sub-bands, applies CSP independently to each,
    and concatenates the log-variance features.

    Parameters
    ----------
    bands : list of (low, high) tuples
        Frequency bands for the filter bank.
    sfreq : float
        Sampling frequency in Hz.
    n_components : int
        Number of CSP components per band.
    """

    def __init__(
        self,
        bands: list[tuple[float, float]] | None = None,
        sfreq: float = 500.0,
        n_components: int = 4,
    ) -> None:
        self.bands = bands or DEFAULT_FILTER_BANKS
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(self, X: NDArray, y: NDArray) -> "FilterBankCSP":
        """Fit one CSP per sub-band."""
        from mne.decoding import CSP

        self.csps_: list[tuple[float, float, Any]] = []
        for low, high in self.bands:
            X_filt = mne.filter.filter_data(
                X.astype(np.float64), self.sfreq, low, high,
                method="iir",
                iir_params=dict(order=5, ftype="butter"),
                verbose=False,
            )
            csp = CSP(
                n_components=self.n_components,
                reg="ledoit_wolf",
                log=True,
            )
            csp.fit(X_filt, y)
            self.csps_.append((low, high, csp))
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Extract and concatenate CSP features from each sub-band."""
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


def build_fbcsp_pipeline(
    sfreq: float = 500.0,
    bands: list[tuple[float, float]] | None = None,
    n_components: int = 4,
) -> Pipeline:
    """Build a Filter Bank CSP + LDA pipeline.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    bands : list of (low, high) tuples, optional
        Frequency bands. Defaults to DEFAULT_FILTER_BANKS.
    n_components : int
        CSP components per band.

    Returns
    -------
    Pipeline
        FBCSP+LDA pipeline.
    """
    return Pipeline([
        ("fbcsp", FilterBankCSP(bands=bands, sfreq=sfreq, n_components=n_components)),
        ("lda", LDA()),
    ])


# ---------------------------------------------------------------------------
# Augmented Covariance Method (ACM)
# ---------------------------------------------------------------------------

class AugmentedDataset(BaseEstimator, TransformerMixin):
    """Takens delay embedding for the Augmented Covariance Method (ACM).

    Stacks time-delayed copies of the signal along the channel axis,
    creating an augmented representation that captures temporal dynamics
    in the covariance matrix.

    Parameters
    ----------
    order : int
        Number of delay embeddings (copies of the signal).
    lag : int
        Lag in samples between consecutive embeddings.
    """

    def __init__(self, order: int = 3, lag: int = 7) -> None:
        self.order = order
        self.lag = lag

    def fit(self, X: NDArray, y: NDArray | None = None) -> "AugmentedDataset":
        """No fitting required."""
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Apply Takens delay embedding.

        Parameters
        ----------
        X : NDArray, shape (n_epochs, n_channels, n_times)
            Input EEG epochs.

        Returns
        -------
        NDArray, shape (n_epochs, n_channels * order, n_times - (order-1)*lag)
            Augmented epochs with time-delayed channel copies.
        """
        n_epochs, n_channels, n_times = X.shape
        max_delay = (self.order - 1) * self.lag
        if max_delay >= n_times:
            raise ValueError(
                f"Delay embedding too large: (order-1)*lag = {max_delay} >= n_times = {n_times}"
            )
        trimmed_len = n_times - max_delay

        augmented = np.zeros((n_epochs, n_channels * self.order, trimmed_len))
        for k in range(self.order):
            offset = k * self.lag
            augmented[:, k * n_channels : (k + 1) * n_channels, :] = (
                X[:, :, offset : offset + trimmed_len]
            )
        return augmented


def build_acm_pipeline(order: int = 3, lag: int = 7) -> Pipeline:
    """Build an Augmented Covariance Method pipeline.

    Parameters
    ----------
    order : int
        Takens embedding order.
    lag : int
        Takens embedding lag (samples).

    Returns
    -------
    Pipeline
        ACM pipeline: AugmentedDataset → Covariances → TangentSpace → SVM.
    """
    return Pipeline([
        ("augment", AugmentedDataset(order=order, lag=lag)),
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_all_pipelines(sfreq: float = 500.0) -> dict[str, Pipeline]:
    """Build all classification pipelines in hackathon priority order.

    Run order:
      1. FBCSP+LDA  — primary
      2. ACM(3,7)   — secondary
      3. TS+LR      — reliable backup
      4. CSP+LDA    — baseline to beat
      5–8. Additional Riemannian pipelines for comparison

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz (needed for FBCSP filtering).

    Returns
    -------
    dict[str, Pipeline]
        Pipeline name → sklearn Pipeline (insertion-ordered).

    Notes
    -----
    Covariance estimators use 'oas' or 'lwf' — never 'scm'
    (ill-conditioned with 16 channels on stroke data).
    """
    from mne.decoding import CSP

    pipelines: dict[str, Pipeline] = {}

    # 1. FBCSP+LDA — primary (filter-bank decomposition)
    pipelines["FBCSP+LDA"] = build_fbcsp_pipeline(sfreq=sfreq)

    # 2. ACM(3,7) — secondary (temporal dynamics via delay embedding)
    pipelines["ACM(3,7)"] = build_acm_pipeline(order=3, lag=7)

    # 3. TS+LR — reliable Riemannian backup
    pipelines["TS+LR"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])

    # 4. CSP+LDA — baseline to beat
    pipelines["CSP+LDA"] = Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("lda", LDA()),
    ])

    # 5–8. Additional Riemannian pipelines for thorough comparison
    pipelines["FgMDM"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("fgmdm", FgMDM(metric="riemann")),
    ])

    pipelines["TS+SVM"] = Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("ts", TangentSpace(metric="riemann")),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])

    pipelines["MDM"] = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("mdm", MDM(metric="riemann")),
    ])

    pipelines["TS+LDA"] = Pipeline([
        ("cov", Covariances(estimator="lwf")),
        ("ts", TangentSpace(metric="riemann")),
        ("lda", LDA()),
    ])

    return pipelines


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    X: NDArray,
    y: NDArray,
    pipelines: dict[str, Pipeline],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate all pipelines with stratified k-fold cross-validation.

    Parameters
    ----------
    X : NDArray, shape (n_epochs, n_channels, n_times)
        Epoch data.
    y : NDArray, shape (n_epochs,)
        Labels.
    pipelines : dict[str, Pipeline]
        Pipeline name → sklearn Pipeline.
    n_splits : int
        Number of cross-validation folds.

    Returns
    -------
    pd.DataFrame
        Results sorted by mean accuracy descending, columns:
        pipeline, mean_acc, std_acc, fold_scores.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: list[dict[str, Any]] = []

    for name, pipe in pipelines.items():
        logger.info("Evaluating %s...", name)
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            results.append({
                "pipeline": name,
                "mean_acc": scores.mean(),
                "std_acc": scores.std(),
                "fold_scores": scores.tolist(),
            })
            logger.info("  %s: %.3f ± %.3f", name, scores.mean(), scores.std())
        except Exception as exc:
            logger.error("  %s failed: %s", name, exc)
            results.append({
                "pipeline": name,
                "mean_acc": np.nan,
                "std_acc": np.nan,
                "fold_scores": [],
            })

    df = pd.DataFrame(results).sort_values("mean_acc", ascending=False)
    df = df.reset_index(drop=True)
    return df
