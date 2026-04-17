"""Classification pipelines for stroke MI-BCI.

Riemannian geometry classifiers, CSP baselines, ACM (Augmented Covariance
Method), deep learning ensemble, and evaluation utilities.
"""

from __future__ import annotations

import logging
from typing import Any

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


def build_all_pipelines() -> dict[str, Pipeline]:
    """Build all classification pipelines for comparison.

    Returns
    -------
    dict[str, Pipeline]
        Pipeline name → sklearn Pipeline.

    Notes
    -----
    Covariance estimators use 'oas' or 'lwf' — never 'scm'
    (ill-conditioned with 16 channels on stroke data).
    """
    from mne.decoding import CSP

    pipelines: dict[str, Pipeline] = {}

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


def build_ensemble(
    X_train: NDArray,
    y_train: NDArray,
) -> Any:
    """Build and fit a soft voting ensemble.

    Combines TS+LR (weight 0.4), EEGNet (0.3), and ShallowConvNet (0.3)
    via weighted probability averaging.

    Parameters
    ----------
    X_train : NDArray, shape (n_epochs, n_channels, n_times)
        Training EEG epochs.
    y_train : NDArray, shape (n_epochs,)
        Training labels.

    Returns
    -------
    EnsembleClassifier
        Fitted ensemble with predict and predict_proba methods.
    """
    from braindecode.models import EEGNetv4, ShallowFBCSPNet
    from braindecode import EEGClassifier

    import torch

    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    n_classes = len(np.unique(y_train))

    # 1. Riemannian TS+LR
    ts_lr = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    ])

    # 2. EEGNet via braindecode
    eegnet_model = EEGNetv4(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
    )
    eegnet = EEGClassifier(
        module=eegnet_model,
        max_epochs=100,
        batch_size=32,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        optimizer__weight_decay=1e-4,
        train_split=None,
        verbose=0,
    )

    # 3. ShallowConvNet via braindecode
    shallow_model = ShallowFBCSPNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
    )
    shallow = EEGClassifier(
        module=shallow_model,
        max_epochs=100,
        batch_size=32,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        optimizer__weight_decay=1e-4,
        train_split=None,
        verbose=0,
    )

    ensemble = _EnsembleClassifier(
        estimators=[ts_lr, eegnet, shallow],
        weights=[0.4, 0.3, 0.3],
    )
    ensemble.fit(X_train, y_train)
    return ensemble


class _EnsembleClassifier(BaseEstimator):
    """Soft voting ensemble with weighted probability averaging.

    Parameters
    ----------
    estimators : list
        List of sklearn-compatible classifiers.
    weights : list[float]
        Weight for each estimator's predicted probabilities.
    """

    def __init__(
        self,
        estimators: list[Any] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        self.estimators = estimators or []
        self.weights = weights or [1.0 / len(self.estimators)] * len(self.estimators)

    def fit(self, X: NDArray, y: NDArray) -> "_EnsembleClassifier":
        """Fit all estimators."""
        self.classes_ = np.unique(y)
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        """Weighted average of predicted probabilities."""
        probas = []
        for est, w in zip(self.estimators, self.weights):
            if hasattr(est, "predict_proba"):
                probas.append(w * est.predict_proba(X))
            else:
                # Fall back to decision function for classifiers without predict_proba
                dec = est.decision_function(X)
                if dec.ndim == 1:
                    prob = np.column_stack([1 - dec, dec])
                else:
                    prob = dec
                prob = prob / prob.sum(axis=1, keepdims=True)
                probas.append(w * prob)
        return np.sum(probas, axis=0)

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels via argmax of averaged probabilities."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


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
